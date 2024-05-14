#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# commented methods -> unused
#

import os
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import torch
import trimesh
from torch import nn
from loguru import logger
import torch.nn.functional as F
from hugs.models.hugs_wo_trimlp import smpl_lbsmap_top_k, smpl_lbsweight_top_k
from hugs.cfg.config import cfg as default_cfg

from hugs.utils.general import (
    inverse_sigmoid,
    get_expon_lr_func,
)
from hugs.utils.rotations import (
    axis_angle_to_rotation_6d,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_multiply,
    quaternion_to_matrix,
    rotation_6d_to_axis_angle,
    rotation_6d_to_matrix,
    torch_rotation_matrix_from_vectors,
)
from hugs.cfg.constants import SMPL_PATH
from hugs.utils.subdivide_smpl import subdivide_smpl_model

from .modules.lbs import lbs_extra
from .modules.smpl_layer import SMPL
from .modules.triplane import TriPlane
from .modules.decoders import AppearanceDecoder, DeformationDecoder, GeometryDecoder


from hugs.models.gaussian_model import GaussianModel
from hugs.scene.appearance_network import AppearanceNetwork
from hugs.utils.graphics_utils import BasicPointCloud


SCALE_Z = 1e-5

class HUGS_TRIMLP(GaussianModel):

    def __init__(
        self,
        cfg = None,
        init_betas = None,
        eval_mode: bool = False
    ):
        self.type: str = 'human'
        self.name: str = "HUGS_TRIMLP"
        self.active_sh_degree = 0
        self.max_sh_degree = cfg.sh_degree
        self._xyz = torch.empty(0)

        # only this class doesn't have these members; 
        # as they are encoded in the triplane
        self.scaling_multiplier = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.device = "cuda"
        self.use_surface: bool = cfg.use_surface
        self.init_2d: bool = cfg.init_2d
        self.rotate_sh: bool = cfg.rotate_sh
        self.isotropic: bool = cfg.isotropic
        self.init_scale_multiplier: float = cfg.init_scale_multiplier
        self.use_deformer = cfg.use_deformer
        self.disable_posedirs = cfg.disable_posedirs

        self.n_features: int = 32 
        self.triplane = TriPlane(
            self.n_features, resX=cfg.triplane_res, resY=cfg.triplane_res, resZ=cfg.triplane_res
        ).to("cuda")
        self.appearance_dec = AppearanceDecoder(n_features=self.n_features * 3).to("cuda")
        self.deformation_dec = DeformationDecoder(
            n_features=self.n_features * 3, disable_posedirs=self.disable_posedirs
        ).to("cuda")
        self.geometry_dec = GeometryDecoder(
            n_features=self.n_features * 3, use_surface=self.use_surface
        ).to("cuda")

        # init betas 
        self.create_betas(init_betas, cfg.optim_betas)

        if cfg.n_subdivision > 0:
            logger.info(f"Subdividing SMPL model {cfg.n_subdivision} times")
            self.smpl_template = subdivide_smpl_model(
                smoothing=True, n_iter=cfg.n_subdivision
            ).to(self.device)
        else:
            self.smpl_template = SMPL(SMPL_PATH).to(self.device)

        self.smpl = SMPL(SMPL_PATH).to(self.device)

        edges = trimesh.Trimesh(
            vertices=self.smpl_template.v_template.detach().cpu().numpy(),
            faces=self.smpl_template.faces,
            process=False,
        ).edges_unique
        self.edges = torch.from_numpy(edges).to(self.device).long()

        self.init_values = {}
        self._get_vitruvian_verts()

        # appearance network and appearance embedding
        self.appearance_network = AppearanceNetwork(3+64, 3).cuda()
        std = 1e-4
        self._appearance_embeddings = nn.Parameter(torch.empty(2048, 64).cuda())
        self._appearance_embeddings.data.normal_(0, std)

        self.non_densify_params_keys = [
            "appearance_embeddings", 
            "appearance_network",
            "global_orient",
            "body_pose",
            "betas",
            "transl",
            "v_embed",
            "geometry_dec",
            "appearance_dec",
            "deform_dec",
        ]

        # same for both human models 
        if not eval_mode:
            self.initialize()

    def capture(self):
        save_dict = {
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz,
            # new; taken from scene gaussian ~~~~~~~~~~~~~
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "max_radii0D": self.max_radii2D,
            "scaling_multiplier": self.scaling_multiplier, 
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            "triplane": self.triplane.state_dict(),
            "appearance_dec": self.appearance_dec.state_dict(),
            "geometry_dec": self.geometry_dec.state_dict(),
            "deformation_dec": self.deformation_dec.state_dict(),
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "optimizer": self.optimizer.state_dict(),
            "spatial_lr_scale": self.spatial_lr_scale,
        }
        return save_dict

    def restore(self, state_dict, cfg=None):
        self.active_sh_degree = state_dict["active_sh_degree"]
        self._xyz = state_dict["xyz"]
        # new; taken from scene gaussian ~~~~~~~~~~~~~
        self._features_dc = state_dict['features_dc']
        self._features_rest = state_dict['features_rest']
        self._scaling = state_dict['scaling']
        self._rotation = state_dict['rotation']
        self._opacity = state_dict['opacity'] 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.max_radii2D = state_dict["max_radii2D"]
        xyz_gradient_accum = state_dict["xyz_gradient_accum"]
        denom = state_dict["denom"]
        opt_dict = state_dict["optimizer"]
        self.spatial_lr_scale = state_dict["spatial_lr_scale"]
        # old; HUGS TRIMLP specific stuff ~~~~~~~~~~~~~
        self.triplane.load_state_dict(state_dict["triplane"])
        self.appearance_dec.load_state_dict(state_dict["appearance_dec"])
        self.geometry_dec.load_state_dict(state_dict["geometry_dec"])
        self.deformation_dec.load_state_dict(state_dict["deformation_dec"])
        self.scaling_multiplier = state_dict["scaling_multiplier"]

        # not sure if this is necessary; possibly remove if case
        if cfg is None:
            from hugs.cfg.config import cfg as default_cfg
            cfg = default_cfg.human.lr

        self.setup_optimizer(cfg)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        try:
            self.optimizer.load_state_dict(opt_dict)
        except ValueError as e:
            logger.warning(f"Optimizer load failed: {e}")
            logger.warning("Continue without a pretrained optimizer")

    @property
    def get_scaling(self):
        tri_feats = self.triplane(self.get_xyz) # this breaks if xyz contains nan; should be filtered out...
        geometry_out = self.geometry_dec(tri_feats)
        scales = geometry_out["scales"] * self.scaling_multiplier 
        if self.isotropic:
            scales = torch.ones_like(scales) * torch.mean(
                scales, dim=-1, keepdim=True
            )
        return scales

    # @property 
    def get_rotation(self, rot6D: bool = False):
        """returns rotation quaternion; if rot6D is True, returns rotation 6D instead"""
        tri_feats = self.triplane(self.get_xyz)
        geometry_out = self.geometry_dec(tri_feats)
        rot6d = geometry_out["rotations"]
        if rot6D:
            return rot6d
        else:
            rotq = matrix_to_quaternion(rotation_6d_to_matrix(rot6d))
            return rotq
    @property
    def get_features(self):
        tri_feats = self.triplane(self.get_xyz)
        appearance_out = self.appearance_dec(tri_feats)
        shs = appearance_out["shs"].reshape(-1, 16, 3)
        # self._features_dc = shs[:, :1]
        # self._features_rest = shs[:, 1:]
        return shs
   
    @property
    def get_opacity(self):
        tri_feats = self.triplane(self.get_xyz)
        appearance_out = self.appearance_dec(tri_feats)
        opacity = appearance_out["opacity"]
        return opacity

    # def get_appearance_embedding(self, idx):
    #     pass

    @property
    def get_xyz_offset(self):
        tri_feats = self.triplane(self.get_xyz)
        geometry_out = self.geometry_dec(tri_feats)
        xyz_offsets = geometry_out["xyz"]
        return xyz_offsets # torch.Size([110210, 3])

    # @property
    def get_lbs_weights(self, use_softmax: bool = False):
        lbs_weights = None
        if self.use_deformer:
            tri_feats = self.triplane(self.get_xyz)
            deformation_out = self.deformation_dec(tri_feats)
            lbs_weights = deformation_out["lbs_weights"]
            if use_softmax:
                lbs_weights = F.softmax(lbs_weights / 0.1, dim=-1)
                if abs(lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                    pass
                else:
                    logger.warning(
                        f"LBS weights should sum to 1, but it is: {lbs_weights.sum(-1).mean().item()}"
                    )
        return lbs_weights
   
    @property
    def get_posedirs(self):
        posedirs = None
        if self.use_deformer:
            tri_feats = self.triplane(self.get_xyz)
            deformation_out = self.deformation_dec(tri_feats)
            posedirs = deformation_out["posedirs"]
        return posedirs

    # def reset_opacity(self):
    #     pass
    
    # def get_covariance(self, scaling_modifier = 1):
    #     pass 

    # def get_view2gaussian(self):
    #     pass 

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        # TODO - HELP: not sure how to implement this in the case of TRIMLP; in original implementation
        # features are derived from pcd....
        # perhaps, to make things easier, this can be skipped for now
        raise NotImplementedError

    # FIXME: this method needs to be aligned w/ SceneGS
    def setup_optimizer(self, cfg):
        self.percent_dense = cfg.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.spatial_lr_scale = cfg.smpl_spatial # human specific

        params = [
            {
                "params": [self._xyz],
                "lr": cfg.position_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            { 
                "params": self.triplane.parameters(), 
                "lr": cfg.vembed, 
                "name": "v_embed"
            },
            {
                "params": self.geometry_dec.parameters(),
                "lr": cfg.geometry,
                "name": "geometry_dec",
            },
            {
                "params": self.appearance_dec.parameters(),
                "lr": cfg.appearance,
                "name": "appearance_dec",
            },
            {
                "params": self.deformation_dec.parameters(),
                "lr": cfg.deformation,
                "name": "deform_dec",
            },
            {
                'params': [self._appearance_embeddings], 
                'lr': 0.001, # TODO: cfg.appearance_embeddings, 
                "name": "appearance_embeddings"
            },
            {
                'params': self.appearance_network.parameters(), 
                'lr': 0.001, # TODO: cfg.appearance_network, 
                "name": "appearance_network"
            }
        ]

        if hasattr(self, "global_orient") and self.global_orient.requires_grad:
            params.append(
                {
                    "params": self.global_orient,
                    "lr": cfg.smpl_pose,
                    "name": "global_orient",
                }
            )

        if hasattr(self, "body_pose") and self.body_pose.requires_grad:
            params.append(
                {"params": self.body_pose, "lr": cfg.smpl_pose, "name": "body_pose"}
            )

        if hasattr(self, "betas") and self.betas.requires_grad:
            params.append({"params": self.betas, "lr": cfg.smpl_betas, "name": "betas"})

        if hasattr(self, "transl") and self.betas.requires_grad:
            params.append(
                {"params": self.transl, "lr": cfg.smpl_trans, "name": "transl"}
            )

        for param in params:
            logger.info(f"Parameter: {param['name']}, lr: {param['lr']}")

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=cfg.position_init * self.spatial_lr_scale,
            lr_final=cfg.position_final * self.spatial_lr_scale,
            lr_delay_mult=cfg.position_delay_mult,
            max_steps=cfg.position_max_steps,
        )

    @torch.no_grad()
    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.get_xyz + self.xyz_offsets 
        xyz = xyz.cpu()
        normals = np.zeros_like(xyz)
        features_dc = self.get_features[:, :1].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        features_rest = self.get_features[:, 1:].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(self.get_opacity).cpu().numpy()
        scale = torch.log(self.get_scaling).cpu().numpy()
        rotation = self.get_rotation().cpu().numpy()

        filter_3D = self.filter_3D.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, features_dc, features_rest, opacities, scale, rotation, filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def save_fused_ply(self, path):
        pass
        
    @torch.no_grad()
    def get_tetra_points(self):
        # method unused; skip for now!
        pass
        
    def load_ply(self, path):
        pass
   
    def update_specific_attributes(self, optimizable_tensors, valid_points_mask):
        # TODO: align w/ correpsonding method in scene.py
        self.scaling_multiplier = self.scaling_multiplier[valid_points_mask]
        self.scales_tmp = self.scales_tmp[valid_points_mask] # update scales_tmp
        self.opacity_tmp = self.opacity_tmp[valid_points_mask]
        self.rotmat_tmp = self.rotmat_tmp[valid_points_mask] 
    
    def densification_postfix(
        self,
        new_xyz,
        new_scaling_multiplier,
        new_opacity_tmp,
        new_scales_tmp,
        new_rotmat_tmp,
    ):
        # TODO: align w/ densification_postfix in scene.py
        d = {
            "xyz": new_xyz,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self.scaling_multiplier = torch.cat(
            (self.scaling_multiplier, new_scaling_multiplier), dim=0
        )
        self.opacity_tmp = torch.cat([self.opacity_tmp, new_opacity_tmp], dim=0)
        self.scales_tmp = torch.cat([self.scales_tmp, new_scales_tmp], dim=0) # udpate scales_tmp
        self.rotmat_tmp = torch.cat([self.rotmat_tmp, new_rotmat_tmp], dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        scales = self.scales_tmp
        rotation = self.rotmat_tmp
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scales, dim=1).values > self.percent_dense * scene_extent,
        )
        # filter elongated gaussians
        med = scales.median(dim=1, keepdim=True).values
        stdmed_mask = (((scales - med) / med).squeeze(-1) >= 1.0).any(dim=-1)
        selected_pts_mask = torch.logical_and(selected_pts_mask, stdmed_mask)

        stds = scales[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=torch.relu(stds))
        rots = rotation[selected_pts_mask].repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling_multiplier = self.scaling_multiplier[selected_pts_mask].repeat(
            N, 1
        ) / (0.8 * N)
        new_opacity_tmp = self.opacity_tmp[selected_pts_mask].repeat(N, 1)
        new_scales_tmp = self.scales_tmp[selected_pts_mask].repeat(N, 1)
        new_rotmat_tmp = self.rotmat_tmp[selected_pts_mask].repeat(N, 1, 1)

        self.densification_postfix(
            new_xyz,
            new_scaling_multiplier,
            new_opacity_tmp,
            new_scales_tmp,
            new_rotmat_tmp,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        scales = self.scales_tmp
        scales = self.get_scaling
        
        
        grad_cond = torch.norm(grads, dim=-1) >= grad_threshold

        
        scale_cond = (
            torch.max(scales, dim=1).values <= self.percent_dense * scene_extent
        )

        selected_pts_mask = torch.where(grad_cond, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, scale_cond)

        new_xyz = self._xyz[selected_pts_mask]
        new_scaling_multiplier = self.scaling_multiplier[selected_pts_mask]
        new_opacity_tmp = self.opacity_tmp[selected_pts_mask]
        new_scales_tmp = self.scales_tmp[selected_pts_mask]
        new_rotmat_tmp = self.rotmat_tmp[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_scaling_multiplier,
            new_opacity_tmp,
            new_scales_tmp,
            new_rotmat_tmp,
        )

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        max_n_gs=None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.opacity_tmp = self.get_opacity
        self.scales_tmp = self.get_scaling

        max_n_gs = max_n_gs if max_n_gs else self.get_xyz.shape[0] + 1

        if self.get_xyz.shape[0] <= max_n_gs:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.opacity_tmp < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.scales_tmp.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
        self.n_gs = self.get_xyz.shape[0]
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[: update_filter.shape[0]][update_filter, :2],
            dim=-1,
            keepdim=True,
        )
        self.denom[update_filter] += 1

    # all methods below this line are specific for the human case ~~~~~~~~~~~~~~~~~~~~~~~~~~
    def create_body_pose(self, body_pose, requires_grad=False):
        body_pose = axis_angle_to_rotation_6d(body_pose.reshape(-1, 3)).reshape(
            -1, 23 * 6
        )
        self.body_pose = nn.Parameter(body_pose, requires_grad=requires_grad)
        logger.info(
            f"Created body pose with shape: {body_pose.shape}, requires_grad: {requires_grad}"
        )

    def create_global_orient(self, global_orient, requires_grad=False):
        global_orient = axis_angle_to_rotation_6d(global_orient.reshape(-1, 3)).reshape(
            -1, 6
        )
        self.global_orient = nn.Parameter(global_orient, requires_grad=requires_grad)
        logger.info(
            f"Created global_orient with shape: {global_orient.shape}, requires_grad: {requires_grad}"
        )

    def create_betas(self, betas, requires_grad=False):
        self.betas = nn.Parameter(betas, requires_grad=requires_grad)
        logger.info(
            f"Created betas with shape: {betas.shape}, requires_grad: {requires_grad}"
        )

    def create_transl(self, transl, requires_grad=False):
        self.transl = nn.Parameter(transl, requires_grad=requires_grad)
        logger.info(
            f"Created transl with shape: {transl.shape}, requires_grad: {requires_grad}"
        )

    def create_eps_offsets(self, eps_offsets, requires_grad=False):
        logger.info(
            f"NOT CREATED eps_offsets with shape: {eps_offsets.shape}, requires_grad: {requires_grad}"
        )
    
    def forward_old(
        self,
        global_orient=None, 
        body_pose=None, 
        betas=None, 
        transl=None, 
        smpl_scale=None,
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    ):
        
        tri_feats = self.triplane(self.get_xyz)
        appearance_out = self.appearance_dec(tri_feats)
        geometry_out = self.geometry_dec(tri_feats)
        
        xyz_offsets = geometry_out['xyz']
        gs_rot6d = geometry_out['rotations']
        gs_scales = geometry_out['scales'] * self.scaling_multiplier
        
        gs_xyz = self.get_xyz + xyz_offsets
        
        gs_rotmat = rotation_6d_to_matrix(gs_rot6d)
        gs_rotq = matrix_to_quaternion(gs_rotmat)

        gs_opacity = appearance_out['opacity']
        gs_shs = appearance_out['shs'].reshape(-1, 16, 3)
        
        if self.isotropic:
            gs_scales = torch.ones_like(gs_scales) * torch.mean(gs_scales, dim=-1, keepdim=True)
            
        gs_scales_canon = gs_scales.clone()
        
        if self.use_deformer:
            deformation_out = self.deformation_dec(tri_feats)
            lbs_weights = deformation_out['lbs_weights']
            lbs_weights = F.softmax(lbs_weights/0.1, dim=-1)
            posedirs = deformation_out['posedirs']
            if abs(lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                pass
            else:
                logger.warning(f"LBS weights should sum to 1, but it is: {lbs_weights.sum(-1).mean().item()}")
        else:
            lbs_weights = None
            posedirs = None
        
        if hasattr(self, 'global_orient') and global_orient is None:
            global_orient = rotation_6d_to_axis_angle(
                self.global_orient[dataset_idx].reshape(-1, 6)).reshape(3)
        
        if hasattr(self, 'body_pose') and body_pose is None:
            body_pose = rotation_6d_to_axis_angle(
                self.body_pose[dataset_idx].reshape(-1, 6)).reshape(23*3)
            
        if hasattr(self, 'betas') and betas is None:
            betas = self.betas
            
        if hasattr(self, 'transl') and transl is None:
            transl = self.transl[dataset_idx]
        
        # vitruvian -> t-pose -> posed
        # remove and reapply the blendshape
        smpl_output = self.smpl(
            betas=betas.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            disable_posedirs=False,
            return_full_pose=True,
        )
        
        gt_lbs_weights = None
        if self.use_deformer:
            A_t2pose = smpl_output.A[0]
            A_vitruvian2pose = A_t2pose @ self.inv_A_t2vitruvian
            deformed_xyz, _, lbs_T, _, _ = lbs_extra(
                A_vitruvian2pose[None], gs_xyz[None], posedirs, lbs_weights, 
                smpl_output.full_pose, disable_posedirs=self.disable_posedirs, pose2rot=True
            )
            deformed_xyz = deformed_xyz.squeeze(0)
            lbs_T = lbs_T.squeeze(0)

            with torch.no_grad():
                # gt lbs is needed for lbs regularization loss
                # predicted lbs should be close to gt lbs
                _, gt_lbs_weights = smpl_lbsweight_top_k(
                    lbs_weights=self.smpl.lbs_weights,
                    points=gs_xyz.unsqueeze(0),
                    template_points=self.vitruvian_verts.unsqueeze(0),
                )
                gt_lbs_weights = gt_lbs_weights.squeeze(0)
                if abs(gt_lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                    pass
                else:
                    logger.warning(f"GT LBS weights should sum to 1, but it is: {gt_lbs_weights.sum(-1).mean().item()}")
        else:
            curr_offsets = (smpl_output.shape_offsets + smpl_output.pose_offsets)[0]
            T_t2pose = smpl_output.T[0]
            T_vitruvian2t = self.inv_T_t2vitruvian.clone()
            T_vitruvian2t[..., :3, 3] = T_vitruvian2t[..., :3, 3] + self.canonical_offsets - curr_offsets
            T_vitruvian2pose = T_t2pose @ T_vitruvian2t

            _, lbs_T = smpl_lbsmap_top_k(
                lbs_weights=self.smpl.lbs_weights,
                verts_transform=T_vitruvian2pose.unsqueeze(0),
                points=gs_xyz.unsqueeze(0),
                template_points=self.vitruvian_verts.unsqueeze(0),
                K=6,
            )
            lbs_T = lbs_T.squeeze(0)
        
            homogen_coord = torch.ones_like(gs_xyz[..., :1])
            gs_xyz_homo = torch.cat([gs_xyz, homogen_coord], dim=-1)
            deformed_xyz = torch.matmul(lbs_T, gs_xyz_homo.unsqueeze(-1))[..., :3, 0]
        
        if smpl_scale is not None:
            deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)
        
        if transl is not None:
            deformed_xyz = deformed_xyz + transl.unsqueeze(0)
        
        deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)
        
        if ext_tfs is not None:
            tr, rotmat, sc = ext_tfs
            deformed_xyz = (tr[..., None] + (sc[None] * (rotmat @ deformed_xyz[..., None]))).squeeze(-1)
            gs_scales = sc * gs_scales
            
            rotq = matrix_to_quaternion(rotmat)
            deformed_gs_rotq = quaternion_multiply(rotq, deformed_gs_rotq)
            deformed_gs_rotmat = quaternion_to_matrix(deformed_gs_rotq)
        
        self.normals = torch.zeros_like(gs_xyz)
        self.normals[:, 2] = 1.0
        
        canon_normals = (gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        deformed_normals = (deformed_gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        
        deformed_gs_shs = gs_shs.clone()
       
       
        human_gs_out_dict = {
            'xyz': deformed_xyz,
            'xyz_canon': gs_xyz,
            'xyz_offsets': xyz_offsets,
            'scales': gs_scales,
            'scales_canon': gs_scales_canon,
            'rotq': deformed_gs_rotq,
            'rotq_canon': gs_rotq,
            'rotmat': deformed_gs_rotmat,
            'rotmat_canon': gs_rotmat,
            'shs': deformed_gs_shs,
            'opacity': gs_opacity,
            'normals': deformed_normals,
            'normals_canon': canon_normals,
            'active_sh_degree': self.active_sh_degree,
            'rot6d_canon': gs_rot6d,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
            'gt_lbs_weights': gt_lbs_weights,
        } 

        human_gs_out: HUGS_TRIMLP_MINIMAL = HUGS_TRIMLP_MINIMAL(human_gs_out_dict)
        
        self.forward(smpl_scale=smpl_scale,
                    dataset_idx=dataset_idx,
                    is_train=True,
                    ext_tfs=None)
        return human_gs_out 
        
        return {
            'xyz': deformed_xyz,
            'xyz_canon': gs_xyz,
            'xyz_offsets': xyz_offsets,
            'scales': gs_scales,
            'scales_canon': gs_scales_canon,
            'rotq': deformed_gs_rotq,
            'rotq_canon': gs_rotq,
            'rotmat': deformed_gs_rotmat,
            'rotmat_canon': gs_rotmat,
            'shs': deformed_gs_shs,
            'opacity': gs_opacity,
            'normals': deformed_normals,
            'normals_canon': canon_normals,
            'active_sh_degree': self.active_sh_degree,
            'rot6d_canon': gs_rot6d,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
            'gt_lbs_weights': gt_lbs_weights,
        }
        
    def forward(
        self,
        global_orient=None,
        body_pose=None,
        betas=None,
        transl=None,
        smpl_scale=None,
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    ):
        """refactored version of forward method"""
        gs_scales = self.get_scaling
        gs_xyz = self.get_xyz + self.get_xyz_offset
        gs_rotmat = rotation_6d_to_matrix(self.get_rotation(rot6D=True))
        gs_rotq = matrix_to_quaternion(gs_rotmat)
        lbs_weights = self.get_lbs_weights(use_softmax=True)
        posedirs = self.get_posedirs

        if hasattr(self, "global_orient") and global_orient is None:
            global_orient = rotation_6d_to_axis_angle(
                self.global_orient[dataset_idx].reshape(-1, 6)
            ).reshape(3)

        if hasattr(self, "body_pose") and body_pose is None:
            body_pose = rotation_6d_to_axis_angle(
                self.body_pose[dataset_idx].reshape(-1, 6)
            ).reshape(23 * 3)

        if hasattr(self, "betas") and betas is None:
            betas = self.betas

        if hasattr(self, "transl") and transl is None:
            transl = self.transl[dataset_idx]

        # vitruvian -> t-pose -> posed
        # remove and reapply the blendshape
        smpl_output = self.smpl(
            betas=betas.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            disable_posedirs=False,
            return_full_pose=True,
        )

        gt_lbs_weights = None
        if self.use_deformer:
            A_t2pose = smpl_output.A[0]
            A_vitruvian2pose = A_t2pose @ self.inv_A_t2vitruvian
            deformed_xyz, _, lbs_T, _, _ = lbs_extra(
                A_vitruvian2pose[None],
                gs_xyz[None],
                posedirs,
                lbs_weights,
                smpl_output.full_pose,
                disable_posedirs=self.disable_posedirs,
                pose2rot=True,
            )
            deformed_xyz = deformed_xyz.squeeze(0)
            lbs_T = lbs_T.squeeze(0)

            with torch.no_grad():
                # gt lbs is needed for lbs regularization loss
                # predicted lbs should be close to gt lbs
                _, gt_lbs_weights = smpl_lbsweight_top_k(
                    lbs_weights=self.smpl.lbs_weights,
                    points=gs_xyz.unsqueeze(0),
                    template_points=self.vitruvian_verts.unsqueeze(0),
                )
                gt_lbs_weights = gt_lbs_weights.squeeze(0)
                if abs(gt_lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                    pass
                else:
                    logger.warning(
                        f"GT LBS weights should sum to 1, but it is: {gt_lbs_weights.sum(-1).mean().item()}"
                    )
        else:
            curr_offsets = (smpl_output.shape_offsets + smpl_output.pose_offsets)[0]
            T_t2pose = smpl_output.T[0]
            T_vitruvian2t = self.inv_T_t2vitruvian.clone()
            T_vitruvian2t[..., :3, 3] = (
                T_vitruvian2t[..., :3, 3] + self.canonical_offsets - curr_offsets
            )
            T_vitruvian2pose = T_t2pose @ T_vitruvian2t

            _, lbs_T = smpl_lbsmap_top_k(
                lbs_weights=self.smpl.lbs_weights,
                verts_transform=T_vitruvian2pose.unsqueeze(0),
                points=gs_xyz.unsqueeze(0),
                template_points=self.vitruvian_verts.unsqueeze(0),
                K=6,
            )
            lbs_T = lbs_T.squeeze(0)

            homogen_coord = torch.ones_like(gs_xyz[..., :1])
            gs_xyz_homo = torch.cat([gs_xyz, homogen_coord], dim=-1)
            deformed_xyz = torch.matmul(lbs_T, gs_xyz_homo.unsqueeze(-1))[..., :3, 0]

        if smpl_scale is not None:
            deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)

        if transl is not None:
            deformed_xyz = deformed_xyz + transl.unsqueeze(0)

        deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)

        if ext_tfs is not None:
            tr, rotmat, sc = ext_tfs
            deformed_xyz = (
                tr[..., None] + (sc[None] * (rotmat @ deformed_xyz[..., None]))
            ).squeeze(-1)
            gs_scales = sc * gs_scales

            rotq = matrix_to_quaternion(rotmat)
            deformed_gs_rotq = quaternion_multiply(rotq, deformed_gs_rotq)
            deformed_gs_rotmat = quaternion_to_matrix(deformed_gs_rotq)

        self.normals = torch.zeros_like(gs_xyz)
        self.normals[:, 2] = 1.0

        canon_normals = (gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        deformed_normals = (deformed_gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)

        update_dict = {
            "xyz": deformed_xyz,
            "xyz_offsets": self.get_xyz_offset,
            "scales": gs_scales,
            "rotq": deformed_gs_rotq,
            "rotq_canon": gs_rotq,
            "rotmat": deformed_gs_rotmat,
            "shs": self.get_features.clone(),
            "opacity": self.get_opacity,
            "normals": deformed_normals,
            "normals_canon": canon_normals,
            "active_sh_degree": self.active_sh_degree,
            "rot6d_canon": self.get_rotation(rot6D=True),
            "lbs_weights": lbs_weights,
            "posedirs": posedirs,
            "gt_lbs_weights": gt_lbs_weights,
        }

        self.update_attributes(update_dict)
    
    # Example method to update attributes based on a dictionary (if needed later)
    def update_attributes(self, data_dict):
        for key, value in data_dict.items():
            setattr(self, key, value)

    @torch.no_grad()
    def _get_vitruvian_verts(self):
        "return vertices spread over: https://en.wikipedia.org/wiki/Vitruvian_Man"
        vitruvian_pose = torch.zeros(69, dtype=self.smpl.dtype, device=self.device)
        vitruvian_pose[2] = 1.0
        vitruvian_pose[5] = -1.0
        smpl_output = self.smpl(
            body_pose=vitruvian_pose[None],
            betas=self.betas[None],
            disable_posedirs=False,
        )
        vitruvian_verts = smpl_output.vertices[0]
        self.A_t2vitruvian = smpl_output.A[0].detach()
        self.T_t2vitruvian = smpl_output.T[0].detach()
        self.inv_T_t2vitruvian = torch.inverse(self.T_t2vitruvian)
        self.inv_A_t2vitruvian = torch.inverse(self.A_t2vitruvian)
        self.canonical_offsets = smpl_output.shape_offsets + smpl_output.pose_offsets
        self.canonical_offsets = self.canonical_offsets[0].detach()
        self.vitruvian_verts = vitruvian_verts.detach()
        return vitruvian_verts.detach()

    @torch.no_grad()
    def _get_vitruvian_verts_template(self):
        vitruvian_pose = torch.zeros(
            69, dtype=self.smpl_template.dtype, device=self.device
        )
        vitruvian_pose[2] = 1.0
        vitruvian_pose[5] = -1.0
        smpl_output = self.smpl_template(
            body_pose=vitruvian_pose[None],
            betas=self.betas[None],
            disable_posedirs=False,
        )
        vitruvian_verts = smpl_output.vertices[0]
        return vitruvian_verts.detach()

    def initialize(self):
        # in SceneGS, the corresponding method is called "create_from_pcd"
        t_pose_verts = self._get_vitruvian_verts_template()

        n_verts = t_pose_verts.shape[0]
        self.scaling_multiplier = torch.ones((n_verts, 1), device="cuda")  # N x 1

        xyz_offsets = torch.zeros_like(t_pose_verts)  # N x 3
        colors = torch.ones_like(t_pose_verts) * 0.5  # N x 3

        shs = torch.zeros((n_verts, 3, 16)).float().cuda()  # N x 3 x 16
        shs[:, :3, 0] = colors
        shs[:, 3:, 1:] = 0.0
        shs = shs.transpose(1, 2).contiguous()  # N x 16 x 3

        scales = torch.zeros_like(t_pose_verts)  # N x 3
        for v in range(n_verts): # slow
            selected_edges = torch.any(self.edges == v, dim=-1)
            selected_edges_len = torch.norm(
                t_pose_verts[self.edges[selected_edges][0]]
                - t_pose_verts[self.edges[selected_edges][1]],
                dim=-1,
            )
            selected_edges_len *= self.init_scale_multiplier
            scales[v, 0] = torch.log(torch.max(selected_edges_len))
            scales[v, 1] = torch.log(torch.max(selected_edges_len))

            if not self.use_surface:
                scales[v, 2] = torch.log(torch.max(selected_edges_len))

        if self.use_surface or self.init_2d:
            scales = scales[..., :2]

        scales = torch.exp(scales)

        if self.use_surface or self.init_2d:
            scale_z = torch.ones_like(scales[:, -1:]) * SCALE_Z
            scales = torch.cat([scales, scale_z], dim=-1)

        import trimesh

        mesh = trimesh.Trimesh(
            vertices=t_pose_verts.detach().cpu().numpy(), faces=self.smpl_template.faces
        ) 
        vert_normals = torch.tensor(mesh.vertex_normals).float().cuda()

        gs_normals = torch.zeros_like(vert_normals)  # N x 3
        gs_normals[:, 2] = 1.0

        norm_rotmat = torch_rotation_matrix_from_vectors(
            gs_normals, vert_normals
        )  # N x 3 x 3

        rotq = matrix_to_quaternion(norm_rotmat)  # N x 4
        rot6d = matrix_to_rotation_6d(norm_rotmat)  # N x 3 x 3

        self.normals = gs_normals  # QUESTION: why assign gs_normals to self.normals? they're all identical
        deformed_normals = (norm_rotmat @ gs_normals.unsqueeze(-1)).squeeze(-1)

        opacity = 0.1 * torch.ones((n_verts, 1), dtype=torch.float, device="cuda")

        posedirs = self.smpl_template.posedirs.detach().clone()
        lbs_weights = self.smpl_template.lbs_weights.detach().clone()

        self.n_gs = n_verts
        self._xyz = nn.Parameter(t_pose_verts.requires_grad_(True))
        # init these as well;
        self._features_dc = nn.Parameter(shs[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(shs[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rotq.requires_grad_(True))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self.max_radii2D = torch.zeros((n_verts), device="cuda")

        # optimize human gaussian model for 7k iterations
        logger.info("Optimizing initial human gaussian model for 7000 iterations.")

        _gt = {
            "xyz_offsets": xyz_offsets,
            "scales": scales,
            "rot6d_canon": rot6d,
            "shs": shs,
            "opacity": opacity,
            "lbs_weights": lbs_weights,
            "posedirs": posedirs,
            "deformed_normals": deformed_normals,
            "faces": self.smpl.faces_tensor,
            "edges": self.edges,
        }

        # TODO: num_steps should be 7k
        # self.optimize_init(self, num_steps=10, gt_vals=_gt)

    def optimize_init(
        self, lr: float = 1e-3, num_steps: int = 2000, gt_vals: dict = None
    ):

        lr = 1e-3
        default_cfg.human.lr.appearance = lr
        default_cfg.human.lr.geometry = lr
        default_cfg.human.lr.vembed = lr
        default_cfg.human.lr.deformation = 5e-4
        self.setup_optimizer(default_cfg.human.lr)
        optim = self.optimizer

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=1000, verbose=True, factor=0.5
        )

        fn = torch.nn.MSELoss()

        body_pose = torch.zeros((69)).to("cuda").float()
        global_orient = torch.zeros((3)).to("cuda").float()
        betas = torch.zeros((10)).to("cuda").float()

        print("===== Ground truth values: =====")
        for k, v in gt_vals.items():
            print(k, v.shape)
            gt_vals[k] = v.detach().clone().to("cuda").float()
        print("================================")

        losses = []

        pbar = tqdm(range(num_steps))
        for i in pbar:

            # remove this shit...
            model_out = {
                "xyz_offsets": self.get_xyz_offset,
                "scales": self.get_scaling,
                "rot6d_canon": self.get_rotation(rot6D=True),
                "shs": self.get_features,
                "opacity": self.get_opacity,
                "lbs_weights": self.get_lbs_weights(use_softmax=True),
                "posedirs": self.get_posedirs
            }
            # else: # case for HUGS_WO_TRIMLP
            #     model_out = self.forward(global_orient, body_pose, betas)

            if i % 1000 == 0: # why tho?
                continue

            loss_dict = {}
            for k, v in gt_vals.items():
                if k in ["faces", "deformed_normals", "edges"]:
                    continue
                if k in model_out:
                    if model_out[k] is not None:
                        loss_dict["loss_" + k] = fn(model_out[k], v)

            loss = sum(loss_dict.values())
            loss.backward()
            loss_str = ", ".join([f"{k}: {v.item():.7f}" for k, v in loss_dict.items()])
            pbar.set_description(f"Step {i:04d}: {loss.item():.7f} ({loss_str})")
            optim.step()
            optim.zero_grad(set_to_none=True)
            lr_scheduler.step(loss.item())

            losses.append(loss.item())

        return

class HUGS_TRIMLP_MINIMAL(HUGS_TRIMLP):
    def __init__(self, init_dict):
        for key, value in init_dict.items():
            setattr(self, key, value)