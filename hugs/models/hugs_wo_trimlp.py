#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
import os
import numpy as np
from torch import nn
from loguru import logger
from plyfile import PlyData, PlyElement
from pytorch3d.ops.knn import knn_points
import trimesh

from hugs.utils.general import (
    inverse_sigmoid, 
    get_expon_lr_func, 
    build_rotation,
)
from hugs.utils.rotations import (
    axis_angle_to_rotation_6d, 
    matrix_to_quaternion,  
    quaternion_multiply,
    quaternion_to_matrix, 
    rotation_6d_to_axis_angle, 
    torch_rotation_matrix_from_vectors,
)
from hugs.cfg.constants import SMPL_PATH
from hugs.utils.subdivide_smpl import subdivide_smpl_model

from .modules.smpl_layer import SMPL


# GOF
from hugs.scene.appearance_network import AppearanceNetwork
from hugs.utils.graphics_utils import BasicPointCloud

# GaussianModel(ABC)
from hugs.models.gaussian_model import GaussianModel

SCALE_Z = 1e-5


def batch_index_select(data, inds):
    bs, nv = data.shape[:2]
    device = data.device
    inds = inds + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    data = data.reshape(bs*nv, *data.shape[2:])
    return data[inds.long()]


def smpl_lbsmap_top_k(
        lbs_weights, 
        verts_transform, 
        points, 
        template_points,
        K=6, 
        addition_info=None
    ):
    '''ref: https://github.com/JanaldoChen/Anim-NeRF
    Args:  
    '''
    bz, np, _ = points.shape
    with torch.no_grad():
        results = knn_points(points, template_points, K=K)
        dists, idxs = results.dists, results.idx
    neighbs_dist = dists
    neighbs = idxs
    weight_std = 0.1
    weight_std2 = 2. * weight_std ** 2
    xyz_neighbs_lbs_weight = lbs_weights[neighbs] # (bs, n_rays*K, k_neigh, 24)
    xyz_neighbs_weight_conf = torch.exp(
        -torch.sum(
            torch.abs(xyz_neighbs_lbs_weight - xyz_neighbs_lbs_weight[..., 0:1, :]), dim=-1
        )/weight_std2) # (bs, n_rays*K, k_neigh)
    xyz_neighbs_weight_conf = torch.gt(xyz_neighbs_weight_conf, 0.9).float()
    xyz_neighbs_weight = torch.exp(-neighbs_dist) # (bs, n_rays*K, k_neigh)
    xyz_neighbs_weight *= xyz_neighbs_weight_conf
    xyz_neighbs_weight = xyz_neighbs_weight / xyz_neighbs_weight.sum(-1, keepdim=True) # (bs, n_rays*K, k_neigh)

    xyz_neighbs_transform = batch_index_select(verts_transform, neighbs) # (bs, n_rays*K, k_neigh, 4, 4)
    xyz_transform = torch.sum(xyz_neighbs_weight.unsqueeze(-1).unsqueeze(-1) * xyz_neighbs_transform, dim=2) # (bs, n_rays*K, 4, 4)
    xyz_dist = torch.sum(xyz_neighbs_weight * neighbs_dist, dim=2, keepdim=True) # (bs, n_rays*K, 1)

    if addition_info is not None: #[bz, nv, 3]
        xyz_neighbs_info = batch_index_select(addition_info, neighbs)
        xyz_info = torch.sum(xyz_neighbs_weight.unsqueeze(-1)* xyz_neighbs_info, dim=2) 
        return xyz_dist, xyz_transform, xyz_info
    else:
        return xyz_dist, xyz_transform
    

def smpl_lbsweight_top_k(
        lbs_weights, 
        points, 
        template_points, 
        K=6, 
    ):
    '''ref: https://github.com/JanaldoChen/Anim-NeRF
    Args:  
    '''
    bz, np, _ = points.shape
    with torch.no_grad():
        results = knn_points(points, template_points, K=K)
        dists, idxs = results.dists, results.idx
    neighbs_dist = dists
    neighbs = idxs
    weight_std = 0.1
    weight_std2 = 2. * weight_std ** 2
    xyz_neighbs_lbs_weight = lbs_weights[neighbs] # (bs, n_rays*K, k_neigh, 24)
    xyz_neighbs_weight_conf = torch.exp(
        -torch.sum(
            torch.abs(xyz_neighbs_lbs_weight - xyz_neighbs_lbs_weight[..., 0:1, :]), dim=-1
        )/weight_std2) # (bs, n_rays*K, k_neigh)
    xyz_neighbs_weight_conf = torch.gt(xyz_neighbs_weight_conf, 0.9).float()
    xyz_neighbs_weight = torch.exp(-neighbs_dist) # (bs, n_rays*K, k_neigh)
    xyz_neighbs_weight *= xyz_neighbs_weight_conf
    xyz_neighbs_weight = xyz_neighbs_weight / xyz_neighbs_weight.sum(-1, keepdim=True) # (bs, n_rays*K, k_neigh)

    # xyz_neighbs_transform = batch_index_select(verts_transform, neighbs) # (bs, n_rays*K, k_neigh, 4, 4)
    xyz_neighbs_lbs_weight = torch.sum(xyz_neighbs_weight.unsqueeze(-1) * xyz_neighbs_lbs_weight, dim=2) # (bs, n_rays*K, 4, 4)
    xyz_dist = torch.sum(xyz_neighbs_weight * neighbs_dist, dim=2, keepdim=True) # (bs, n_rays*K, 1)

    return xyz_dist, xyz_neighbs_lbs_weight


class HUGS_WO_TRIMLP(GaussianModel):
   
    def __init__(
        self, 
        cfg,
        init_betas = None,
        eval_mode: bool = False,
    ):
        self.type: str = 'human'
        self.name: str  = 'HUGS_WO_TRIMLP'
        self.active_sh_degree = 0
        self.max_sh_degree: int = cfg.sh_degree  
        self._xyz = torch.empty(0)
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
        self.setup_functions()
        # appearance network and appearance embedding
        self.appearance_network = AppearanceNetwork(3+64, 3).cuda()
        std = 1e-4
        self._appearance_embeddings = nn.Parameter(torch.empty(2048, 64).cuda())
        self._appearance_embeddings.data.normal_(0, std) 
        self.non_densify_params_keys = [
            'global_orient', 
            'body_pose', 
            'betas', 
            'transl',
            "appearance_embeddings", 
            "appearance_network"
            ]        

        # HUGS_WO_TRIMLP specific ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.device = 'cuda'
        self.use_surface: bool = cfg.use_surface
        self.init_2d: bool = cfg.init_2d
        self.rotate_sh: bool = cfg.rotate_sh
        self.isotropic: bool = cfg.isotropic
        self.init_scale_multiplier: float = cfg.init_scale_multiplier
        
        if cfg.n_subdivision > 0:
            logger.info(f"Subdividing SMPL model {cfg.n_subdivision} times")
            self.smpl_template = subdivide_smpl_model(smoothing=True, n_iter=cfg.n_subdivision).to(self.device)
        else:
            self.smpl_template = SMPL(SMPL_PATH).to(self.device)
            
        self.smpl = SMPL(SMPL_PATH).to(self.device)
            
        edges = trimesh.Trimesh(
            vertices=self.smpl_template.v_template.detach().cpu().numpy(), 
            faces=self.smpl_template.faces, process=False
        ).edges_unique
        self.edges = torch.from_numpy(edges).to(self.device).long()
        
        self.init_values = {}
        
        # init betas 
        self.create_betas(init_betas, cfg.optim_betas)

        # same for both human models 
        if not eval_mode:
            self.initialize()
    
    def capture(self):
        state_dict = {
            'active_sh_degree': self.active_sh_degree,
            'xyz': self._xyz,
            'features_dc': self._features_dc,
            'features_rest': self._features_rest,
            'scaling': self._scaling,
            'rotation': self._rotation,
            'opacity': self._opacity,
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'optimizer': self.optimizer.state_dict(),
            'spatial_lr_scale': self.spatial_lr_scale,
        }
        return state_dict
    
    def restore(self, state_dict, cfg):
        self.active_sh_degree = state_dict['active_sh_degree']
        self._xyz = state_dict['xyz']
        self._features_dc = state_dict['features_dc']
        self._features_rest = state_dict['features_rest']
        self._scaling = state_dict['scaling']
        self._rotation = state_dict['rotation']
        self._opacity = state_dict['opacity']
        self.max_radii2D = state_dict['max_radii2D']
        xyz_gradient_accum = state_dict['xyz_gradient_accum']
        denom = state_dict['denom']
        opt_dict = state_dict['optimizer']
        self.spatial_lr_scale = state_dict['spatial_lr_scale']
        
        self.setup_optimizer(cfg)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
   
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    # @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
   
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
   
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
   
    def get_view2gaussian(self, viewmatrix):
        # NOTE: unused; see hugs/gaussian_renderer/__init__()
        """
        Taken from GOF; this method is (currently) unused 
        (because pipe.compute_view2gaussian_python is set to False per default).
        For more information, refer to the original code; i.e to include it again...
        """
        r = self._rotation
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]
        
        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    
        rots = R
        xyz = self.get_xyz
        N = xyz.shape[0]
        G2W = torch.zeros((N, 4, 4), device='cuda')
        G2W[:, :3, :3] = rots # TODO check if we need to transpose here
        G2W[:, :3, 3] = xyz
        G2W[:, 3, 3] = 1.0
        
        viewmatrix = viewmatrix.transpose(0, 1)
        G2V = viewmatrix @ G2W
        
        R = G2V[:, :3, :3]
        t = G2V[:, :3, 3]
        
        t2 = torch.bmm(-R.transpose(1, 2), t[..., None])[..., 0]
        V2G = torch.zeros((N, 4, 4), device='cuda')
        V2G[:, :3, :3] = R.transpose(1, 2)
        V2G[:, :3, 3] = t2
        V2G[:, 3, 3] = 1.0
        
        # transpose view2gaussian to match glm in CUDA code
        V2G = V2G.transpose(2, 1).contiguous()
        return V2G 
    
    # not necessary for human case
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        pass
   
    def setup_optimizer(self, cfg):
        self.percent_dense = cfg.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.spatial_lr_scale = cfg.smpl_spatial
       
        params = [
            {'params': [self._xyz], 'lr': cfg.position_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': cfg.feature, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': cfg.feature / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': cfg.opacity, "name": "opacity"},
            {'params': [self._scaling], 'lr': cfg.scaling, "name": "scaling"},
            {'params': [self._rotation], 'lr': cfg.rotation, "name": "rotation"},
            {'params': [self._appearance_embeddings], 'lr': cfg.appearance_embeddings, "name": "appearance_embeddings"},
            {'params': self.appearance_network.parameters(), 'lr': cfg.appearance_network, "name": "appearance_network"}
        ]
        
        if hasattr(self, 'global_orient') and self.global_orient.requires_grad:
            params.append({'params': self.global_orient, 'lr': cfg.smpl_pose, 'name': 'global_orient'})
        
        if hasattr(self, 'body_pose') and self.body_pose.requires_grad:
            params.append({'params': self.body_pose, 'lr': cfg.smpl_pose, 'name': 'body_pose'})
            
        if hasattr(self, 'betas') and self.betas.requires_grad:
            params.append({'params': self.betas, 'lr': cfg.smpl_betas, 'name': 'betas'})
            
        if hasattr(self, 'transl') and self.betas.requires_grad:
            params.append({'params': self.transl, 'lr': cfg.smpl_trans, 'name': 'transl'})
        
        for param in params:
            logger.info(f"Parameter: {param['name']}, lr: {param['lr']}")

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=cfg.position_init * self.spatial_lr_scale,
            lr_final=cfg.position_final * self.spatial_lr_scale,
            lr_delay_mult=cfg.position_delay_mult,
            max_steps=cfg.position_max_steps,
        )
    
    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        filter_3D = self.filter_3D.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def save_fused_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # fuse opacity and scale
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities = self.inverse_opacity_activation(current_opacity_with_filter).detach().cpu().numpy()
        scale = self.scaling_inverse_activation(self.get_scaling_with_3D_filter).detach().cpu().numpy()
        
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=True)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    # `get_tetra_points()` is only used for GOF's rasterization, not implemented atm 
    
    # `reset_opacity()` not necessary for human case

    # unused method
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree
    
    def update_specific_attributes(self, optimizable_tensors, valid_points_mask):
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # NOTE: following 4 lines SceneGS specific
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)

         
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        self.densification_postfix(
            new_xyz, 
            new_features_dc, 
            new_features_rest, 
            new_opacity, 
            new_scaling, 
            new_rotation
            )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
   
    def densify_and_clone(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        grad_cond = torch.norm(grads, dim=-1) >= grad_threshold
        scale_cond = torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent
        selected_pts_mask = torch.where(grad_cond, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask, scale_cond)
        
        new_xyz = self._xyz[selected_pts_mask]
        # sample a new gaussian instead of fixing position
        stds = self.get_scaling[selected_pts_mask]
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
   
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_n_gs=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        max_n_gs = max_n_gs if max_n_gs else self.get_xyz.shape[0] + 1
        logger.warning(f"Max N_Gaussians vs N_Gaussians:\t{max_n_gs} vs. {self.get_xyz.shape[0]}") 
        if self.get_xyz.shape[0] <= max_n_gs:
            grads_abs = self.xyz_gradient_accum_abs / self.denom
            grads_abs[grads_abs.isnan()] = 0.0
            ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
            Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)
            
            self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
            self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
 
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        self.prune_points(prune_mask)
       
        self.n_gs = self.get_xyz.shape[0]
        torch.cuda.empty_cache()
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[:update_filter.shape[0]][update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        #TODO maybe use max instead of average
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1 
   
    # all methods above this line have been checked & look good ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # all methods below this line are specific for the human case ~~~~~~~~~~~~~~~~~~~~~~~~~~
   
    def create_body_pose(self, body_pose, requires_grad=False):
        body_pose = axis_angle_to_rotation_6d(body_pose.reshape(-1, 3)).reshape(-1, 23*6)
        self.body_pose = nn.Parameter(body_pose, requires_grad=requires_grad)
        logger.info(f"Created body pose with shape: {body_pose.shape}, requires_grad: {requires_grad}")
        
    def create_global_orient(self, global_orient, requires_grad=False):
        global_orient = axis_angle_to_rotation_6d(global_orient.reshape(-1, 3)).reshape(-1, 6)
        self.global_orient = nn.Parameter(global_orient, requires_grad=requires_grad)
        logger.info(f"Created global_orient with shape: {global_orient.shape}, requires_grad: {requires_grad}")
        
    def create_betas(self, betas, requires_grad=False):
        self.betas = nn.Parameter(betas, requires_grad=requires_grad)
        logger.info(f"Created betas with shape: {betas.shape}, requires_grad: {requires_grad}")
        
    def create_transl(self, transl, requires_grad=False):
        self.transl = nn.Parameter(transl, requires_grad=requires_grad)
        logger.info(f"Created transl with shape: {transl.shape}, requires_grad: {requires_grad}")
        
    def create_eps_offsets(self, eps_offsets, requires_grad=False):
        logger.info(f"NOT CREATED eps_offsets with shape: {eps_offsets.shape}, requires_grad: {requires_grad}")
    
    def forward(
        self,
        global_orient=None, 
        body_pose=None, 
        betas=None, 
        transl=None, 
        smpl_scale=None,
        dataset_idx=-1,
        ext_tfs=None,
    ):
        gs_scales = self.scaling_activation(self._scaling)
        gs_rotq = self.rotation_activation(self._rotation)
        gs_xyz = self._xyz
        gs_opacity = self.opacity_activation(self._opacity)
        gs_shs = self.get_features
        
        if self.isotropic:
            gs_scales = torch.ones_like(gs_scales) * torch.mean(gs_scales, dim=-1, keepdim=True)
            
        gs_scales_canon = gs_scales.clone()
        gs_rotmat = quaternion_to_matrix(gs_rotq)
        
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
        # remove & reapply the blendshape
        smpl_output = self.smpl(
            betas=betas.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            disable_posedirs=False,
        )
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
        
        return {
            'xyz': deformed_xyz,
            'xyz_canon': gs_xyz,
            'xyz_offsets': torch.zeros_like(gs_xyz),
            'scales': gs_scales,
            'scales_canon': gs_scales_canon,
            'rotq': deformed_gs_rotq,
            'rotq_canon': gs_rotq,
            'rotmat': deformed_gs_rotmat,
            'shs': deformed_gs_shs,
            'opacity': gs_opacity,
            'normals': deformed_normals,
            'normals_canon': canon_normals,
            'active_sh_degree': self.active_sh_degree,
        }

    @torch.no_grad()
    def _get_vitruvian_verts(self):
        vitruvian_pose = torch.zeros(69, dtype=self.smpl.dtype, device=self.device)
        vitruvian_pose[2] = 1.0
        vitruvian_pose[5] = -1.0
        smpl_output = self.smpl(body_pose=vitruvian_pose[None], betas=self.betas[None], disable_posedirs=False)
        vitruvian_verts = smpl_output.vertices[0]
        self.T_t2vitruvian = smpl_output.T[0].detach()
        self.inv_T_t2vitruvian = torch.inverse(self.T_t2vitruvian)
        self.canonical_offsets = smpl_output.shape_offsets + smpl_output.pose_offsets
        self.canonical_offsets = self.canonical_offsets[0]
        self.vitruvian_verts = vitruvian_verts.detach()
        return vitruvian_verts.detach()
    
    @torch.no_grad()
    def _get_vitruvian_verts_template(self):
        vitruvian_pose = torch.zeros(69, dtype=self.smpl.dtype, device=self.device)
        vitruvian_pose[2] = 1.0
        vitruvian_pose[5] = -1.0
        smpl_output = self.smpl_template(body_pose=vitruvian_pose[None], betas=self.betas[None], disable_posedirs=False)
        vitruvian_verts = smpl_output.vertices[0]
        return vitruvian_verts.detach()
    
    def initialize(self):
        self._get_vitruvian_verts()
        t_pose_verts = self._get_vitruvian_verts_template()
        
        xyz_offsets = torch.zeros_like(t_pose_verts) # NOTE: unused!
        colors = torch.ones_like(t_pose_verts) * 0.5
        
        shs = torch.zeros((colors.shape[0], 3, 16)).float().cuda()
        shs[:, :3, 0 ] = colors
        shs[:, 3:, 1:] = 0.0
        
        scales = torch.zeros_like(t_pose_verts)
        for v in range(t_pose_verts.shape[0]):
            selected_edges = torch.any(self.edges == v, dim=-1)
            selected_edges_len = torch.norm(
                t_pose_verts[self.edges[selected_edges][0]] - t_pose_verts[self.edges[selected_edges][1]], 
                dim=-1
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
            
        # only for this one we need log
        scales = torch.log(scales)
        
        import trimesh
        mesh = trimesh.Trimesh(vertices=t_pose_verts.detach().cpu().numpy(), faces=self.smpl_template.faces)
        vert_normals = torch.tensor(mesh.vertex_normals).float().cuda()
        
        gs_normals = torch.zeros_like(vert_normals)
        gs_normals[:, 2] = 1.0
        
        norm_rotmat = torch_rotation_matrix_from_vectors(gs_normals, vert_normals)

        rotq = matrix_to_quaternion(norm_rotmat)
                
        self.normals = gs_normals
        
        opacity = inverse_sigmoid(0.1 * torch.ones((t_pose_verts.shape[0], 1), dtype=torch.float, device="cuda"))

        self.n_gs = t_pose_verts.shape[0]
        self._xyz = nn.Parameter(t_pose_verts.requires_grad_(True))
        self._features_dc = nn.Parameter(shs[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(shs[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rotq.requires_grad_(True))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")