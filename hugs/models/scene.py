# Code based on 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import os
import numpy as np
from torch import nn
from loguru import logger
from simple_knn._C import distCUDA2
from plyfile import PlyData, PlyElement

from hugs.utils.spherical_harmonics import RGB2SH
from hugs.utils.general import (
    get_expon_lr_func, 
    build_rotation,
)

# GOF
from hugs.scene.appearance_network import AppearanceNetwork
from hugs.utils.graphics_utils import BasicPointCloud
from hugs.models.gaussian_model import GaussianModel

class SceneGS(GaussianModel):
    def __init__(self, sh_degree: int):
        self.type: str = "scene"
        self.name: str = "SceneGS"
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
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
            "appearance_embeddings", 
            "appearance_network"
            ]
    
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
        try:
            self.optimizer.load_state_dict(opt_dict)
        except ValueError as e:
            logger.warning(f"Optimizer load failed: {e}")
            logger.warning("Continue without a pretrained optimizer")
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) # torch.Size([N_GAUSSIANS, 3])
   
    # @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation) # torch.Size([N_GAUSSIANS, 4])
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1) # torch.Size([N_GAUSSIANS, 16, 3])
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity) # torch.Size([N_GAUSSIANS, 1])
 
    def get_covariance(self, scaling_modifier = 1):
        # NOTE: unused; unused; see hugs/gaussian_renderer/__init__()
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation) # torch.Size([N_GAUSSIANS, 6])

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

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        logger.info(f'Number of scene points at initialisation: {fused_point_cloud.shape[0]}')

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def setup_optimizer(self, cfg):
        self.percent_dense = cfg.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
 
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
  
    @torch.no_grad()
    def get_tetra_points(self):
        import trimesh
        M = trimesh.creation.box()
        M.vertices *= 2
        
        rots = build_rotation(self._rotation)
        xyz = self.get_xyz
        scale = self.get_scaling_with_3D_filter * 3. 
        # filter points with small opacity for bicycle scene
        # opacity = self.get_opacity_with_3D_filter
        # mask = (opacity > 0.1).squeeze(-1)
        # xyz = xyz[mask]
        # scale = scale[mask]
        # rots = rots[mask]
        
        vertices = M.vertices.T    
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)
        
        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)
        
        return vertices, vertices_scale
   
    def reset_opacity(self):
        # reset opacity to by considering 3D filter
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.01)
        
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = self.inverse_opacity_activation(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

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
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

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

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[:update_filter.shape[0]][update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        #TODO maybe use max instead of average
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1