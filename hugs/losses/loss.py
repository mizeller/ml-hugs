#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
from lpips import LPIPS
import torch.nn as nn
import torch.nn.functional as F
from .utils import l1_loss, ssim 
from hugs.utils.rotations import quaternion_to_matrix

from hugs.utils.sampler import PatchSampler

from .utils import l1_loss, ssim
from hugs.utils.depth_utils import depth_to_normal
from hugs.models.gaussian_model import GaussianModel    

class ViewPointCamera:
    def __init__(self, data):
        self.image_width = data["image_width"]
        self.image_height = data["image_height"]
        self.FoVy = data["fovy"]
        self.FoVx = data["fovx"]
        self.znear = 0.01
        self.zfar = 100.0
        self.world_view_transform = data['world_view_transform']
        self.full_proj_transform = data['full_proj_transform']
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
    
class HumanSceneLoss(nn.Module):
    def __init__(
        self,
        l_ssim_w=0.2,
        l_l1_w=0.8,
        l_lpips_w=0.0,
        l_lbs_w=0.0,
        l_normal_w=0.0,
        l_humansep_w=0.0,
        l_dist_from_iter: int = 15000, # iteration after which distortion is added to loss    
        l_dist_w=0.0, 
        l_depth_normal_from_iter: int = 15000, # iteration after which depth normal is added to loss
        l_depth_normal_w=0.0,
        num_patches=4,
        patch_size=32,
        use_patches=True,
        bg_color='white',
    ):
        super(HumanSceneLoss, self).__init__()
        
        self.l_ssim_w = l_ssim_w
        self.l_l1_w = l_l1_w
        self.l_lpips_w = l_lpips_w
        self.l_lbs_w = l_lbs_w
        self.l_dist_from_iter = l_dist_from_iter
        self.l_dist_w = l_dist_w
        self.l_depth_normal_from_iter = l_depth_normal_from_iter
        self.l_normal_w = l_normal_w
        self.l_depth_normal_w = l_depth_normal_w
        self.l_humansep_w = l_humansep_w
        self.use_patches = use_patches
        
        self.bg_color = bg_color
        self.lpips = LPIPS(net="vgg", pretrained=True).to('cuda')
    
        for param in self.lpips.parameters(): param.requires_grad=False
        
        if self.use_patches:
            self.patch_sampler = PatchSampler(num_patch=num_patches, patch_size=patch_size, ratio_mask=0.9, dilate=0)
    
    def get_edge_aware_distortion_map(self, gt_image, distortion_map):
        # taken from GOF
        grad_img_left = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2]), 0)
        grad_img_right = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:]), 0)
        grad_img_top = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1]), 0)
        grad_img_bottom = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1]), 0)
        max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0]
        # pad
        max_grad = torch.exp(-max_grad)
        max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
        return distortion_map * max_grad
 
    def forward(
        self, 
        data, 
        render_pkg,
        human_gs, # TODO: fix this type (to GaussianModel), once the input is self.human_gs and not human_gs_out !!!
        render_mode, 
        human_gs_init_values=None,
        bg_color=None,
        human_bg_color=None,
        iteration: int = None
    ):

        # construct pseudo viewpoint_cam object to avoid 
        # having to change all subsequent GOF function calls
        viewpoint_cam = ViewPointCamera(data)

        loss_dict = {}
        extras_dict = {}
        
        if bg_color is not None:
            self.bg_color = bg_color
            
        if human_bg_color is None:
            human_bg_color = self.bg_color
            
        gt_image = data['rgb']
        mask = data['mask'].unsqueeze(0)
        rendering = render_pkg['render']
        pred_img = rendering[:3, :, :]
        norm_img = render_pkg['normal_img']

        extras_dict['gt_img'] = gt_image
        extras_dict['pred_img'] = pred_img
        extras_dict['normal_img'] = norm_img
        
        if render_mode == "human":
            gt_image = gt_image * mask + human_bg_color[:, None, None] * (1. - mask)
            extras_dict['gt_img'] = gt_image
        elif render_mode == "scene":
            # invert the mask
            mask = (1. - data['mask'].unsqueeze(0))
            gt_image = gt_image * mask
            extras_dict['gt_img'] = gt_image
            pred_img = pred_img * mask
 

        if self.l_l1_w > 0.0:
            if render_mode == "human":
                Ll1 = l1_loss(pred_img, gt_image, mask)
            elif render_mode == "scene":
                Ll1 = l1_loss(pred_img, gt_image, 1 - mask)
            elif render_mode == "human_scene":
                Ll1 = l1_loss(pred_img, gt_image)
            else:
                raise NotImplementedError
            loss_dict['l1'] = self.l_l1_w * Ll1 # rgb loss

        if self.l_ssim_w > 0.0:
            loss_ssim = 1.0 - ssim(pred_img, gt_image)
            if render_mode == "human":
                loss_ssim = loss_ssim * (mask.sum() / (pred_img.shape[-1] * pred_img.shape[-2]))
            elif render_mode == "scene":
                loss_ssim = loss_ssim * ((1 - mask).sum() / (pred_img.shape[-1] * pred_img.shape[-2]))
            elif render_mode == "human_scene":
                loss_ssim = loss_ssim
                
            loss_dict['ssim'] = self.l_ssim_w * loss_ssim
        
        if self.l_lpips_w > 0.0 and not render_mode == "scene":
            if self.use_patches:
                if render_mode == "human":
                    bg_color_lpips = torch.rand_like(pred_img)
                    image_bg = pred_img * mask + bg_color_lpips * (1. - mask)
                    gt_image_bg = gt_image * mask + bg_color_lpips * (1. - mask)
                    _, pred_patches, gt_patches = self.patch_sampler.sample(mask, image_bg, gt_image_bg)
                else:
                    _, pred_patches, gt_patches = self.patch_sampler.sample(mask, pred_img, gt_image)
                    
                loss_lpips = self.lpips(pred_patches.clip(max=1), gt_patches).mean()
                loss_dict['lpips_patch'] = self.l_lpips_w * loss_lpips
            else:
                bbox = data['bbox'].to(int)
                cropped_gt_image = gt_image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                cropped_pred_img = pred_img[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                loss_lpips = self.lpips(cropped_pred_img.clip(max=1), cropped_gt_image).mean()
                loss_dict['lpips'] = self.l_lpips_w * loss_lpips
                
        if self.l_humansep_w > 0.0 and render_mode == "human_scene":
            pred_human_img = render_pkg['human_img']
            gt_human_image = gt_image * mask + human_bg_color[:, None, None] * (1. - mask)
            
            Ll1_human = l1_loss(pred_human_img, gt_human_image, mask)
            loss_dict['l1_human'] = self.l_l1_w * Ll1_human * self.l_humansep_w
            
            loss_ssim_human = 1.0 - ssim(pred_human_img, gt_human_image)
            loss_ssim_human = loss_ssim_human * (mask.sum() / (pred_human_img.shape[-1] * pred_human_img.shape[-2]))
            loss_dict['ssim_human'] = self.l_ssim_w * loss_ssim_human * self.l_humansep_w
            
            bg_color_lpips = torch.rand_like(pred_human_img)
            image_bg = pred_human_img * mask + bg_color_lpips * (1. - mask)
            gt_image_bg = gt_human_image * mask + bg_color_lpips * (1. - mask)
            _, pred_patches, gt_patches = self.patch_sampler.sample(mask, image_bg, gt_image_bg)
            loss_lpips_human = self.lpips(pred_patches.clip(max=1), gt_patches).mean()
            loss_dict['lpips_patch_human'] = self.l_lpips_w * loss_lpips_human * self.l_humansep_w

        if self.l_lbs_w > 0.0 and human_gs.get_lbs_weights() is not None and not render_mode == "scene":
            if hasattr(human_gs, 'gt_lbs_weights'): # this attr is set in the forward pass in HUGS_MLP
                loss_lbs = F.mse_loss(
                    human_gs.get_lbs_weights(),
                    human_gs.gt_lbs_weights.detach()).mean()
            else:
                loss_lbs = F.mse_loss(
                    human_gs.get_lbs_weights(),
                    human_gs_init_values['lbs_weights']).mean()
            loss_dict['lbs'] = self.l_lbs_w * loss_lbs
       
        if self.l_normal_w > 0 and not render_mode == "scene":
            # edges = human_gaussians.edges
            if 'normals' in render_pkg.keys():
                gs_normals = human_gs['normals']
            else:
                rot_q = human_gs['rotq']
                rotmat = quaternion_to_matrix(rot_q)
                xyz = human_gs['xyz']
                gs_normals = torch.zeros_like(xyz).detach()
                gs_normals[:, 2] = 1.0
                
                gs_normals = (rotmat @ gs_normals.unsqueeze(-1)).squeeze(-1)
            
            loss_normal_reg = 1 - torch.cosine_similarity(human_gs['normals_canon'], 
                                                          human_gs_init_values['smpl_normals']).mean()
            loss_dict['normal_reg'] = self.l_normal_w * loss_normal_reg

        # NOTE: add Gaussian Opacity Field regularizers here; i.e. depth distortion 
        #       and depth normal consistency (initially only for scene rendering)
        if render_mode in ["scene", "human"]:
            # depth distortion regularization
            distortion_map = rendering[8, :, :]
            distortion_map = self.get_edge_aware_distortion_map(gt_image, distortion_map)
            distortion_loss = distortion_map.mean()
            
            # depth normal consistency
            depth = rendering[6, :, :]
            depth_normal, _ = depth_to_normal(viewpoint_cam, depth[None, ...])
            depth_normal = depth_normal.permute(2, 0, 1)

            render_normal =rendering[3:6, :, :]
            render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
           
            # TODO: extract viewpoint_cam from data 
            c2w = (viewpoint_cam.world_view_transform.T).inverse()
            normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
            render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
            
            normal_error = 1 - (render_normal_world * depth_normal).sum(dim=0)
            depth_normal_loss = normal_error.mean()
            # TODO: add weights for additional terms to config file 
            lambda_distortion = self.l_dist_w if iteration >= self.l_dist_from_iter else 0.0
            lambda_depth_normal = self.l_depth_normal_w if iteration >= self.l_depth_normal_from_iter else 0.0

            # add new regularization terms to loss_dict 
            loss_dict['distortion'] = lambda_distortion * distortion_loss
            loss_dict['depth_normal'] = lambda_depth_normal * depth_normal_loss
  
        loss = 0.0
        for k, v in loss_dict.items():
            loss += v
        
        return loss, loss_dict, extras_dict
    