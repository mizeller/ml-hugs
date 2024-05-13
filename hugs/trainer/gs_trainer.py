#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import glob
import shutil
import random
import torch
import itertools
import torchvision
import numpy as np
from tqdm import tqdm
from lpips import LPIPS
from loguru import logger

from hugs.datasets.utils import (
    get_rotating_camera,
    get_smpl_static_params,
)
from hugs.losses.utils import ssim
from hugs.datasets import NeumanDataset
from hugs.losses.loss import HumanSceneLoss
from hugs.models.hugs_trimlp import HUGS_TRIMLP
from hugs.models.hugs_wo_trimlp import HUGS_WO_TRIMLP
from hugs.models import SceneGS
from hugs.utils.image import psnr, save_image
from hugs.utils.general import (
    RandomIndexIterator,
    save_images,
    create_video,
)

from hugs.renderer.gs_renderer import render_human_scene
from hugs import gaussian_renderer as GOF_renderer
from hugs.scene import Scene
from hugs.utils.depth_utils import depth_to_normal
from hugs.utils.vis_utils import apply_depth_colormap, colormap
from hugs.losses.loss import l1_loss

def get_train_dataset(cfg):
    if cfg.dataset.name == "neuman":
        logger.info(f"Loading NeuMan dataset {cfg.dataset.seq}-train")
        dataset = NeumanDataset(
            cfg.dataset.seq,
            "train",
            render_mode=cfg.mode,
            add_bg_points=cfg.scene.add_bg_points,
            num_bg_points=cfg.scene.num_bg_points,
            bg_sphere_dist=cfg.scene.bg_sphere_dist,
            clean_pcd=cfg.scene.clean_pcd,
        )

    return dataset

def get_val_dataset(cfg):
    if cfg.dataset.name == "neuman":
        logger.info(f"Loading NeuMan dataset {cfg.dataset.seq}-val")
        dataset = NeumanDataset(cfg.dataset.seq, "val", cfg.mode)

    return dataset

def get_anim_dataset(cfg):
    if cfg.dataset.name == "neuman":
        logger.info(f"Loading NeuMan dataset {cfg.dataset.seq}-anim")
        dataset = NeumanDataset(cfg.dataset.seq, "anim", cfg.mode)
    elif cfg.dataset.name == "zju":
        dataset = None

    return dataset

def L1_loss_appearance(image, gt_image, gaussians, view_idx, return_transformed_image=False):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]
    
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
    
    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    transformed_image = mapping_image * crop_image
    if not return_transformed_image:
        return l1_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
        return transformed_image

class GaussianTrainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        # get dataset
        if not cfg.eval:
            self.train_dataset = get_train_dataset(cfg)
        self.val_dataset = get_val_dataset(cfg)
        self.anim_dataset = get_anim_dataset(cfg)

        init_betas = self.val_dataset.cached_data[0]["betas"]
        
        self.eval_metrics = {}
        self.lpips = LPIPS().to("cuda")

        # init models
        self.human_gs, self.scene_gs = None, None

        if cfg.mode in ["human", "human_scene"]:
            if cfg.human.name == "hugs_wo_trimlp":
                self.human_gs = HUGS_WO_TRIMLP(
                    cfg=cfg.human,
                    init_betas=init_betas,
                    eval_mode=cfg.eval,
                )
            elif cfg.human.name == "hugs_trimlp":
                self.human_gs = HUGS_TRIMLP(
                    cfg=cfg.human,
                    init_betas=init_betas,
                    eval_mode=cfg.eval,
                )
            else:
                raise ValueError(f"Unknown human model {cfg.human.name}")
            
            self.scene = Scene(self.cfg, self.train_dataset, self.human_gs) 

        if cfg.mode in ["scene", "human_scene"]:
            self.scene_gs = SceneGS(
                sh_degree=cfg.scene.sh_degree,
            )
            # init scene object for to use GOF's loss terms
            self.scene = Scene(self.cfg, self.train_dataset, self.scene_gs)

        # setup the optimizers
        if self.human_gs:
            self.human_gs.setup_optimizer(cfg=cfg.human.lr)
            logger.info(self.human_gs)
            if cfg.human.ckpt:
                # load_human_ckpt(self.human_gs, cfg.human.ckpt)
                self.human_gs.load_state_dict(torch.load(cfg.human.ckpt))
                logger.info(f"Loaded human model from {cfg.human.ckpt}")
            else:
                ckpt_files = sorted(glob.glob(f"{cfg.logdir_ckpt}/*human*.pth"))
                if len(ckpt_files) > 0:
                    ckpt = torch.load(ckpt_files[-1])
                    self.human_gs.load_state_dict(ckpt)
                    logger.info(f"Loaded human model from {ckpt_files[-1]}")
                else:
                    print("Pseudo-Not-Implemented Error")
                    # raise NotImplementedError, "Implement the create_from_pcd() method in HUGS_TRIMLP class!!"
                #     pcd = self.train_dataset.init_pcd
                #     spatial_lr_scale = self.train_dataset.radius
                #     self.human_gs.create_from_pcd(pcd, spatial_lr_scale)

            if not cfg.eval:
                init_smpl_global_orient = torch.stack(
                    [x["global_orient"] for x in self.train_dataset.cached_data]
                )
                init_smpl_body_pose = torch.stack(
                    [x["body_pose"] for x in self.train_dataset.cached_data]
                )
                init_smpl_trans = torch.stack(
                    [x["transl"] for x in self.train_dataset.cached_data], dim=0
                )

                self.human_gs.create_body_pose(
                    init_smpl_body_pose, cfg.human.optim_pose
                )
                self.human_gs.create_global_orient(
                    init_smpl_global_orient, cfg.human.optim_pose
                )
                self.human_gs.create_transl(init_smpl_trans, cfg.human.optim_trans)


        if self.scene_gs:
            logger.info(self.scene_gs)
            if cfg.scene.ckpt:
                ckpt = torch.load(cfg.scene.ckpt)
                self.scene_gs.load_state_dict(ckpt, cfg.scene.lr)
                logger.info(f"Loaded scene model from {cfg.scene.ckpt}")
            else:
                ckpt_files = sorted(glob.glob(f"{cfg.logdir_ckpt}/*scene*.pth"))
                if len(ckpt_files) > 0:
                    ckpt = torch.load(ckpt_files[-1])
                    self.scene_gs.load_state_dict(ckpt, cfg.scene.lr)
                    logger.info(f"Loaded scene model from {cfg.scene.ckpt}")
                else:
                    pcd = self.train_dataset.init_pcd
                    spatial_lr_scale = self.train_dataset.radius
                    self.scene_gs.create_from_pcd(pcd, spatial_lr_scale)

            self.scene_gs.setup_optimizer(cfg=cfg.scene.lr)

        bg_color = cfg.bg_color
        if bg_color == "white":
            self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        elif bg_color == "black":
            self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        else:
            raise ValueError(f"Unknown background color {bg_color}")

        if cfg.mode in ["human", "human_scene"]:
            l = cfg.human.loss

            self.loss_fn = HumanSceneLoss(
                l_ssim_w=l.ssim_w,
                l_l1_w=l.l1_w,
                l_lpips_w=l.lpips_w,
                l_lbs_w=l.lbs_w,
                l_humansep_w=l.humansep_w,
                num_patches=l.num_patches,
                patch_size=l.patch_size,
                use_patches=l.use_patches,
                bg_color=self.bg_color,
            )
        else:
            self.cfg.train.optim_scene = True
            l = cfg.scene.loss
            self.loss_fn = HumanSceneLoss(
                l_ssim_w=l.ssim_w,
                l_l1_w=l.l1_w,
                l_dist_w=l.dist_w,
                l_dist_from_iter=l.dist_from_iter,
                l_depth_normal_w=l.depth_normal_w,
                l_depth_normal_from_iter=l.depth_normal_from_iter,
                bg_color=self.bg_color,
            )

        if cfg.mode in ["human", "human_scene"]:
            self.canon_camera_params = get_rotating_camera(
                dist=5.0,
                img_size=512,
                nframes=cfg.human.canon_nframes,
                device="cuda",
                angle_limit=2 * torch.pi,
            )
            betas = (
                self.human_gs.betas.detach()
                if hasattr(self.human_gs, "betas")
                else self.train_dataset.betas[0]
            )
            self.static_smpl_params = get_smpl_static_params(
                betas=betas, pose_type=self.cfg.human.canon_pose_type
            )

    def train(self):
        
        # GOF
        trainCameras = self.scene.getTrainCameras().copy()
        for idx, camera in enumerate(self.scene.getTrainCameras()):
            camera.idx = idx
        
        # init 3D_filter attr for scene & human gaussian splats
        if self.scene_gs:
            self.scene_gs.compute_3D_filter(cameras=trainCameras)
        if self.human_gs:
            self.human_gs.compute_3D_filter(cameras=trainCameras)

        pbar = tqdm(range(self.cfg.train.num_steps + 1), desc="Training")

        rand_idx_iter = RandomIndexIterator(len(self.train_dataset))
        sgrad_means, sgrad_stds = [], []
        render_mode = self.cfg.mode
        for t_iter in range(self.cfg.train.num_steps + 1):
            if self.scene_gs and self.cfg.train.optim_scene:
                self.scene_gs.update_learning_rate(t_iter)

            if hasattr(self.human_gs, "update_learning_rate"):
                self.human_gs.update_learning_rate(t_iter)

            rnd_idx = next(rand_idx_iter)
            data = self.train_dataset[rnd_idx]

            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    smpl_scale=data["smpl_scale"][None],
                    dataset_idx=rnd_idx,
                    is_train=True,
                    ext_tfs=None,
                )
            
            bg_color = torch.rand(3, dtype=torch.float32, device="cuda")

            if self.cfg.human.loss.humansep_w > 0.0 and render_mode == 'human_scene':
                render_human_separate = True
                human_bg_color = torch.rand(3, dtype=torch.float32, device="cuda")
            else:
                human_bg_color = None
                render_human_separate = False

            if render_mode != "scene":
                render_pkg = render_human_scene(
                                data=data,
                                human_gs_out=human_gs_out,
                                scene_gs=self.scene_gs,
                                bg_color=bg_color,
                                human_bg_color=human_bg_color,
                                render_mode=render_mode,
                                render_human_separate=render_human_separate,
                            )
            else:
                render_pkg = GOF_renderer.render_new(
                    data=data,
                    gaussians=self.scene_gs,
                    bg_color=bg_color
                )

            if self.human_gs:
                self.human_gs.init_values["edges"] = self.human_gs.edges

            loss, loss_dict, loss_extras = self.loss_fn(
                data,
                render_pkg,
                None, # human_gs_out, # BUG: add the variable back later!
                render_mode=render_mode,
                human_gs_init_values=(
                    self.human_gs.init_values if self.human_gs else None
                ),
                bg_color=bg_color,
                human_bg_color=human_bg_color,
                iteration = t_iter
            )

            loss.backward()

            loss_dict["loss"] = loss

            # ~~~~ save images (GOF way) ~~~~
            if render_mode in ["scene"]: # TODO: add human
                # TODO: extend this method to handle the different cases of HUGS
                gaussians = self.human_gs if self.human_gs else self.scene_gs
                is_save_images = False
                if is_save_images and (t_iter % self.cfg.scene.densification_interval == 0):
                    with torch.no_grad():
                        eval_cam = trainCameras[random.randint(0, len(trainCameras) -1)]
                        rendering = GOF_renderer.render(eval_cam, gaussians, bg_color)["render"]
                        image = rendering[:3, :, :]
                        transformed_image = L1_loss_appearance(image, eval_cam.original_image.cuda(), gaussians, eval_cam.idx, return_transformed_image=True)
                        normal = rendering[3:6, :, :]
                        normal = torch.nn.functional.normalize(normal, p=2, dim=0)
                        
                    # transform to world space
                    c2w = (eval_cam.world_view_transform.T).inverse()
                    normal2 = c2w[:3, :3] @ normal.reshape(3, -1)
                    normal = normal2.reshape(3, *normal.shape[1:])
                    normal = (normal + 1.) / 2.
                    
                    depth = rendering[6, :, :]
                    depth_normal, _ = depth_to_normal(eval_cam, depth[None, ...])
                    depth_normal = (depth_normal + 1.) / 2.
                    depth_normal = depth_normal.permute(2, 0, 1)
                    
                    gt_image = eval_cam.original_image.cuda()
                    
                    depth_map = apply_depth_colormap(depth[..., None], rendering[7, :, :, None], near_plane=None, far_plane=None)
                    depth_map = depth_map.permute(2, 0, 1)
                    
                    accumlated_alpha = rendering[7, :, :, None]
                    colored_accum_alpha = apply_depth_colormap(accumlated_alpha, None, near_plane=0.0, far_plane=1.0)
                    colored_accum_alpha = colored_accum_alpha.permute(2, 0, 1)
                    
                    distortion_map = rendering[8, :, :]
                    distortion_map = colormap(distortion_map.detach().cpu().numpy()).to(normal.device)
                
                    row0 = torch.cat([gt_image, image, depth_normal, normal], dim=2)
                    row1 = torch.cat([depth_map, colored_accum_alpha, distortion_map, transformed_image], dim=2)
                    
                    image_to_show = torch.cat([row0, row1], dim=1)
                    image_to_show = torch.clamp(image_to_show, 0, 1)
                    
                    os.makedirs(f"{self.cfg.logdir}/log_images", exist_ok = True)
                    torchvision.utils.save_image(image_to_show, f"{self.cfg.logdir}/log_images/{t_iter}.jpg")

                # save pointcloud for subsequent mesh extraction
                if render_mode == "scene" and t_iter in [1_000, 5_000, 15_000]:
                    print("\n[ITER {}] Saving Gaussians".format(t_iter))
                    self.scene.save(t_iter)
            # ~~~~ save images (GOF way) ~~~~
            if t_iter % 10 == 0:
                postfix_dict = {
                    "#hp": f"{self.human_gs.n_gs/1000 if self.human_gs else 0:.1f}K",
                    "#sp": f"{self.scene_gs.get_xyz.shape[0]/1000 if self.scene_gs else 0:.1f}K",
                    "h_sh_d": self.human_gs.active_sh_degree if self.human_gs else 0,
                    "s_sh_d": self.scene_gs.active_sh_degree if self.scene_gs else 0,
                }
                for k, v in loss_dict.items():
                    postfix_dict["l_" + k] = f"{v.item():.4f}"

                pbar.set_postfix(postfix_dict)
                pbar.update(10)

            if t_iter == self.cfg.train.num_steps:
                pbar.close()

            if t_iter % 10 == 0: # NOTE: change viz_interval here
                with torch.no_grad():
                    pred_img = loss_extras["pred_img"]
                    gt_img = loss_extras["gt_img"]
                    log_pred_img = (
                        pred_img.cpu().numpy().transpose(1, 2, 0) * 255
                    ).astype(np.uint8)
                    log_gt_img = (gt_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(
                        np.uint8
                    )
                    log_img = np.concatenate([log_gt_img, log_pred_img], axis=1)
                    save_images(log_img, f"{self.cfg.logdir}/train/{t_iter:06d}.png")

            if t_iter >= self.cfg.scene.opt_start_iter:
                if (
                    t_iter - self.cfg.scene.opt_start_iter
                ) < self.cfg.scene.densify_until_iter and self.cfg.mode in [
                    "scene",
                    "human_scene",
                ]:
                    render_pkg["scene_viewspace_points"] = render_pkg[
                        "viewspace_points"
                    ]
                    render_pkg["scene_viewspace_points"].grad = render_pkg[
                        "viewspace_points"
                    ].grad

                    sgrad_mean, sgrad_std = (
                        render_pkg["scene_viewspace_points"].grad.mean(),
                        render_pkg["scene_viewspace_points"].grad.std(),
                    )
                    sgrad_means.append(sgrad_mean.item())
                    sgrad_stds.append(sgrad_std.item())
                    with torch.no_grad():
                        self.scene_densification(
                            visibility_filter=render_pkg["visibility_filter"],
                            radii=render_pkg["radii"],
                            viewspace_point_tensor=render_pkg["scene_viewspace_points"],
                            iteration=(t_iter - self.cfg.scene.opt_start_iter) + 1,
                            trainCameras=trainCameras
                        )

            if t_iter < self.cfg.human.densify_until_iter and self.cfg.mode in [
                "human",
                "human_scene",
            ]:
                render_pkg["human_viewspace_points"] = render_pkg["viewspace_points"][
                    : human_gs_out["xyz"].shape[0]
                ]
                render_pkg["human_viewspace_points"].grad = render_pkg[
                    "viewspace_points"
                ].grad[: human_gs_out["xyz"].shape[0]]
                with torch.no_grad():
                    self.human_densification(
                        human_gs_out=human_gs_out,
                        visibility_filter=render_pkg["human_visibility_filter"],
                        radii=render_pkg["human_radii"],
                        viewspace_point_tensor=render_pkg["human_viewspace_points"],
                        iteration=t_iter + 1,
                    )

            if self.human_gs:
                self.human_gs.optimizer.step()
                self.human_gs.optimizer.zero_grad(set_to_none=True)

            # save checkpoint
            if (t_iter % self.cfg.train.save_ckpt_interval == 0 and t_iter > 0) or (
                t_iter == self.cfg.train.num_steps and t_iter > 0
            ):
                self.save_ckpt(t_iter)

            # run validation
            if t_iter % self.cfg.train.val_interval == 0 and t_iter > 0:
                self.validate(t_iter)

            if t_iter == 0:
                if self.scene_gs:
                    self.scene_gs.save_ply(
                        f"{self.cfg.logdir}/meshes/scene_{t_iter:06d}_splat.ply"
                    )
                if self.human_gs:
                    self.human_gs.save_ply(
                        human_gs_out,
                        f"{self.cfg.logdir}/meshes/human_{t_iter:06d}_splat.ply",
                    )

                if self.cfg.mode in ["human", "human_scene"]:
                    self.render_canonical(t_iter, nframes=self.cfg.human.canon_nframes)

            if (
                t_iter % self.cfg.train.anim_interval == 0
                and t_iter > 0
                and self.cfg.train.anim_interval > 0
            ):
                if self.human_gs:
                    self.human_gs.save_ply(
                        human_gs_out,
                        f"{self.cfg.logdir}/meshes/human_{t_iter:06d}_splat.ply",
                    )
                if self.anim_dataset is not None:
                    self.animate(t_iter)

                if self.cfg.mode in ["human", "human_scene"]:
                    self.render_canonical(t_iter, nframes=self.cfg.human.canon_nframes)

            if t_iter % 1000 == 0 and t_iter > 0:
                if self.human_gs: self.human_gs.oneupSHdegree()
                if self.scene_gs: self.scene_gs.oneupSHdegree()

            if (
                self.cfg.train.save_progress_images
                and t_iter % self.cfg.train.progress_save_interval == 0
                and self.cfg.mode in ["human", "human_scene"]
            ):
                self.render_canonical(t_iter, nframes=2, is_train_progress=True)
            
            if self.scene_gs and t_iter % 100 == 0 and t_iter > self.cfg.scene.densify_until_iter:
                if t_iter < self.cfg.train.num_steps - 100:
                    # don't update in the end of training
                    self.scene_gs.compute_3D_filter(cameras=trainCameras) 

            # Optimizer step
            if self.scene_gs and self.cfg.train.optim_scene:
                if t_iter >= self.cfg.scene.opt_start_iter:
                    self.scene_gs.optimizer.step()
                    self.scene_gs.optimizer.zero_grad(set_to_none=True)

        # train progress images (only for human, human_scene modes)
        if self.cfg.train.save_progress_images:
            video_fname = f"{self.cfg.logdir}/train_{self.cfg.dataset.name}_{self.cfg.dataset.seq}.mp4"
            create_video(f"{self.cfg.logdir}/train_progress/", video_fname, fps=10)
            shutil.rmtree(f"{self.cfg.logdir}/train_progress/")

    def save_ckpt(self, iter=None):

        iter_s = "final" if iter is None else f"{iter:06d}"

        if self.human_gs:
            torch.save(
                self.human_gs.state_dict(), f"{self.cfg.logdir_ckpt}/human_{iter_s}.pth"
            )

        if self.scene_gs:
            torch.save(
                self.scene_gs.state_dict(), f"{self.cfg.logdir_ckpt}/scene_{iter_s}.pth"
            )
            self.scene_gs.save_ply(f"{self.cfg.logdir}/meshes/scene_{iter_s}_splat.ply")

        logger.info(f"Saved checkpoint {iter_s}")

    def scene_densification(
        self, visibility_filter, radii, viewspace_point_tensor, iteration, trainCameras
    ):
        self.scene_gs.max_radii2D[visibility_filter] = torch.max(
            self.scene_gs.max_radii2D[visibility_filter], radii[visibility_filter]
        )
        self.scene_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if (
            iteration > self.cfg.scene.densify_from_iter
            and iteration % self.cfg.scene.densification_interval == 0
        ):
            size_threshold = (
                20 if iteration > self.cfg.scene.opacity_reset_interval else None
            )
            self.scene_gs.densify_and_prune(
                self.cfg.scene.densify_grad_threshold,
                min_opacity=self.cfg.scene.prune_min_opacity,
                extent=self.train_dataset.radius,
                max_screen_size=size_threshold,
                max_n_gs=self.cfg.scene.max_n_gaussians,
            )
            self.scene_gs.compute_3D_filter(cameras=trainCameras)

        is_white = self.bg_color.sum().item() == 3.0

        if iteration % self.cfg.scene.opacity_reset_interval == 0 or (
            is_white and iteration == self.cfg.scene.densify_from_iter
        ):
            logger.info(f"[{iteration:06d}] Resetting opacity!!!")
            self.scene_gs.reset_opacity()

    def human_densification(self, human_gs_out, visibility_filter, radii, viewspace_point_tensor, iteration):
        self.human_gs.max_radii2D[visibility_filter] = torch.max(
            self.human_gs.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
        
        self.human_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if (
            iteration > self.cfg.human.densify_from_iter
            and iteration % self.cfg.human.densification_interval == 0
        ):
            size_threshold = 20
            self.human_gs.densify_and_prune(
                min_opacity=self.cfg.human.prune_min_opacity, 
                extent=self.cfg.human.densify_extent, 
                max_screen_size=size_threshold,
                max_n_gs=self.cfg.human.max_n_gaussians,
            )

    @torch.no_grad()
    def validate(self, iter=None):

        iter_s = "final" if iter is None else f"{iter:06d}"

        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")

        methods = ["hugs", "hugs_human"]
        metrics = ["lpips", "psnr", "ssim"]
        metrics = dict.fromkeys(
            ["_".join(x) for x in itertools.product(methods, metrics)]
        )
        metrics = {k: [] for k in metrics}

        for idx, data in enumerate(tqdm(self.val_dataset, desc="Validation")):
            render_mode = self.cfg.mode

            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    global_orient=data["global_orient"],
                    body_pose=data["body_pose"],
                    betas=data["betas"],
                    transl=data["transl"],
                    smpl_scale=data["smpl_scale"][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )

                # TODO: at this point, the GOF render method should be used as well..
                render_pkg = render_human_scene(
                    data=data,
                    human_gs_out=human_gs_out,
                    scene_gs=self.scene_gs,
                    bg_color=bg_color,
                    render_mode=render_mode,
                )

                gt_image = data["rgb"]

                image = render_pkg["render"]
            
            if self.scene_gs:
                render_pkg = GOF_renderer.render_new(
                    data=data,
                    gaussians=self.scene_gs,
                    bg_color=bg_color
                )
                image = render_pkg["render"][:3, :, :]
                gt_image = data["rgb"]            

            if self.cfg.dataset.name == "zju":
                image = image * data["mask"]
                gt_image = gt_image * data["mask"]

            metrics["hugs_psnr"].append(psnr(image, gt_image).mean().double())
            metrics["hugs_ssim"].append(ssim(image, gt_image).mean().double())
            metrics["hugs_lpips"].append(
                self.lpips(image.clip(max=1), gt_image).mean().double()
            )

            log_img = torchvision.utils.make_grid(
                [gt_image, image], nrow=2, pad_value=1
            )
            imf = f"{self.cfg.logdir}/val/full_{iter_s}_{idx:03d}.png"
            os.makedirs(os.path.dirname(imf), exist_ok=True)
            torchvision.utils.save_image(log_img, imf)

            log_img = []
            if self.cfg.mode in ["human", "human_scene"]:
                bbox = data["bbox"].to(int)
                cropped_gt_image = gt_image[:, bbox[0] : bbox[2], bbox[1] : bbox[3]]
                cropped_image = image[:, bbox[0] : bbox[2], bbox[1] : bbox[3]]
                log_img += [cropped_gt_image, cropped_image]

                metrics["hugs_human_psnr"].append(
                    psnr(cropped_image, cropped_gt_image).mean().double()
                )
                metrics["hugs_human_ssim"].append(
                    ssim(cropped_image, cropped_gt_image).mean().double()
                )
                metrics["hugs_human_lpips"].append(
                    self.lpips(cropped_image.clip(max=1), cropped_gt_image)
                    .mean()
                    .double()
                )

            if len(log_img) > 0:
                log_img = torchvision.utils.make_grid(
                    log_img, nrow=len(log_img), pad_value=1
                )
                torchvision.utils.save_image(
                    log_img, f"{self.cfg.logdir}/val/human_{iter_s}_{idx:03d}.png"
                )

        self.eval_metrics[iter_s] = {}

        for k, v in metrics.items():
            if v == []:
                continue

            logger.info(f"{iter_s} - {k.upper()}: {torch.stack(v).mean().item():.4f}")
            self.eval_metrics[iter_s][k] = torch.stack(v).mean().item()

        torch.save(metrics, f"{self.cfg.logdir}/val/eval_{iter_s}.pth")

    @torch.no_grad()
    def animate(self, iter=None, keep_images=False):
        if self.anim_dataset is None:
            logger.info("No animation dataset found")
            return 0

        iter_s = "final" if iter is None else f"{iter:06d}"

        os.makedirs(f"{self.cfg.logdir}/anim/", exist_ok=True)

        for idx, data in enumerate(tqdm(self.anim_dataset, desc="Animation")):
            if self.human_gs:
                ext_tfs = (
                    data["manual_trans"],
                    data["manual_rotmat"],
                    data["manual_scale"],
                )
                human_gs_out = self.human_gs.forward(
                    global_orient=data["global_orient"],
                    body_pose=data["body_pose"],
                    betas=data["betas"],
                    transl=data["transl"],
                    smpl_scale=data["smpl_scale"][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=ext_tfs,
                )


            if self.cfg.mode == "scene": 
                render_pkg = GOF_renderer.render_new(
                    data=data,
                    gaussians=self.scene_gs,
                    bg_color=self.bg_color,
                )
                image = render_pkg["render"][:3, :, :]
            else:
                render_pkg = render_human_scene(
                    data=data,
                    human_gs_out=human_gs_out,
                    scene_gs=self.scene_gs,
                    bg_color=self.bg_color,
                    render_mode=self.cfg.mode,
                )
                image = render_pkg["render"]

            torchvision.utils.save_image(image, f"{self.cfg.logdir}/anim/{idx:05d}.png")

        video_fname = f"{self.cfg.logdir}/anim_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4"
        create_video(f"{self.cfg.logdir}/anim/", video_fname, fps=20)
        if not keep_images:
            shutil.rmtree(f"{self.cfg.logdir}/anim/")
            os.makedirs(f"{self.cfg.logdir}/anim/")

    @torch.no_grad()
    def render_canonical(
        self, iter=None, nframes=100, is_train_progress=False, pose_type=None
    ):
        iter_s = "final" if iter is None else f"{iter:06d}"
        iter_s += f"_{pose_type}" if pose_type is not None else ""

        os.makedirs(f"{self.cfg.logdir}/canon/", exist_ok=True)

        camera_params = get_rotating_camera(
            dist=5.0,
            img_size=256 if is_train_progress else 512,
            nframes=nframes,
            device="cuda",
            angle_limit=torch.pi if is_train_progress else 2 * torch.pi,
        )

        betas = (
            self.human_gs.betas.detach()
            if hasattr(self.human_gs, "betas")
            else self.train_dataset.betas[0]
        )

        static_smpl_params = get_smpl_static_params(
            betas=betas,
            pose_type=(
                self.cfg.human.canon_pose_type if pose_type is None else pose_type
            ),
        )

        if is_train_progress:
            progress_imgs = []

        pbar = (
            range(nframes)
            if is_train_progress
            else tqdm(range(nframes), desc="Canonical:")
        )

        for idx in pbar:
            cam_p = camera_params[idx]
            data = dict(static_smpl_params, **cam_p)

            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    global_orient=data["global_orient"],
                    body_pose=data["body_pose"],
                    betas=data["betas"],
                    transl=data["transl"],
                    smpl_scale=data["smpl_scale"],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )

            if is_train_progress:
                scale_mod = 0.5
                render_pkg = render_human_scene(
                    data=data,
                    human_gs_out=human_gs_out if human_gs_out else None,
                    scene_gs=self.scene_gs,
                    bg_color=self.bg_color,
                    render_mode="human",
                    scaling_modifier=scale_mod,
                )

                image = render_pkg["render"]

                progress_imgs.append(image)

                render_pkg = render_human_scene(
                    data=data,
                    human_gs_out=human_gs_out,
                    scene_gs=self.scene_gs,
                    bg_color=self.bg_color,
                    render_mode="human",
                )

                image = render_pkg["render"]

                progress_imgs.append(image)

            else:
                render_pkg = render_human_scene(
                    data=data,
                    human_gs_out=human_gs_out,
                    scene_gs=self.scene_gs,
                    bg_color=self.bg_color,
                    render_mode="human",
                )

                image = render_pkg["render"]

                torchvision.utils.save_image(
                    image, f"{self.cfg.logdir}/canon/{idx:05d}.png"
                )

        if is_train_progress:
            os.makedirs(f"{self.cfg.logdir}/train_progress/", exist_ok=True)
            log_img = torchvision.utils.make_grid(progress_imgs, nrow=4, pad_value=0)
            save_image(
                log_img,
                f"{self.cfg.logdir}/train_progress/{iter:06d}.png",
                text_labels=f"{iter:06d}, n_gs={self.human_gs.n_gs}",
            )
            return

        video_fname = f"{self.cfg.logdir}/canon_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4"
        create_video(f"{self.cfg.logdir}/canon/", video_fname, fps=10)
        shutil.rmtree(f"{self.cfg.logdir}/canon/")
        os.makedirs(f"{self.cfg.logdir}/canon/")
