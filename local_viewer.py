#
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual
# property and proprietary rights in and to this software and related documentation.
# Any commercial use, reproduction, disclosure or distribution of this software and
# related documentation without an express license agreement from Toyota Motor Europe NV/SA
# is strictly prohibited.
#

import tyro
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from pathlib import Path
import time
import dearpygui.dearpygui as dpg
import numpy as np
import torch
import torch.nn.functional as F

from hugs.utils.viewer_utils import OrbitCamera
from hugs.utils.vis_utils import apply_depth_colormap, apply_colormap, colormap
from hugs.utils.depth_utils import depth_to_normal
# from hugs.gaussian_renderer import GaussianModel
from hugs.scene.gaussian_model import GaussianModel # TODO: replace /w my abstract gaussian model class and remove the gaussian_model file again
from hugs.gaussian_renderer import render


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


@dataclass
class Config:
    pipeline: PipelineConfig
    """Pipeline settings for gaussian splatting rendering"""
    model_path: Optional[Path] = None
    point_path: Optional[Path] = None
    """Path to the gaussian splatting file"""
    motion_path: Optional[Path] = None
    """Path to the motion file (npz)"""
    sh_degree: int = 3 # should be 0 i think (for the human, not sure though)
    """Spherical Harmonics degree"""
    render_mode: Literal['rgb', 'depth', 'depth_normal', 'normal', 'acc_alpha', 'depth_distortion'] = 'rgb'
    """NeRF rendering mode"""
    W: int = 1500 # 960
    """GUI width"""
    H: int = 1000 # 540
    """GUI height"""
    radius: float = 1
    """default GUI camera radius from center"""
    fovy: float = 20
    """default GUI camera fovy"""
    _bg_color = torch.rand(3, dtype=torch.float32, device="cuda")
    # Convert the tensor to a list and then to a tuple
    random_bg_color = tuple(_bg_color.tolist())

    background_color: Tuple[float] = random_bg_color, # (0.,0.,0.) # (1., 1., 1.)
    """default GUI background color"""
    save_folder: Path = Path("./viewer_output")
    """default saving folder"""
    fps: int = 25
    """default fps for recording"""
    keyframe_interval: int = 1
    """default keyframe interval"""
    ref_json: Optional[Path] = None
    """Path to the reference json file. We use this file to complement the exported trajectory json file."""
    demo_mode: bool = False
    """The UI will be simplified in demo mode."""
    orbit_path: Optional[Path] = None


class GaussianSplattingViewer:
    def __init__(self, cfg: Config):
        self.cfg = cfg  # shared with the trainer's cfg to support in-place modification of rendering parameters.
        self.cfg.point_path = cfg.model_path / cfg.point_path

        # viewer settings
        self.W = cfg.W
        self.H = cfg.H
        self.cam = OrbitCamera(self.W, self.H, r=cfg.radius, fovy=cfg.fovy, convention="opencv")
        # self.mesh_color = torch.tensor([0.2, 0.5, 1], dtype=torch.float32)  # default white bg
        # self.bg_color = torch.ones(3, dtype=torch.float32)  # default white bg
        self.last_time_fresh = None
        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation

        # buffers for mouse interaction
        self.cursor_x = None
        self.cursor_y = None
        self.drag_begin_x = None
        self.drag_begin_y = None
        self.drag_button = None

        # rendering settings
        self.render_mode = cfg.render_mode
        self.render_modes = ["rgb", "depth", "depth_normal", "normal", "acc_alpha", "depth_distortion"]
        self.scaling_modifier: float = 1
        self.reset_opacity_settings()
        self.num_timesteps = 1
        self.timestep = 0
        self.show_spatting = True
        self.show_depth = False
        self.constrain_orbit = False

        if cfg.orbit_path is not None:
            self.orbit = np.load(cfg.orbit_path)
            self.orbit = np.diff(self.orbit, axis=0)
            self.orbit_timestep = 0

        print("Initializing 3D Gaussians...")
        self.init_gaussians()
        self.gaussian_count = self.gaussians.get_xyz.shape[0]
        self.original_opacity = self.gaussians.get_opacity

        print("Initializing GUI...")
        self.define_gui()

    def __del__(self):
        dpg.destroy_context()

    def init_gaussians(self):
        # load gaussians
        # self.gaussians = GaussianModel(self.cfg.sh_degree)
        self.gaussians = GaussianModel(self.cfg.sh_degree)

        # selected_fid = self.gaussians.flame_model.mask.get_fid_by_region(['left_half'])
        # selected_fid = self.gaussians.flame_model.mask.get_fid_by_region(['right_half'])
        # unselected_fid = self.gaussians.flame_model.mask.get_fid_except_fids(selected_fid)
        unselected_fid = []

        if self.cfg.point_path is not None:
            if self.cfg.point_path.exists():
                self.gaussians.load_ply(self.cfg.point_path)
            else:
                raise FileNotFoundError(f'{self.cfg.point_path} does not exist.')

    def reset_opacity_settings(self):
        self.opacity_modifier: float = 0
        self.opacity_shift: float = 0
        self.opacity_cutoff: list[float] = [0.0, 1.0]  # min, max

    def update_opacity_histogram(self, opacities):
        dpg.set_value("_opacity_hist_series", [opacities])

    def refresh(self):
        dpg.set_value("_texture", self.render_buffer)

        if self.last_time_fresh is not None:
            elapsed = time.time() - self.last_time_fresh
            fps = 1 / elapsed
            dpg.set_value("_log_fps", f'{fps:.1f}')
        self.last_time_fresh = time.time()

    def define_gui(self):
        dpg.create_context()

        # register texture =================================================================================================
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        # window: canvas ==================================================================================================
        with dpg.window(label="canvas", tag="_canvas_window", width=self.W, height=self.H, no_title_bar=True,
                        no_move=True, no_bring_to_front_on_focus=True, no_resize=True):
            dpg.add_image("_texture", width=self.W, height=self.H, tag="_image")

        # window: scene info ==================================================================================================
        # scene info
        with dpg.window(label="Scene Info", tag="_scene_info", autosize=True):
            dpg.add_text(f"Model Path: {self.cfg.model_path}")
            dpg.add_text(f"Total #Gaussians: {self.gaussian_count}")

        # window: opacity debugger ==========================================================================================
        with dpg.window(label="Opacity Debugger", tag="_opacity_window", autosize=True):
            # opacity histogram plot
            with dpg.plot(label="Opacity Histogram", height=250, width=350):
                dpg.add_plot_axis(dpg.mvXAxis, label="Opacity")
                dpg.add_plot_axis(dpg.mvYAxis, label="Frequency", tag="_opacity_hist_yaxis")
                opacities = self.gaussians.get_opacity.cpu().detach().squeeze(-1).numpy().tolist()
                dpg.add_histogram_series(x=opacities, parent="_opacity_hist_yaxis", tag="_opacity_hist_series", bins=60)

            # opacity_scaling slider
            def callback_set_opacity_shift(sender, app_data):
                self.opacity_shift = app_data
                self.need_update = True

            dpg.add_slider_float(label="Opacity shift", min_value=-10, max_value=10, format="%.2f", width=200,
                                 default_value=self.opacity_shift, callback=callback_set_opacity_shift,
                                 tag="_slider_opacity_shift")

            # opacity_modifier slider
            def callback_set_opacity_modifier(sender, app_data):
                self.opacity_modifier = app_data
                self.need_update = True

            dpg.add_slider_float(label="Opacity constant", min_value=0, max_value=1, format="%.2f", width=200,
                                 default_value=self.opacity_modifier, callback=callback_set_opacity_modifier,
                                 tag="_slider_opacity_modifier")

            # opacity_cutoff sliders
            def callback_set_opacity_cutoff_min(sender, app_data):
                self.opacity_cutoff[0] = app_data
                self.need_update = True

            def callback_set_opacity_cutoff_max(sender, app_data):
                self.opacity_cutoff[1] = app_data
                self.need_update = True

            with dpg.group(horizontal=True):
                dpg.add_slider_float(label="Min Opacity", min_value=0, max_value=1, format="%.2f", width=100,
                                     default_value=self.opacity_cutoff[0], callback=callback_set_opacity_cutoff_min,
                                     tag="_slider_opacity_cutoff_min")

                dpg.add_slider_float(label="Max Opacity", min_value=0, max_value=1, format="%.2f", width=100,
                                     default_value=self.opacity_cutoff[1], callback=callback_set_opacity_cutoff_max,
                                     tag="_slider_opacity_cutoff_max")

            with dpg.group(horizontal=True):
                def callback_reset_opacity_settings(sender, app_data):
                    self.reset_opacity_settings()
                    self.need_update = True

                    dpg.set_value("_slider_opacity_shift", self.opacity_shift)
                    dpg.set_value("_slider_opacity_modifier", self.opacity_modifier)
                    dpg.set_value("_slider_opacity_cutoff_min", self.opacity_cutoff[0])
                    dpg.set_value("_slider_opacity_cutoff_max", self.opacity_cutoff[1])

                dpg.add_button(label="reset opacity settings", tag="_button_reset_opacity_settings",
                               callback=callback_reset_opacity_settings)

        # window: rendering options ==================================================================================================
        # rendering options
        with dpg.window(label="Render", tag="_render_window", autosize=True):

            with dpg.group(horizontal=True):
                dpg.add_text("FPS: ", show=not self.cfg.demo_mode)
                dpg.add_text("", tag="_log_fps", show=not self.cfg.demo_mode)

            # # render_mode combo
            # def callback_change_mode(sender, app_data):
            #     self.render_mode = app_data
            #     self.need_update = True
            # dpg.add_combo(('rgb', 'depth', 'opacity'), label='render mode', default_value=self.render_mode, callback=callback_change_mode)

            with dpg.group(horizontal=True):
                def callback_render_mode(mode="rgb"):
                    def cb(sender, app_data):
                        if app_data:
                            self.render_mode = mode
                            for m in self.render_modes:
                                if m != mode:
                                    dpg.set_value(f"_checkbox_render_{m}", False)
                        elif mode != "rgb":
                            callback_render_mode("rgb")(None, True)
                        self.need_update = True

                    return cb

                for m in self.render_modes:
                    dpg.add_checkbox(label=m, default_value=self.render_mode == m,
                                     callback=callback_render_mode(m), tag=f"_checkbox_render_{m}")

            with dpg.group(horizontal=True):
                def callback_constrain_orbit(sender, app_data):
                    if app_data:
                        radius = self.cam.radius
                        fovy = self.cam.fovy
                        self.cam.reset()
                        self.cam.radius = radius
                        self.cam.fovy = fovy
                        self.orbit_timestep = 0

                    self.constrain_orbit = app_data
                    self.need_update = True

                dpg.add_checkbox(label="orbit", default_value=self.constrain_orbit,
                                 callback=callback_constrain_orbit)

            # scaling_modifier slider
            def callback_set_scaling_modifier(sender, app_data):
                self.scaling_modifier = app_data
                self.need_update = True

            dpg.add_slider_float(label="Scale modifier", min_value=0, max_value=1, format="%.2f", width=200,
                                 default_value=self.scaling_modifier, callback=callback_set_scaling_modifier,
                                 tag="_slider_scaling_modifier")

            # # bg_color picker
            # def callback_change_bg(sender, app_data):
            #     self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
            #     self.need_update = True
            # dpg.add_color_edit((self.bg_color*255).tolist(), label="Background Color", width=200, no_alpha=True, callback=callback_change_bg)

            # # near slider
            # def callback_set_near(sender, app_data):
            #     self.cam.znear = app_data
            #     self.need_update = True
            # dpg.add_slider_int(label="near", min_value=1e-8, max_value=2, format="%.2f", default_value=self.cam.znear, callback=callback_set_near, tag="_slider_near")

            # # far slider
            # def callback_set_far(sender, app_data):
            #     self.cam.zfar = app_data
            #     self.need_update = True
            # dpg.add_slider_int(label="far", min_value=1e-3, max_value=10, format="%.2f", default_value=self.cam.zfar, callback=callback_set_far, tag="_slider_far")

            # fov slider
            def callback_set_fovy(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True

            dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, width=200, format="%d deg",
                               default_value=self.cam.fovy, callback=callback_set_fovy, tag="_slider_fovy",
                               show=not self.cfg.demo_mode)

            # camera
            with dpg.group(horizontal=True):
                def callback_reset_camera(sender, app_data):
                    self.cam.reset()
                    self.need_update = True
                    dpg.set_value("_slider_fovy", self.cam.fovy)

                dpg.add_button(label="reset camera", tag="_button_reset_pose", callback=callback_reset_camera,
                               show=not self.cfg.demo_mode)

        ### register mouse handlers ========================================================================================

        def callback_mouse_move(sender, app_data):
            self.cursor_x, self.cursor_y = app_data
            if not dpg.is_item_focused("_canvas_window"):
                return

            if self.drag_begin_x is None or self.drag_begin_y is None:
                self.drag_begin_x = self.cursor_x
                self.drag_begin_y = self.cursor_y
            else:
                dx = self.cursor_x - self.drag_begin_x
                dy = self.cursor_y - self.drag_begin_y

                # button=dpg.mvMouseButton_Left
                if self.drag_button is dpg.mvMouseButton_Left:
                    self.cam.orbit(dx, dy)
                    self.need_update = True
                elif self.drag_button is dpg.mvMouseButton_Middle:
                    self.cam.pan(dx, dy)
                    self.need_update = True

        def callback_mouse_button_down(sender, app_data):
            if not dpg.is_item_focused("_canvas_window"):
                return
            self.drag_begin_x = self.cursor_x
            self.drag_begin_y = self.cursor_y
            self.drag_button = app_data[0]

        def callback_mouse_release(sender, app_data):
            self.drag_begin_x = None
            self.drag_begin_y = None
            self.drag_button = None

            self.dx_prev = None
            self.dy_prev = None

        def callback_mouse_drag(sender, app_data):
            if not dpg.is_item_focused("_canvas_window"):
                return

            button, dx, dy = app_data
            if self.dx_prev is None or self.dy_prev is None:
                ddx = dx
                ddy = dy
            else:
                ddx = dx - self.dx_prev
                ddy = dy - self.dy_prev

            self.dx_prev = dx
            self.dy_prev = dy

            if ddx != 0 and ddy != 0:
                if button is dpg.mvMouseButton_Left:
                    self.cam.orbit(ddx, ddy)
                    self.need_update = True
                elif button is dpg.mvMouseButton_Middle:
                    self.cam.pan(ddx, ddy)
                    self.need_update = True

        def callbackmouse_wheel(sender, app_data):
            delta = app_data
            if dpg.is_item_focused("_canvas_window"):
                self.cam.scale(delta)
                self.need_update = True

        with dpg.handler_registry():
            # this registry order helps avoid false fire
            dpg.add_mouse_release_handler(callback=callback_mouse_release)
            # dpg.add_mouse_drag_handler(callback=callback_mouse_drag)  # not using the drag callback, since it does not return the starting point
            dpg.add_mouse_move_handler(callback=callback_mouse_move)
            dpg.add_mouse_down_handler(callback=callback_mouse_button_down)
            dpg.add_mouse_wheel_handler(callback=callbackmouse_wheel)

        def callback_viewport_resize(sender, app_data):
            while self.rendering:
                time.sleep(0.01)
            self.need_update = False
            self.W = app_data[0]
            self.H = app_data[1]
            self.cam.image_width = self.W
            self.cam.image_height = self.H
            self.render_buffer = np.zeros((self.H, self.W, 3), dtype=np.float32)

            # delete and re-add the texture and image
            dpg.delete_item("_texture")
            dpg.delete_item("_image")

            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")
            dpg.add_image("_texture", width=self.W, height=self.H, tag="_image", parent="_canvas_window")
            dpg.configure_item("_canvas_window", width=self.W, height=self.H)
            self.need_update = True

        dpg.set_viewport_resize_callback(callback_viewport_resize)

        ### global theme ==================================================================================================
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_canvas_window", theme_no_padding)

        ### finish setup ==================================================================================================
        dpg.create_viewport(title='Local Viewer', width=self.W, height=self.H, resizable=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def prepare_camera(self):
        @dataclass
        class Cam:
            FoVx = float(np.radians(self.cam.fovx))
            FoVy = float(np.radians(self.cam.fovy))
            image_height = self.cam.image_height
            image_width = self.cam.image_width
            world_view_transform = torch.tensor(
                self.cam.world_view_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            full_proj_transform = torch.tensor(
                self.cam.full_proj_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            camera_center = torch.tensor(self.cam.pose[:3, 3]).cuda()

        return Cam

    @torch.no_grad()
    def run(self):
        print("Running Viewer...")

        while dpg.is_dearpygui_running():
            if self.cfg.orbit_path is not None and self.constrain_orbit:
                t = self.orbit_timestep % (self.orbit.shape[0] - 1)
                if self.orbit_timestep != 0 and self.orbit_timestep % (self.orbit.shape[0] - 1) == 0:
                    t += 1
                    self.orbit_timestep += 1
                dx, dy = self.orbit[t]
                self.orbit_timestep += 1

                self.cam.orbit(dx, dy)
                self.need_update = True

            if self.need_update:
                self.rendering = True
                cam = self.prepare_camera()

                if self.opacity_modifier > 0:
                    new_opacity = torch.ones_like(self.original_opacity) * self.opacity_modifier
                else:
                    new_opacity = self.original_opacity.clone()

                new_opacity[(self.original_opacity < self.opacity_cutoff[0]) | (self.original_opacity > self.opacity_cutoff[1])] = 0
                # self.gaussians.get_opacity = torch.mul(new_opacity, self.opacity_scaling)
                self.gaussians.get_opacity = self.gaussians.opacity_activation(self.gaussians.inverse_opacity_activation(new_opacity) + self.opacity_shift)
                self.update_opacity_histogram(self.gaussians.get_opacity.cpu().detach().squeeze(-1).numpy().tolist())

                # rgb
                self.cfg.pipeline.compute_view2gaussian_python = False
                render_pkg = render(cam, self.gaussians,
                                    torch.tensor(self.cfg.background_color).cuda(),
                                    kernel_size=0.0, scaling_modifier=self.scaling_modifier)

                if self.render_mode == "rgb":
                    image = render_pkg["render"][:3, :, :]
                    rgb = image.permute(1, 2, 0).contiguous()
                elif self.render_mode == "depth":
                    depth = render_pkg["render"][6, :, :]
                    depth_map = apply_depth_colormap(depth[..., None], render_pkg["render"][7, :, :, None],
                                                     near_plane=None, far_plane=None)
                    depth_map = depth_map.permute(2, 0, 1)
                    rgb = depth_map.permute(1, 2, 0).contiguous()
                elif self.render_mode == "depth_normal":
                    depth = render_pkg["render"][6, :, :]
                    depth_normal, _ = depth_to_normal(cam, depth[None, ...])
                    depth_normal = (depth_normal + 1.) / 2.
                    depth_normal = depth_normal.permute(2, 0, 1)
                    rgb = depth_normal.permute(1, 2, 0).contiguous()
                elif self.render_mode == "normal":
                    normal = render_pkg["render"][3:6, :, :]
                    normal = F.normalize(normal, p=2, dim=0)

                    # transform to world space
                    c2w = (cam.world_view_transform.T).inverse()
                    normal2 = c2w[:3, :3] @ normal.reshape(3, -1)
                    normal = normal2.reshape(3, *normal.shape[1:])
                    normal = (normal + 1.) / 2.
                    rgb = normal.permute(1, 2, 0).contiguous()
                elif self.render_mode == "acc_alpha":
                    accumlated_alpha = render_pkg["render"][7, :, :, None]
                    colored_accum_alpha = apply_depth_colormap(accumlated_alpha, None,
                                                               near_plane=0.0, far_plane=1.0)
                    colored_accum_alpha = colored_accum_alpha.permute(2, 0, 1)
                    rgb = colored_accum_alpha.permute(1, 2, 0).contiguous()
                elif self.render_mode == "depth_distortion":
                    distortion_map = render_pkg["render"][8, :, :]
                    distortion_map = colormap(distortion_map.detach().cpu().numpy())
                    rgb = distortion_map.permute(1, 2, 0).contiguous()

                self.render_buffer = rgb.cpu().numpy()
                self.refresh()
                self.rendering = False
                self.need_update = False

            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    gui = GaussianSplattingViewer(cfg)
    gui.run()
