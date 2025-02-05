import torch
import torch.nn as nn
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import math

from numpy.typing import NDArray
from typing import Tuple
from pathlib import Path
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesAtlas,
    hard_rgb_blend,
    BlendParams,
)

from src.utils.constants import BLENDSHAPE_NAMES, VIDEO_FPS
from src.utils.landmarks import euler_to_rotation_matrix


class Scene:
    """
    A class to render a 3D scene with a given mesh.
    """

    def __init__(self, obj_folder_path: Path, device: torch.device):
        """
        Initialize the scene with the given device and mesh.

        Args:
            device: Device to run the renderer on
            obj_folder_path: Path to the folder containing the .obj files for the
                neutral mesh and shape keys. Must contain a file named "_neutral.obj"
        """
        self.device = device
        self.obj_folder_path = obj_folder_path
        self.neutral_mesh_path = obj_folder_path / "_neutral.obj"
        self.renderer, self.cameras = setup_renderer(device)
        self.verts, self.faces, self.tex = setup_obj(self.neutral_mesh_path, device)

    def image(self, output_file: Path):
        """
        Render the mesh to an image.

        Returns:
            matplotlib axis with the rendered image
        """
        flame_mesh = Meshes(
            verts=[self.verts.to(self.device)],
            faces=[self.faces.verts_idx.to(self.device)],
            textures=self.tex,
        )
        target_images = self.renderer(flame_mesh, cameras=self.cameras)
        render_np = target_images[0][..., :3].detach().cpu().numpy()

        _, ax = plt.subplots()
        # return the image
        ax.imshow(render_np)
        ax.axis("off")
        plt.savefig(output_file)
        print("Image saved to", output_file)
        ax.clear()

    def render_verts_sequence(
        self, verts_sequence: NDArray[np.float32], output_file: Path
    ):
        """
        Render a sequence of vertices into a video file.

        Args:
            verts_sequence: List of vertices to render
            output_file: Path to the output video file
            fps: Frames per second of the video
        """
        writer = imageio.get_writer(output_file, fps=VIDEO_FPS, macro_block_size=1)
        for verts in verts_sequence:
            # create a mesh and render it
            flame_mesh = Meshes(
                verts=[verts.to(self.device)],
                faces=[self.faces.verts_idx.to(self.device)],
                textures=self.tex,
            )
            rendered_frame = self.renderer(flame_mesh, cameras=self.cameras)
            rendered_frame_np = rendered_frame[0][..., :3].detach().cpu().numpy()
            # write the rendered frame into a video
            writer.append_data(np.uint8(rendered_frame_np * 255))
        writer.close()
        print("Video saved to", output_file)

    def render_sequence(self, shapekey_coeffs, euler_angles, output_file):
        shape_deltas = dict()
        for key in BLENDSHAPE_NAMES:
            cur_path = f"{self.obj_folder_path}/{key}.obj"
            cur_verts, _, _ = load_obj(cur_path, device=self.device)
            shape_deltas[key] = cur_verts - self.verts

        # Calculate center of mass of neutral mesh for rotation
        center = self.verts.mean(dim=0, keepdim=True)

        all_vertices = []
        for frame_idx in range(shapekey_coeffs.shape[0]):
            # Apply shape keys
            cur_coeffs = dict(zip(BLENDSHAPE_NAMES, shapekey_coeffs[frame_idx]))
            weighted_deltas = 0.0
            for key in cur_coeffs.keys():
                weighted_deltas += cur_coeffs[key] * shape_deltas[key]
            current_verts = self.verts + weighted_deltas

            # Apply rotation
            pitch, yaw, roll = euler_angles[frame_idx]
            R = euler_to_rotation_matrix(pitch, yaw, roll).float().to(self.device)

            # Center, rotate, and uncenter vertices
            centered_verts = current_verts - center
            rotated_verts = torch.matmul(centered_verts, R.T)
            final_verts = rotated_verts + center

            all_vertices.append(final_verts)

        self.render_verts_sequence(all_vertices, output_file)


class FlatShader(nn.Module):
    """
    A shader which simply returns the color of the face, ignoring lighting.
    """

    def __init__(self, blend_params: BlendParams = None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image


def setup_renderer(device: torch.device, shade_flat: bool = False) -> MeshRenderer:
    """
    Set up a 3d renderer with the given device.

    Args:
        device: Device to run the renderer on
        shade_flat: Whether to shade the mesh flat or with lighting

    Returns:
        MeshRenderer object
    """
    # Set up the renderer
    distance = 0.3  # distance from camera to the object
    elevation = 0.0  # angle of elevation in degrees
    azimuth = math.pi

    lights = PointLights(device=device, location=[[0, 0.0, 3.0]])

    # Get the position of the camera based on the spherical angles
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.1)
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    shader = (
        FlatShader(device=device)
        if shade_flat
        else SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=shader,
    )
    return renderer, cameras


def setup_obj(obj_filename: Path, device: torch.device) -> Tuple:
    """
    Load the obj file and create a texture atlas if needed.

    Args:
        obj_filename: Path to the obj file
        device: Device to load the data onto

    Returns:
        Tuple of vertices, faces, and textures
    """
    verts, faces, aux = load_obj(
        obj_filename, load_textures=True, create_texture_atlas=True, device=device
    )

    if aux.texture_atlas is not None:
        atlas = aux.texture_atlas.unsqueeze(0)  # Add batch dimension
        tex = TexturesAtlas(atlas=atlas.to(device))
    else:
        # Create single color texture for each face
        atlas = torch.ones((1, len(faces.verts_idx), 4, 4, 3), device=device)
        for face_idx, mat_name in enumerate(faces.materials_idx):
            if mat_name in aux.material_colors:
                color = aux.material_colors[mat_name]["diffuse_color"]
                atlas[0, face_idx] = color.expand(4, 4, 3)
        tex = TexturesAtlas(atlas=atlas)

    return verts, faces, tex


def test_render_mesh():
    device = torch.device("cuda:0")
    shapekey_path = "assets/reference_mesh/shape_keys"
    scene = Scene(Path(shapekey_path), device)
    output_dir = "render_output"
    os.makedirs(output_dir, exist_ok=True)
    scene.image(Path(f"{output_dir}/neutral.png"))


if __name__ == "__main__":
    test_render_mesh()
