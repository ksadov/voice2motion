import argparse
import os

from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm

from src.utils.landmarks import (
    get_frames_from_video,
    init_face_landmarker,
    get_video_landmarks,
    scale_and_center_head_angles,
    regularize_blendshapes,
    deblink,
)
from src.utils.rendering import Scene
from src.utils.video import combine_videos_side_by_side


def render_from_path(
    video_path, shape_keys_path, out_folder, cleanup, device
) -> Tuple[List[str], dict]:
    """
    Render video(s) from a video file path or folder containing video files.

    Args:
        video_path (Path): Path to video file or folder containing video files
        shape_keys_path (Path): Path to shape keys folder
        out_folder (Path): Path to output video file
        cleanup (bool): Whether to center the head, deblink and regularize the blendshapes
        device (str): Device to run the face landmarking

    Returns:
        Tuple[List[str], dict]: List of output video file names and dictionary of bad frame counts
    """
    videos_to_render = []
    is_folder = os.path.isdir(video_path)
    if is_folder:
        videos_to_render = list(video_path.glob("**/*.mp4"))
    else:
        videos_to_render.append(video_path)
    delegate = "GPU" if device == "cuda" else "CPU"
    detector = init_face_landmarker(delegate)
    os.makedirs(out_folder, exist_ok=True)
    output_filenames = []
    bad_frame_counts = []
    for video in tqdm(videos_to_render):
        sanitized_video_stem = video.stem.replace("--", "__")
        output_fname = out_folder / f"{sanitized_video_stem}_rendered.mp4"
        frames = get_frames_from_video(video)
        blendshape_names, video_coeffs, head_angles, bad_frames = get_video_landmarks(
            frames, detector
        )
        if cleanup:
            head_angles = scale_and_center_head_angles(
                head_angles, bad_frames, just_center=True
            )
            video_coeffs = deblink(video_coeffs)
            video_coeffs = regularize_blendshapes(video_coeffs, bad_frames)
        scene = Scene(shape_keys_path, device)
        scene.render_sequence(
            video_coeffs,
            head_angles,
            output_fname,
        )
        combine_videos_side_by_side(video, output_fname, output_fname)
        output_filenames.append(output_fname)
        bad_frame_counts.append(len(bad_frames))
    bad_frame_dict = dict(zip(videos_to_render, bad_frame_counts))
    return output_filenames, bad_frame_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default="data/TalkingHead-1KH/small/cropped_clips/--Y9imYnfBw_0000_S1015_E1107_L488_T23_R824_B359.mp4",
        help="Path to video file or folder containing video files",
    )
    parser.add_argument("--frame_idx", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--shape_keys_path", type=Path, default="assets/reference_mesh/shape_keys"
    )
    parser.add_argument("--out_folder", type=Path, default=Path("render_output"))
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    output_filenames, bad_frame_dict = render_from_path(
        args.path, args.shape_keys_path, args.out_folder, args.cleanup, args.device
    )
    print(f"Output video files: {output_filenames}")
    print(f"Bad frame counts: {bad_frame_dict}")


if __name__ == "__main__":
    main()
