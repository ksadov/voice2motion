import argparse
import os

from pathlib import Path

from src.utils.landmarks import (
    get_frames_from_video,
    init_face_landmarker,
    info_for_frame,
    plot_blendshapes,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=Path,
        default="data/TalkingHead-1KH/small/cropped_clips/--Y9imYnfBw_0000_S1015_E1107_L488_T23_R824_B359.mp4",
    )
    parser.add_argument("--frame_idx", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    output_dir = Path("render_output")
    os.makedirs(output_dir, exist_ok=True)
    args = parser.parse_args()
    delegate = "GPU" if args.device == "cuda" else "CPU"
    detector = init_face_landmarker(delegate)
    frames = get_frames_from_video(args.video_path)
    blendshape_names, frame_scores, head_pitch, head_yaw, head_roll = info_for_frame(
        frames, args.frame_idx, detector
    )
    fig = plot_blendshapes(frames, args.frame_idx, frame_scores, blendshape_names)
    fig.savefig(output_dir / "blendshapes.png")
    print("Saved blendshapes plot to", output_dir / "blendshapes.png")
    print(f"Head Pitch: {head_pitch}, Head Yaw: {head_yaw}, Head Roll: {head_roll}")


if __name__ == "__main__":
    main()
