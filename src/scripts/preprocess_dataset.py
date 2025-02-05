import argparse
import numpy as np
import os
import json
import glob
import cv2
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial

from pathlib import Path
from numpy.typing import NDArray
from typing import Tuple, Dict, Any
from tqdm import tqdm
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker

from src.utils.landmarks import (
    init_face_landmarker,
    get_frames_from_video,
    get_video_landmarks,
)
from src.utils.constants import BLENDSHAPE_NAMES


def get_video_info(
    face_landmarker: FaceLandmarker, video_path: Path, bad_frame_percent: float
) -> Tuple[NDArray, NDArray]:
    """
    Get the blendshapes and head angles from a video.

    Args:
        face_landmarker: FaceLandmarker instance.
        video_path: Path to the video.
        bad_frame_percent: percent for bad frames in a video.

    Returns:
        Tuple[NDArray, NDArray]: Blendshapes and head angle arrays of shape
            (n_frames, n_blendshapes) and (n_frames, 3).
    """
    frames = get_frames_from_video(video_path)
    blendshape_names, blendshapes, head_angles, bad_frames = get_video_landmarks(
        frames,
        face_landmarker,
        bad_frame_percent=bad_frame_percent,
    )
    return blendshape_names, blendshapes, head_angles, bad_frames


def analyze_video(video_path: Path) -> Dict[str, float]:
    """Analyze a single video and return its duration and framerate."""
    cap = cv2.VideoCapture(str(video_path))
    framerate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / framerate
    cap.release()
    return {"duration": duration_s, "framerate": framerate}


# Global variable to store the face landmarker in each process
_process_face_landmarker = None


def init_worker():
    """Initialize the face landmarker for this worker process."""
    global _process_face_landmarker
    _process_face_landmarker = init_face_landmarker("GPU")


def calculate_sequence_stats(data: np.ndarray) -> Dict[str, float]:
    """Calculate mean and standard deviation for each dimension of the input sequence."""
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return {"means": means.tolist(), "stds": stds.tolist()}


def process_video(
    video_path: str, output_dir: Path, analyze_only: bool, bad_frame_percent: float
) -> Dict[str, Any]:
    """Process a single video using the process-local face landmarker."""
    full_path = os.path.abspath(video_path)
    basename = os.path.basename(video_path)
    result = {"path": full_path}

    try:
        # Always analyze video duration and framerate
        video_info = analyze_video(full_path)
        result.update(video_info)

        if not analyze_only:
            blendshape_names, blendshapes, head_angles, bad_frames = get_video_info(
                _process_face_landmarker, full_path, bad_frame_percent
            )
            # reorganize the blendshapes
            blendshapes = blendshapes[
                :, [blendshape_names.index(name) for name in BLENDSHAPE_NAMES]
            ]

            # Save the processed data
            np.savez(
                os.path.join(output_dir, basename + ".npz"),
                blendshapes=blendshapes,
                head_angles=head_angles,
                original_fname=full_path,
                bad_frames=np.array(bad_frames),
            )

        return result
    except Exception as e:
        print(f"Error processing {full_path}: {e}")
        return {"path": full_path, "error": str(e)}


def preprocess_dir(
    video_dir: Path,
    output_dir: Path,
    analyze_only: bool,
    bad_frame_percent: float,
    max_videos: int = None,
):
    """
    Preprocess a directory of videos using multiprocessing.

    Args:
        video_dir: Path to the video directory.
        output_dir: Path to the output directory.
        analyze_only: Whether to only analyze the videos.
        bad_frame_percent: percent for bad frames in a video
        max_videos: Maximum number of videos to process. If None, process all videos.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save info dictionary
    info_dict = {
        "blendshape_names": BLENDSHAPE_NAMES,
        "head_angles": ["pitch", "yaw", "roll"],
    }
    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json.dump(info_dict, f)

    # Get list of videos
    video_list = glob.glob(str(video_dir) + "/**/*.mp4", recursive=True)
    if max_videos:
        video_list = video_list[:max_videos]

    num_processes = max(1, mp.cpu_count() - 1)
    process_func = partial(
        process_video,
        output_dir=output_dir,
        analyze_only=analyze_only,
        bad_frame_percent=bad_frame_percent,
    )

    # Process videos
    successful = 0
    failed = 0
    initializer = init_worker if not analyze_only else None

    with mp.Pool(processes=num_processes, initializer=initializer) as pool:
        for result in tqdm(
            pool.imap(process_func, video_list),
            total=len(video_list),
            desc="Processing videos",
        ):
            if "error" not in result:
                successful += 1
            else:
                failed += 1

    print(f"\nProcessing complete:")
    print(f"Total videos: {len(video_list)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=Path("data/TalkingHead-1KH/small/cropped_clips"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/TalkingHead-1KH/small/cropped_clips_info"),
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only analyze the videos, do not preprocess.",
    )
    parser.add_argument(
        "--bad_frame_percent",
        type=float,
        default=0.8,
        help="percent for bad frames in a video.",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to process. If not specified, process all videos.",
    )
    args = parser.parse_args()
    preprocess_dir(
        args.video_dir,
        args.output_dir,
        args.analyze_only,
        args.bad_frame_percent,
        args.max_videos,
    )


if __name__ == "__main__":
    main()
