import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
from scipy.signal import correlate
from tqdm import tqdm

from src.model.inference import init_pipeline, render_inference
from src.utils.constants import BLENDSHAPE_NAMES, SAMPLE_RATE
from src.utils.landmarks import (
    get_frames_from_video,
    get_video_landmarks,
    init_face_landmarker,
)
from src.utils.rendering import Scene


def compute_temporal_consistency(
    head_angles: np.ndarray, blendshapes: np.ndarray
) -> Dict[str, float]:
    """
    Compute temporal consistency metrics using acceleration-based measures.

    Args:
        head_angles: Array of head angles of shape (T, 3)
        blendshapes: Array of blendshapes of shape (T, D)

    Returns:
        Dictionary containing mean squared jerk for head angles and blendshapes
    """
    # Compute jerk (third derivative) for head angles
    head_angles_jerk = np.gradient(
        np.gradient(np.gradient(head_angles, axis=0), axis=0), axis=0
    )
    head_angles_msj = np.mean(np.square(head_angles_jerk))

    # Compute jerk for blendshapes
    blendshapes_jerk = np.gradient(
        np.gradient(np.gradient(blendshapes, axis=0), axis=0), axis=0
    )
    blendshapes_msj = np.mean(np.square(blendshapes_jerk))

    return {
        "head_angles_mean_squared_jerk": head_angles_msj,
        "blendshapes_mean_squared_jerk": blendshapes_msj,
    }


def compute_lip_sync_accuracy(
    audio_path: Path, blendshapes: np.ndarray, sample_rate: int = SAMPLE_RATE
) -> Dict[str, float]:
    """
    Compute lip-sync accuracy by cross-correlating jaw opening with audio energy.

    Args:
        audio_path: Path to audio file
        blendshapes: Array of blendshapes of shape (T, D)
        sample_rate: Audio sample rate

    Returns:
        Dictionary containing peak correlation and temporal offset
    """
    # Load and process audio
    audio, sr = torchaudio.load(audio_path)
    if audio.size(0) > 1:
        audio = audio.mean(0, keepdim=True)
    audio = audio.squeeze().numpy()

    # Compute audio energy
    frame_length = int(sample_rate / 30)  # Assuming 30fps video
    audio_energy = np.array(
        [
            np.mean(np.square(audio[i : i + frame_length]))
            for i in range(0, len(audio) - frame_length, frame_length)
        ]
    )

    # Get jaw opening blendshape
    jaw_open_idx = BLENDSHAPE_NAMES.index("jawOpen")
    jaw_opening = blendshapes[:, jaw_open_idx]

    # Normalize signals
    audio_energy = (audio_energy - np.mean(audio_energy)) / np.std(audio_energy)
    jaw_opening = (jaw_opening - np.mean(jaw_opening)) / np.std(jaw_opening)

    # Compute cross-correlation
    correlation = correlate(audio_energy, jaw_opening, mode="full")
    lags = np.arange(-(len(jaw_opening) - 1), len(audio_energy))

    # Find peak correlation and corresponding lag
    peak_idx = np.argmax(correlation)
    peak_correlation = correlation[peak_idx]
    temporal_offset = lags[peak_idx] / 30.0  # Convert to seconds

    return {
        "peak_correlation": peak_correlation,
        "temporal_offset_seconds": temporal_offset,
    }


def compute_motion_diversity(
    head_angles: List[np.ndarray], blendshapes: List[np.ndarray]
) -> Dict[str, float]:
    """
    Compute motion diversity metrics using standard deviations across clips.

    Args:
        head_angles: List of arrays containing head pose angles for different sequences
        blendshapes: List of arrays containing blendshape coefficients for different sequences

    Returns:
        Dictionary containing diversity metrics based on standard deviations
    """
    # Stack all sequences into a single array
    all_head_angles = np.vstack(head_angles)  # Shape: (total_frames, 3)
    all_blendshapes = np.vstack(blendshapes)  # Shape: (total_frames, n_blendshapes)

    # Compute standard deviations for each dimension
    head_angle_std = np.std(all_head_angles, axis=0)  # Shape: (3,)
    blendshape_std = np.std(all_blendshapes, axis=0)  # Shape: (n_blendshapes,)

    # Compute mean standard deviation across dimensions
    mean_head_angle_std = np.mean(head_angle_std)
    mean_blendshape_std = np.mean(blendshape_std)

    # Also compute max standard deviation to capture extreme variations
    max_head_angle_std = np.max(head_angle_std)
    max_blendshape_std = np.max(blendshape_std)

    return {
        "mean_head_angle_std": mean_head_angle_std,
        "mean_blendshape_std": mean_blendshape_std,
        "max_head_angle_std": max_head_angle_std,
        "max_blendshape_std": max_blendshape_std,
        "head_angle_std_per_dim": {
            "pitch": head_angle_std[0],
            "yaw": head_angle_std[1],
            "roll": head_angle_std[2],
        },
    }


def evaluate_video(
    video_path: Path,
    pipeline,
    scene: Scene,
    device: str,
    max_audio_samples_per_step: int = 8000,
    min_audio_samples_per_step: int = 8000,
) -> Tuple[
    Dict[str, float], Dict[str, float], Dict[str, np.ndarray], Dict[str, np.ndarray]
]:
    """
    Evaluate a single video file.

    Args:
        video_path: Path to video file
        pipeline: Inference pipeline
        scene: Scene object for rendering
        device: Device to run inference on
        max_audio_samples_per_step: Maximum audio samples per step
        min_audio_samples_per_step: Minimum audio samples per step

    Returns:
        Tuple of (model_metrics, gt_metrics, model_data, gt_data) where:
        - model_metrics and gt_metrics are dictionaries of computed metrics
        - model_data and gt_data are dictionaries containing raw head angles and blendshapes
    """
    # Get ground truth using MediaPipe
    delegate = "GPU" if device == "cuda" else "CPU"
    detector = init_face_landmarker(delegate)
    frames = get_frames_from_video(video_path)
    _, gt_blendshapes, gt_head_angles, _ = get_video_landmarks(frames, detector)

    # Get model predictions
    output_path = Path(f"eval_render_output/{video_path.stem}.mp4")
    os.makedirs("eval_render_output", exist_ok=True)
    model_output, _, _, _ = render_inference(
        pipeline,
        scene,
        video_path,
        min_audio_samples_per_step,
        max_audio_samples_per_step,
        output_path,
        mouth_exaggeration=3.0,
        brow_exaggeration=1.0,
        head_wiggle_exaggeration=1.0,
        unsquinch_fix=0.2,
        eye_contact_fix=0.75,
        exaggerate_above=0.01,
        symmetrize_eyes=True,
        return_model_output=True,
    )
    model_blendshapes = model_output["blendshapes"]
    model_head_angles = model_output["head_angles"]

    # Compute metrics for both ground truth and model output
    gt_metrics = {
        **compute_temporal_consistency(gt_head_angles, gt_blendshapes),
        **compute_lip_sync_accuracy(video_path, gt_blendshapes),
    }

    model_metrics = {
        **compute_temporal_consistency(model_head_angles, model_blendshapes),
        **compute_lip_sync_accuracy(video_path, model_blendshapes),
    }

    # Store raw data for diversity computation
    model_data = {"head_angles": model_head_angles, "blendshapes": model_blendshapes}
    gt_data = {"head_angles": gt_head_angles, "blendshapes": gt_blendshapes}

    return model_metrics, gt_metrics, model_data, gt_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir", type=Path, required=True, help="Directory containing video files"
    )
    parser.add_argument(
        "--max_files", type=int, default=10, help="Maximum number of files to evaluate"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_path", type=Path, help="Path to model checkpoint")
    parser.add_argument(
        "--hubert_onnx_path", type=Path, help="Path to HuBERT ONNX model"
    )
    parser.add_argument(
        "--encoder_onnx_path", type=Path, help="Path to encoder ONNX model"
    )
    parser.add_argument(
        "--decoder_onnx_path", type=Path, help="Path to decoder ONNX model"
    )
    parser.add_argument(
        "--shape_keys_path", type=Path, default="assets/reference_mesh/shape_keys"
    )
    args = parser.parse_args()

    # Initialize pipeline
    device = torch.device(args.device)
    scene = Scene(args.shape_keys_path, device)
    pipeline = init_pipeline(
        checkpoint_path=args.checkpoint_path,
        hubert_onnx_path=args.hubert_onnx_path,
        encoder_onnx_path=args.encoder_onnx_path,
        decoder_onnx_path=args.decoder_onnx_path,
        device=device,
    )

    # Get list of video files
    video_files = list(args.video_dir.glob("**/*.mp4"))[: args.max_files]

    # Collect metrics and data for all videos
    model_metrics_list = []
    gt_metrics_list = []
    model_head_angles = []
    model_blendshapes = []
    gt_head_angles = []
    gt_blendshapes = []

    for video_path in tqdm(video_files, desc="Evaluating videos"):
        model_metrics, gt_metrics, model_data, gt_data = evaluate_video(
            video_path,
            pipeline,
            scene,
            args.device,
        )
        model_metrics_list.append(model_metrics)
        gt_metrics_list.append(gt_metrics)
        model_head_angles.append(model_data["head_angles"])
        model_blendshapes.append(model_data["blendshapes"])
        gt_head_angles.append(gt_data["head_angles"])
        gt_blendshapes.append(gt_data["blendshapes"])

    # Compute average metrics
    avg_model_metrics = {
        k: np.mean([m[k] for m in model_metrics_list])
        for k in model_metrics_list[0].keys()
    }

    avg_gt_metrics = {
        k: np.mean([m[k] for m in gt_metrics_list]) for k in gt_metrics_list[0].keys()
    }

    # Print results
    print("\nModel Metrics:")
    for k, v in avg_model_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nGround Truth Metrics:")
    for k, v in avg_gt_metrics.items():
        print(f"{k}: {v:.4f}")

    # Compute motion diversity
    model_diversity = compute_motion_diversity(model_head_angles, model_blendshapes)
    gt_diversity = compute_motion_diversity(gt_head_angles, gt_blendshapes)

    print("\nModel Motion Diversity:")
    for k, v in model_diversity.items():
        if isinstance(v, dict):
            print(f"\n{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v:.4f}")
        else:
            print(f"{k}: {v:.4f}")

    print("\nGround Truth Motion Diversity:")
    for k, v in gt_diversity.items():
        if isinstance(v, dict):
            print(f"\n{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v:.4f}")
        else:
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
