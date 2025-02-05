import os
import requests
from pathlib import Path
import torch
from src.model.inference import init_pipeline, render_inference
from src.utils.rendering import Scene


def run_inference(checkpoint_path):
    device = torch.device("cuda")
    pipeline = init_pipeline(
        checkpoint_path=checkpoint_path,
        chunk_size=90,
        crossfade_size=5,
        batch_size=1,
        device=device,
    )
    scene = Scene(Path("assets/reference_mesh/shape_keys"), device)
    os.makedirs("render_output", exist_ok=True)
    mean_step_time, mean_rtf, time_to_first_sound = render_inference(
        pipeline,
        scene,
        Path("assets/example.wav"),
        48000,
        48000,
        Path("render_output/test_setup.mp4"),
        mouth_exaggeration=3.0,
        brow_exaggeration=1.0,
        head_wiggle_exaggeration=1.0,
        unsquinch_fix=0.2,
        eye_contact_fix=0.75,
        exaggerate_above=0.01,
        symmetrize_eyes=True,
    )

    print(f"Mean step time: {mean_step_time:.3f}s")
    print(f"Time to first sound: {time_to_first_sound:.3f}s")
    print(f"Mean RTF: {mean_rtf:.3f}x real-time")


if __name__ == "__main__":
    run_inference("assets/demo_checkpoint.pt")
