import argparse
from pathlib import Path
import torch
from src.model.inference import init_pipeline, render_inference
from src.utils.rendering import Scene


def main():
    parser = argparse.ArgumentParser()
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint_path", type=str)
    model_group.add_argument(
        "--onnx", action="store_true", help="Use ONNX models instead of PyTorch"
    )

    parser.add_argument(
        "--hubert_onnx_path",
        type=str,
        default="onnx_models/hubert.onnx",
        help="Path to ONNX HuBERT model (required if --onnx is set)",
    )
    parser.add_argument(
        "--encoder_onnx_path",
        type=str,
        default="onnx_models/encoder.onnx",
        help="Path to ONNX decoder model (required if --onnx is set)",
    )
    parser.add_argument(
        "--decoder_onnx_path",
        type=str,
        default="onnx_models/decoder.onnx",
        help="Path to ONNX decoder model (required if --onnx is set)",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--file_path", type=str, default="assets/example.wav")
    parser.add_argument(
        "--shape_keys_path", type=str, default="assets/reference_mesh/shape_keys"
    )
    parser.add_argument("--output_file", type=str, default="render_output/output.mp4")
    parser.add_argument("--chunk_size", type=int, default=90)
    parser.add_argument("--min_audio_samples_per_step", type=int, default=8000)
    parser.add_argument("--max_audio_samples_per_step", type=int, default=8000)
    parser.add_argument("--crossfade_size", type=int, default=5)
    parser.add_argument("--upper_face_exaggeration", type=float, default=1.0)
    parser.add_argument("--lower_face_exaggeration", type=float, default=1.0)
    parser.add_argument("--head_wiggle_exaggeration", type=float, default=1.0)
    parser.add_argument("--unsquinch_fix", type=float, default=0.2)
    parser.add_argument("--eye_contact_fix", type=float, default=0.75)
    parser.add_argument("--exaggerate_above", type=float, default=0.01)
    parser.add_argument("--symmetrize_eyes", action="store_true")

    args = parser.parse_args()

    if args.onnx and (
        args.hubert_onnx_path is None
        or args.decoder_onnx_path is None
        or args.encoder_onnx_path is None
    ):
        parser.error(
            "--onnx requires --hubert_onnx_path, --encoder_onnx_path, and --decoder_onnx_path"
        )

    device = torch.device(args.device)
    scene = Scene(Path(args.shape_keys_path), device)

    pipeline = init_pipeline(
        checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
        hubert_onnx_path=Path(args.hubert_onnx_path) if args.hubert_onnx_path else None,
        encoder_onnx_path=(
            Path(args.encoder_onnx_path) if args.encoder_onnx_path else None
        ),
        decoder_onnx_path=(
            Path(args.decoder_onnx_path) if args.decoder_onnx_path else None
        ),
        chunk_size=args.chunk_size,
        crossfade_size=args.crossfade_size,
        batch_size=1,
        device=device,
    )

    mean_step_time, mean_rtf, time_to_first_sound = render_inference(
        pipeline,
        scene,
        Path(args.file_path),
        args.min_audio_samples_per_step,
        args.max_audio_samples_per_step,
        Path(args.output_file),
        mouth_exaggeration=args.upper_face_exaggeration,
        brow_exaggeration=args.lower_face_exaggeration,
        head_wiggle_exaggeration=args.head_wiggle_exaggeration,
        unsquinch_fix=args.unsquinch_fix,
        eye_contact_fix=args.eye_contact_fix,
        exaggerate_above=args.exaggerate_above,
        symmetrize_eyes=args.symmetrize_eyes,
    )

    print(f"Mean step time: {mean_step_time:.3f}s")
    print(f"Time to first sound: {time_to_first_sound:.3f}s")
    print(f"Mean RTF: {mean_rtf:.3f}x real-time")


if __name__ == "__main__":
    main()
