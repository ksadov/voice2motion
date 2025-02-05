import torch
import argparse
import os
from pathlib import Path

from src.model.simple import load_from_checkpoint
from src.utils.cursed_pytorch_patch import multi_head_attention_forward
from src.utils.constants import HEAD_LANDMARK_DIM

# monkey patching torch.nn.functional.multi_head_attention_forward to fix onnx export
# as per https://github.com/pytorch/pytorch/pull/111800
from src.utils.cursed_pytorch_patch import multi_head_attention_forward

torch.nn.functional.multi_head_attention_forward = multi_head_attention_forward


class DecoderStepWrapper(torch.nn.Module):
    """
    A wrapper module that exposes `decoder_step` as a single `forward` method
    suitable for torch.onnx.export.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src, decoder_in):
        """
        src: [B, T_a, d_model]  -> Already-projected audio features
        decoder_in: [B, T_l, HEAD_LANDMARK_DIM]
        """
        return self.model.decoder_step(src, decoder_in)


class SimpleEncodeWrapper(torch.nn.Module):
    """
    A wrapper module that exposes `simple_encode` as a single `forward` method
    suitable for torch.onnx.export.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src):
        """
        src: [B, T, d_model]  -> Audio features
        """
        return self.model.simple_encode(src)


def export_onnx_decoder(
    checkpoint_path: Path, output_folder: Path, device: torch.device
):
    model = load_from_checkpoint(checkpoint_path, device)
    model.eval()

    decoder_wrapper = DecoderStepWrapper(model)
    simple_encode_wrapper = SimpleEncodeWrapper(model)

    B = 1
    T_a = 16
    T_l = 1

    hubert_src = torch.randn(B, T_a, model.hubert_extractor.feature_dim, device=device)
    projected_src = torch.randn(B, T_a, model.decoder_dimensions.d_model, device=device)

    encoder_output_path = output_folder / "encoder.onnx"
    decoder_output_path = output_folder / "decoder.onnx"
    dummy_decoder_in = torch.randn(B, T_l, HEAD_LANDMARK_DIM, device=device)

    with torch.no_grad():
        _ = simple_encode_wrapper(hubert_src)
        _ = decoder_wrapper(projected_src, dummy_decoder_in)

    torch.onnx.export(
        decoder_wrapper,
        (projected_src, dummy_decoder_in),
        f=str(decoder_output_path),
        export_params=True,
        input_names=["src", "decoder_in"],
        output_names=["decoder_out"],
        dynamic_axes={
            "src": {0: "batch_size", 1: "time_audio"},
            "decoder_in": {0: "batch_size", 1: "time_landmarks"},
            "decoder_out": {0: "batch_size", 1: "time_out"},
        },
        opset_version=14,
    )
    print(f"Exported decoder to {decoder_output_path}")

    torch.onnx.export(
        simple_encode_wrapper,
        hubert_src,
        f=str(encoder_output_path),
        export_params=True,
        input_names=["src"],
        output_names=["encoder_out"],
        dynamic_axes={"src": {0: "batch_size", 1: "time_audio"}},
        opset_version=14,
    )
    print(f"Exported encoder to {encoder_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--onnx_output_folder", type=str, default="onnx_models/")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    onnx_output_folder = Path(args.onnx_output_folder)
    device = torch.device(args.device)

    os.makedirs(onnx_output_folder, exist_ok=True)

    export_onnx_decoder(checkpoint_path, onnx_output_folder, device)


if __name__ == "__main__":
    main()
