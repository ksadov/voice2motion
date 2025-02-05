import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Optional
from pathlib import Path

import time

from src.utils.audio import load_audio_from_video, AudioStream
from src.utils.constants import HEAD_LANDMARK_DIM
from src.model.simple import load_from_checkpoint
from src.utils.landmarks import (
    unscale_and_uncenter_head_angles,
    clean_up_blendshapes,
    exaggerate_head_wiggle,
)
from src.utils.rendering import Scene
from src.utils.video import combine_videos_side_by_side, add_audio_to_video
from src.utils.constants import (
    N_AUDIO_SAMPLES_PER_VIDEO_FRAME,
    SAMPLE_RATE,
)

import onnxruntime as ort
from dataclasses import dataclass
from typing import Optional, Union


class InferencePipeline:
    """
    Pipeline for running WhisperLike model inference on a video file.

    Added crossfade functionality to smooth transitions between chunks.
    """

    def __init__(
        self,
        max_chunk_size: int,
        crossfade_size: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        """
        Initialize streaming inference pipeline.

        Args:
            max_chunk_size: Maximum number of frames to process in a single chunk
            crossfade_size: Number of frames to use for crossfading between chunks
            batch_size: Batch size for inference
            device: Device to run on
        """
        self.max_chunk_size = max_chunk_size
        self.max_audio_input_size = (
            self.max_chunk_size * N_AUDIO_SAMPLES_PER_VIDEO_FRAME
        )
        self.crossfade_size = crossfade_size
        self.audio_crossfade_size = crossfade_size * N_AUDIO_SAMPLES_PER_VIDEO_FRAME
        self.n_feats = HEAD_LANDMARK_DIM
        self.device = device

        # Maintain state between chunks
        self.prev_output = torch.zeros(batch_size, 0, self.n_feats).to(device)
        self.audio_buffer = torch.zeros(batch_size, 0).to(device)

        # Crossfade buffer stores the overlapping region from the previous chunk
        self.crossfade_buffer = None

        # Pre-compute crossfade weights
        self.crossfade_weights = torch.linspace(0, 1, crossfade_size).to(device)
        self.crossfade_weights = self.crossfade_weights.view(1, -1, 1).expand(
            -1, -1, self.n_feats
        )

    def apply_crossfade(
        self, current_chunk: torch.Tensor, update_crossfade_buffer: bool
    ) -> torch.Tensor:
        """Apply crossfade between previous and current chunk predictions."""
        if self.crossfade_buffer is not None:
            # Extract the crossfade region from the current chunk
            current_fade_region = current_chunk[:, : self.crossfade_size]

            # Blend the overlapping regions using the pre-computed weights
            blended_region = (
                self.crossfade_buffer * (1 - self.crossfade_weights)
                + current_fade_region * self.crossfade_weights
            )

            # Replace the beginning of the current chunk with the blended region
            output = current_chunk.clone()
            output[:, : self.crossfade_size] = blended_region
        else:
            output = current_chunk
        if update_crossfade_buffer:
            self.crossfade_buffer = current_chunk[:, -self.crossfade_size :].clone()
            output = output[:, : -self.crossfade_size]
        return output

    def model_generate(self, src, max_len, initial_context=None):
        """
        Generate output sequence with optional initial context.

        Args:
            src: Source audio features of shape [B, T_a, D], where T_a is the number of
                audio frames corresponding to max_len video frames
            max_len: Number of frames to generate
            initial_context: Optional previous output context (B, J, D), where J is
                in [1, max_len + 1]

        Returns:
            Predicted landmarks [B, max_len - J, D]
        """
        pass

    def infer_chunk(self, audio: torch.Tensor, new_audio_len: int) -> torch.Tensor:
        """Process a single chunk of audio, using previous context if available."""
        n_new_frames = (
            new_audio_len // N_AUDIO_SAMPLES_PER_VIDEO_FRAME + self.crossfade_size
        )
        n_generation_frames = audio.shape[1] // N_AUDIO_SAMPLES_PER_VIDEO_FRAME
        n_context_frames = (n_generation_frames - n_new_frames) + 1
        if n_context_frames > 0:
            initial_context = self.prev_output[:, -n_context_frames:]
        else:
            initial_context = None
        # Generate predictions
        with torch.no_grad():
            predictions = self.model_generate(
                audio, n_generation_frames, initial_context
            )

        self.prev_output = torch.cat([self.prev_output, predictions], dim=1)[
            :, -self.max_chunk_size :
        ]
        return predictions

    def prepare_input_chunk(self, audio: torch.Tensor) -> torch.Tensor:
        new_audio_len = audio.shape[1]
        self.audio_buffer = torch.cat([self.audio_buffer, audio], dim=1)[
            :, -self.max_audio_input_size :
        ]
        return self.audio_buffer, new_audio_len

    def process_output_chunk(
        self,
        chunk: np.ndarray,
        update_crossfade_buffer: bool,
        mouth_exaggeration: float,
        brow_exaggeration: float,
        head_wiggle_exaggeration: float,
        unsquinch_fix: float,
        eye_contact_fix: float,
        exaggerate_above: float,
        symmetrize_eyes: bool,
    ) -> np.ndarray:
        chunk[..., :52] = clean_up_blendshapes(
            chunk[..., :52],
            mouth_exaggeration=mouth_exaggeration,
            brow_exaggeration=brow_exaggeration,
            clear_neutral=True,
            unsquinch_fix=unsquinch_fix,
            eye_contact_fix=eye_contact_fix,
            exaggerate_above=exaggerate_above,
            symmetrize_eyes=symmetrize_eyes,
        )
        if head_wiggle_exaggeration != 1.0:
            chunk[..., 52:] = exaggerate_head_wiggle(
                chunk[..., 52:], head_wiggle_exaggeration
            )
        if self.crossfade_size > 0 and chunk.shape[1] > self.crossfade_size:
            chunk = self.apply_crossfade(chunk, update_crossfade_buffer)
        return chunk

    def __call__(
        self,
        audio: torch.Tensor,
        audio_stream_can_step: bool,
        mouth_exaggeration: float,
        brow_exaggeration: float,
        head_wiggle_exaggeration: float,
        unsquinch_fix: float,
        eye_contact_fix: float,
        exaggerate_above: float,
        symmetrize_eyes: bool,
    ) -> torch.Tensor:
        """
        Run the model on an audio tensor.

        Args:
            audio: Audio tensor of shape (batch_size, n_audio_samples)

        Returns:
            torch.Tensor: Model predictions
        """
        input_chunk, new_audio_len = self.prepare_input_chunk(audio)
        output_chunk = self.infer_chunk(input_chunk, new_audio_len)
        return self.process_output_chunk(
            output_chunk,
            update_crossfade_buffer=audio_stream_can_step,
            mouth_exaggeration=mouth_exaggeration,
            brow_exaggeration=brow_exaggeration,
            head_wiggle_exaggeration=head_wiggle_exaggeration,
            unsquinch_fix=unsquinch_fix,
            eye_contact_fix=eye_contact_fix,
            exaggerate_above=exaggerate_above,
            symmetrize_eyes=symmetrize_eyes,
        )

    def reset(self):
        """Reset internal state"""
        self.prev_output = torch.zeros_like(self.prev_output)
        self.audio_buffer = torch.zeros_like(self.audio_buffer)
        self.crossfade_buffer = None

    def infer_path(
        self,
        path: Path,
        min_audio_samples_per_step: int,
        max_audio_samples_per_step: int,
        mouth_exaggeration: float = 1.0,
        brow_exaggeration: float = 1.0,
        head_wiggle_exaggeration: float = 1.0,
        unsquinch_fix: float = 0.0,
        eye_contact_fix: float = 0.0,
        exaggerate_above: float = 0.0,
        symmetrize_eyes: bool = False,
        max_audio_duration: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Run the model on an input audio or video file under simulated streaming conditions.

        Args:
            path: Path to the audio or video file
            min_audio_samples_per_step: Minimum number of audio samples per step
            max_audio_samples_per_step: Maximum number of audio samples per step
            max_audio_duration: Maximum duration of audio to process in seconds

        Returns:
            torch.Tensor: Model predictions for the entire sequence
        """
        is_video = path.suffix == ".mp4"
        # Reset all buffers
        self.reset()

        # Load audio
        if is_video:
            audio = torch.tensor(load_audio_from_video(path)).to(self.device)
        else:
            audio, sr = torchaudio.load(path, normalize=True)
            audio = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(audio).to(
                self.device
            )
            # make sure audio is mono
            if audio.size(0) > 1:
                audio = audio.mean(0, keepdim=True)
            audio = audio.squeeze(0)

        # Apply duration limit if specified
        if max_audio_duration is not None:
            max_audio_duration_frames = int(max_audio_duration * SAMPLE_RATE)
            audio_len = min(len(audio), max_audio_duration_frames)
        else:
            audio_len = len(audio)

        audio_stream = AudioStream(
            audio[:audio_len], min_audio_samples_per_step, max_audio_samples_per_step
        )

        # Process each chunk
        outputs = []
        step_times = []
        audio_durations = []
        while audio_stream.can_step:
            audio_chunk = audio_stream.step()
            audio_durations.append(audio_chunk.shape[-1] / SAMPLE_RATE)
            # Process the chunk
            start_time = time.time()
            chunk_output = chunk_output = self(
                audio_chunk.unsqueeze(0),
                audio_stream.can_step,
                mouth_exaggeration,
                brow_exaggeration,
                head_wiggle_exaggeration,
                unsquinch_fix,
                eye_contact_fix,
                exaggerate_above,
                symmetrize_eyes,
            )
            step_times.append(time.time() - start_time)
            outputs.append(chunk_output)

        # Concatenate all outputs
        full_output = torch.cat(outputs, dim=1)
        mean_step_time = sum(step_times) / len(step_times)
        mean_rtf = sum(audio_durations) / sum(step_times)
        time_to_first_sound = step_times[0] + audio_durations[0]

        return full_output, mean_step_time, mean_rtf, time_to_first_sound


class PytorchInferencePipeline(InferencePipeline):
    """
    PyTorch version of the inference pipeline.
    """

    def __init__(self, model, max_chunk_size, crossfade_size, batch_size, device):
        """
        Initialize PyTorch inference pipeline.

        Args:
            model: PyTorch model
            max_chunk_size: Maximum number of frames to process in a single chunk
            crossfade_size: Number of frames to use for crossfading between chunks
            batch_size: Batch size for inference
            device: Device to run inference on
        """
        super().__init__(max_chunk_size, crossfade_size, batch_size, device)
        self.model = model

    def model_generate(self, src, max_len, initial_context=None):
        self.model.eval()
        src, _ = self.model.encode(src)

        if initial_context is not None:
            decoder_in = initial_context
            initial_ctx_len = initial_context.size(1)

        else:
            start_frame = torch.zeros(src.size(0), 1, HEAD_LANDMARK_DIM).to(self.device)
            decoder_in = start_frame
            initial_ctx_len = 1

        for i in range(max_len - decoder_in.size(1) + 1):
            next_output = self.model.decoder_step(src, decoder_in)
            decoder_in = torch.cat([decoder_in, next_output], dim=1)

        pred_out = decoder_in[:, initial_ctx_len:]
        return pred_out


@dataclass
class ONNXModels:
    hubert_session: ort.InferenceSession
    encoder_session: ort.InferenceSession
    decoder_session: ort.InferenceSession


class ONNXInferencePipeline(InferencePipeline):
    """
    ONNX version of the inference pipeline.
    """

    def __init__(
        self,
        onnx_models: ONNXModels,
        max_chunk_size: int,
        crossfade_size: int,
        batch_size: int,
        device: torch.device,
    ):
        """
        Initialize ONNX inference pipeline.

        Args:
            onnx_models: ONNXModels containing hubert and decoder sessions
            max_chunk_size: Maximum number of frames to process in a single chunk
            crossfade_size: Number of frames to use for crossfading between chunks
            batch_size: Batch size for inference
            device: Device to run inference on
        """
        super().__init__(max_chunk_size, crossfade_size, batch_size, device)
        self.onnx_models = onnx_models

    def model_generate(self, src, max_len, initial_context=None):
        """
        Generate output sequence using ONNX models.
        """
        # Run HuBERT through ONNX
        src_np = src.cpu().numpy().astype(np.float32)
        hubert_out = self.onnx_models.hubert_session.run(
            None, {"input_values": src_np}
        )[0]
        src = self.onnx_models.encoder_session.run(None, {"src": hubert_out})[0]
        src = torch.from_numpy(src).to(self.device)

        if initial_context is not None:
            decoder_in = initial_context.cpu().numpy().astype(np.float32)
        else:
            decoder_in = np.zeros((src.size(0), 1, HEAD_LANDMARK_DIM)).astype(
                np.float32
            )

        outputs = []
        for i in range(max_len - decoder_in.shape[1] + 1):
            # Run decoder step through ONNX
            next_output = self.onnx_models.decoder_session.run(
                None,
                {"src": src.cpu().numpy().astype(np.float32), "decoder_in": decoder_in},
            )[0]

            decoder_in = np.concatenate([decoder_in, next_output], axis=1)
            outputs.append(torch.from_numpy(next_output).to(self.device))

        pred_out = torch.cat(outputs, dim=1)
        return pred_out


def init_pipeline(
    checkpoint_path: Optional[Path] = None,
    hubert_onnx_path: Optional[Path] = None,
    encoder_onnx_path: Optional[Path] = None,
    decoder_onnx_path: Optional[Path] = None,
    chunk_size: int = 90,
    crossfade_size: int = 5,
    batch_size: int = 1,
    device: torch.device = torch.device("cuda"),
) -> Union[InferencePipeline, ONNXInferencePipeline]:
    """
    Initialize either PyTorch or ONNX inference pipeline based on provided paths.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        hubert_onnx_path: Path to ONNX HuBERT model
        decoder_onnx_path: Path to ONNX decoder model
        chunk_size: Maximum number of frames per chunk
        crossfade_size: Number of frames for crossfading
        batch_size: Batch size for inference
        device: Device to run on

    Returns:
        Either PyTorch or ONNX inference pipeline
    """
    if checkpoint_path is not None:
        # PyTorch pipeline
        model = load_from_checkpoint(checkpoint_path, device)
        return PytorchInferencePipeline(
            model, chunk_size, crossfade_size, batch_size, device
        )
    elif hubert_onnx_path is not None and decoder_onnx_path is not None:
        # ONNX pipeline
        providers = (
            ["CUDAExecutionProvider"]
            if device.type == "cuda"
            else ["CPUExecutionProvider"]
        )

        hubert_session = ort.InferenceSession(
            str(hubert_onnx_path), providers=providers
        )
        encoder_session = ort.InferenceSession(
            str(encoder_onnx_path), providers=providers
        )
        decoder_session = ort.InferenceSession(
            str(decoder_onnx_path), providers=providers
        )

        onnx_models = ONNXModels(hubert_session, encoder_session, decoder_session)
        print("USING ONNX MODELS")
        return ONNXInferencePipeline(
            onnx_models, chunk_size, crossfade_size, batch_size, device
        )
    else:
        raise ValueError(
            "Must provide either checkpoint_path or both hubert_onnx_path and decoder_onnx_path"
        )


def render_inference(
    pipeline: InferencePipeline,
    scene: Scene,
    input_path: Path,
    min_audio_samples_per_step: int,
    max_audio_samples_per_step: int,
    output_file: Path,
    max_audio_duration: Optional[float] = None,
    mouth_exaggeration: float = 1.0,
    brow_exaggeration: float = 1.0,
    head_wiggle_exaggeration: float = 1.0,
    unsquinch_fix: float = 0.0,
    eye_contact_fix: float = 0.0,
    exaggerate_above: float = 0.0,
    symmetrize_eyes: bool = False,
):
    """
    Run the model on an input audio or video file under simulated streaming conditions and save the output video.

    Args:
        pipeline: InferencePipeline object
        scene: Scene object for rendering
        min_audio_samples_per_step: Minimum number of audio samples per step
        max_audio_samples_per_step: Maximum number of audio samples per step
        input_path: Path to the audio or video file
        output_file: Path to the output video file
        max_audio_duration: Maximum duration of audio to process in seconds
        mouth_exaggeration: Lower face exaggeration factor
        brow_exaggeration: Upper face exaggeration factor
        head_wiggle_exaggeration: Head wiggle exaggeration factor
        unsquinch_fix: Offset for eyeBlink and eyeSquint
        eye_contact_fix: Offset for eyeLook
        exaggerate_above: Minimum value to exaggerate
        symmetrize_eyes: Average eye blink between left and right

    Requires:
        - If input_path is a video, the video extension must be .mp4
    """
    file_is_video = input_path.suffix == ".mp4"
    out, mean_step_time, mean_rtf, time_to_first_sound = pipeline.infer_path(
        input_path,
        min_audio_samples_per_step,
        max_audio_samples_per_step,
        mouth_exaggeration,
        brow_exaggeration,
        head_wiggle_exaggeration,
        unsquinch_fix,
        eye_contact_fix,
        exaggerate_above,
        symmetrize_eyes,
        max_audio_duration,
    )
    out = out.squeeze().cpu().numpy()
    blendshapes, head_angles = out[:, :52], out[:, 52:]
    euler_angles = unscale_and_uncenter_head_angles(head_angles)
    print("Rendering video...")
    temp_output_path = Path("render_output/temp_output.mp4")
    scene.render_sequence(blendshapes, euler_angles, temp_output_path)
    if file_is_video:
        combine_videos_side_by_side(input_path, temp_output_path, output_file)
    else:
        add_audio_to_video(temp_output_path, input_path, output_file)
    return mean_step_time, mean_rtf, time_to_first_sound
