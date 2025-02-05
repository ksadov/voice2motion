import math
import torch
from torch import nn
from dataclasses import dataclass
from pathlib import Path

from src.utils.constants import HEAD_LANDMARK_DIM
from src.utils.hubert import HubertExtractor


@dataclass
class DecoderDimensions:
    n_ctx: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    activation: str


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, offset=0):
        x = x + self.pe[:, offset : offset + x.size(1)]
        return self.dropout(x)


class SplitOutputProjection(nn.Module):
    def __init__(self, d_model, main_dim=52, aux_dim=3):
        super().__init__()
        self.main_projection = nn.Sequential(nn.Linear(d_model, main_dim), nn.Sigmoid())
        self.aux_projection = nn.Sequential(
            nn.Linear(d_model, aux_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        main_output = self.main_projection(x)
        aux_output = self.aux_projection(x)
        return torch.cat([main_output, aux_output], dim=-1)


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        device,
        decoder_dimensions: DecoderDimensions,
        landmark_dim: int,
        hubert_params: dict,
    ):
        super().__init__()

        self.decoder_dimensions = decoder_dimensions
        self.hubert_extractor = HubertExtractor(device=device, **hubert_params)
        self.audio_proj = nn.Linear(
            self.hubert_extractor.feature_dim, decoder_dimensions.d_model
        )
        self.landmark_proj = nn.Linear(landmark_dim, decoder_dimensions.d_model)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dimensions.d_model,
            nhead=decoder_dimensions.nhead,
            dim_feedforward=decoder_dimensions.dim_feedforward,
            dropout=decoder_dimensions.dropout,
            activation=decoder_dimensions.activation,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer, num_layers=decoder_dimensions.num_layers
        )
        self.output_projection = SplitOutputProjection(decoder_dimensions.d_model)

        self.positional_encoding = PositionalEncoding(
            decoder_dimensions.d_model, decoder_dimensions.dropout
        )

        self.device = device
        self.n_ctx = decoder_dimensions.n_ctx
        self._init_parameters()
        self.to(device)

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, tgt_lens, max_tgt_len, bad_frame_masks=None):
        tgt_mask = (
            torch.arange(max_tgt_len)[None, :].to(self.device) >= tgt_lens[:, None]
        )
        tgt_mask = tgt_mask.to(self.device)
        if bad_frame_masks is not None:
            tgt_mask = tgt_mask | bad_frame_masks
        return tgt_mask

    def create_causal_mask(self, size):
        # Lower triangular mask for autoregressive generation
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def hubert_encode(self, audio, audio_lens=None):
        src = audio.to(self.device)
        src, src_mask = self.hubert_extractor(audio, audio_lens)
        return src, src_mask

    def simple_encode(self, src):
        src = self.audio_proj(src)
        src = self.positional_encoding(src)
        return src

    def encode(self, audio, audio_lens=None):
        src, src_mask = self.hubert_encode(audio, audio_lens)
        src = self.simple_encode(src)
        return src, src_mask

    def forward_non_autoregressive(
        self,
        audio,
        tgt_landmarks,
        start_frame,
        tgt_lens=None,
        audio_lens=None,
        bad_frame_masks=None,
    ):
        src, src_mask = self.encode(audio, audio_lens)

        # First concatenate
        tgt = torch.cat([start_frame.unsqueeze(1), tgt_landmarks], dim=1)

        # Then project the full sequence
        tgt = self.landmark_proj(tgt)
        tgt = self.positional_encoding(tgt)

        tgt_padding_mask = None
        if tgt_lens is not None:
            # Create padding mask for landmarks
            landmark_padding_mask = self.create_padding_mask(
                tgt_lens, tgt_landmarks.size(1), bad_frame_masks
            )
            # Add mask value for start frame
            tgt_padding_mask = torch.cat(
                [
                    torch.ones(
                        tgt_landmarks.size(0), 1, device=landmark_padding_mask.device
                    ),
                    landmark_padding_mask,
                ],
                dim=1,
            )

        tgt_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)

        output = self.transformer_decoder(
            tgt,
            src,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_mask,
        )

        # Project back to output dimension
        projected_out = self.output_projection(output)
        return projected_out[:, 1:]

    def forward(
        self,
        audio,
        tgt_landmarks,
        start_frame,
        tgt_lens=None,
        audio_lens=None,
        bad_frame_masks=None,
        teacher_forcing_prob=1.0,
    ):
        """
        Run a forward pass with scheduled sampling.

        Args:
            audio: Audio input tensor [B, Audio_length]
            tgt_landmarks: Ground truth landmarks [B, T, D]
            start_frame: Initial frame to start decoding from [B, D]
            tgt_lens: Lengths of each target sequence [B]
            audio_lens: Lengths of each audio sequence [B]
            bad_frame_masks: Boolean masks indicating invalid frames [B, T]
            teacher_forcing_prob: Probability of using ground-truth targets as input for the next step.
                                If 1.0, always uses ground truth (teacher forcing).
                                If 0.0, always uses model predictions (fully autoregressive).

        Returns:
            Predicted landmarks [B, T, D]
        """
        if teacher_forcing_prob == 1.0:
            return self.forward_non_autoregressive(
                audio, tgt_landmarks, start_frame, tgt_lens, audio_lens, bad_frame_masks
            )
        src, src_mask = self.encode(audio, audio_lens)
        B, T, D = tgt_landmarks.shape

        preds = []
        decoder_inputs = [start_frame.unsqueeze(1)]

        for t in range(T):
            partial_input = torch.cat(decoder_inputs, dim=1)
            partial_input = self.landmark_proj(partial_input)
            partial_input = self.positional_encoding(partial_input)

            # Slice masks appropriately
            current_bad_frame_masks = (
                bad_frame_masks[:, :t] if bad_frame_masks is not None else None
            )
            tgt_padding_mask = None
            if tgt_lens is not None:
                tgt_padding_mask = self.create_padding_mask(
                    tgt_lens, t, current_bad_frame_masks
                )
                # pad front with 1 to account for start_frame
                tgt_padding_mask = torch.cat(
                    [
                        torch.ones(B, 1, device=tgt_padding_mask.device),
                        tgt_padding_mask,
                    ],
                    dim=1,
                )

            tgt_mask = self.create_causal_mask(t + 1).to(src.device)

            output = self.transformer_decoder(
                partial_input,
                src,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_mask,
            )

            output_frame = self.output_projection(output[:, -1:])
            preds.append(output_frame)

            use_ground_truth = torch.rand(B, device=src.device) < teacher_forcing_prob
            next_input = torch.where(
                use_ground_truth.unsqueeze(-1),
                tgt_landmarks[:, t],
                output_frame.squeeze(1),
            )

            decoder_inputs.append(next_input.unsqueeze(1))

        preds = torch.cat(preds, dim=1)

        return preds

    def decoder_step(self, src, decoder_in):
        """
        Run a single step of the decoder.

        Args:
            src: Embedded and positional encoded audio features of shape [B, T_a, self.decoder_dimensions.d_model]
            decoder_in: Initial decoder input of shape [B, T_l, self.landmark_dim]

        Returns:
            Predicted landmarks of shape [B, 1, self.landmark_dim]
        """
        tgt = self.landmark_proj(decoder_in)
        tgt = self.positional_encoding(tgt)
        tgt_mask = self.create_causal_mask(tgt.size(1)).to(self.device)
        out = self.transformer_decoder(tgt, src, tgt_mask=tgt_mask)
        next_output = self.output_projection(out[:, -1:])
        return next_output


def load_from_dict(model_dict, device):
    decoder_dimensions = DecoderDimensions(**model_dict["decoder_dimensions"])
    # if model dict doesn't have hubert_params, set it to an empty dict
    if "hubert_params" not in model_dict:
        model_dict["hubert_params"] = {
            "model_name": "facebook/hubert-base-ls960",
            "return_attention_mask": False,
            "feature_dim": 768,
        }
    model = SimpleDecoder(
        device, decoder_dimensions, HEAD_LANDMARK_DIM, model_dict["hubert_params"]
    )
    return model


def load_from_checkpoint(checkpoint_path: Path, device: torch.device) -> SimpleDecoder:
    """
    Load model from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_param_dict = checkpoint["train_param_dict"]
    model = load_from_dict(train_param_dict["model_config"], device)
    model.load_state_dict(checkpoint["model"])
    return model
