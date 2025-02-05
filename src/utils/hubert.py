import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

from src.utils.constants import N_AUDIO_SAMPLES_PER_VIDEO_FRAME
from typing import Optional


class HubertExtractor:
    def __init__(
        self,
        model_name,
        return_attention_mask,
        feature_dim,
        device: torch.device,
    ):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
        self.model.to(device).eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.device = device
        # for information about how this param affects masking
        # see https://huggingface.co/docs/transformers/en/model_doc/hubert#transformers.HubertModel.forward.attention_mask
        self.return_attention_mask = return_attention_mask
        self.feature_dim = feature_dim

    def __call__(
        self,
        audio: torch.Tensor,
        audio_lens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if audio_lens is not None and self.return_attention_mask:
            attention_mask = self.make_attention_mask(audio_lens, audio.size(1))
        else:
            attention_mask = None
        inputs = (
            self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            .input_values.squeeze(0)
            .to(audio.device)
        )
        with torch.no_grad():
            model_output = self.model(
                inputs, attention_mask=attention_mask
            ).last_hidden_state
            if audio_lens is not None:
                output_mask = self.make_output_mask(audio_lens, model_output.size(1))
            else:
                output_mask = None
        return model_output, output_mask

    def make_attention_mask(
        self, audio_lens: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        # as per https://huggingface.co/docs/transformers/en/model_doc/hubert#transformers.HubertModel.forward.attention_mask,
        # the attention mask should be 0 for padding tokens and 1 for the rest
        # mask is torch.LongTensor of shape (batch_size, sequence_length)
        mask = torch.arange(max_len).to(self.device)
        mask = mask.unsqueeze(0).expand(audio_lens.size(0), -1)
        mask = mask < audio_lens.unsqueeze(1)
        return mask.long()

    def make_output_mask(self, audio_lens: torch.Tensor, max_len: int) -> torch.Tensor:
        # output mask should be True for padding tokens and False for the rest
        output_lens = audio_lens // 320
        max_output_len = max(output_lens)
        output_mask_shape = (audio_lens.size(0), max_output_len)
        output_mask = torch.arange(max_output_len).to(self.device)
        output_mask = output_mask.unsqueeze(0).expand(*output_mask_shape)
        output_mask = output_mask < output_lens.unsqueeze(1)
        return output_mask
