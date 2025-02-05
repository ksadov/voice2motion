import json
import os
import torch
import glob
import numpy as np

from torch.utils.data import Dataset
from torch.nn import functional as F

from src.utils.audio import load_audio_from_video
from src.utils.constants import N_AUDIO_SAMPLES_PER_VIDEO_FRAME
from src.utils.landmarks import scale_and_center_head_angles, regularize_blendshapes


class HeadDataset(Dataset):
    def __init__(
        self,
        data_dir,
        n_ctx,
        device,
        random_start_idx,
        zero_padding_prob,
    ):
        self.data_dir = data_dir
        # recursively glob for all npz files in the data_dir
        info_clips = glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
        # sort the clips to ensure they are in the same order
        info_clips.sort()
        self.info_clips = info_clips
        self.device = device
        self.n_ctx = n_ctx
        self.random_start_idx = random_start_idx
        self.zero_padding_prob = zero_padding_prob

    def __len__(self):
        return len(self.info_clips)

    def __getitem__(self, idx):
        clip_info = np.load(self.info_clips[idx])
        blendshapes = clip_info["blendshapes"].astype(np.float32)
        head_angles = clip_info["head_angles"].astype(np.float32)
        bad_frames = clip_info["bad_frames"].astype(np.int32)
        # n_ctx frames for the target, plus one as the start frame
        frames_needed = self.n_ctx + 1
        tgt_len = min(blendshapes.shape[0] - 1, self.n_ctx)
        if self.random_start_idx and len(blendshapes) > frames_needed:
            start_idx = np.random.randint(0, len(blendshapes) - frames_needed)
        else:
            start_idx = 0
        bad_frames_for_format = [
            bf for bf in (bad_frames - start_idx) if bf > 0 and bf < frames_needed
        ]
        assert (
            blendshapes.shape[0] == head_angles.shape[0]
        ), "Mismatched blendshape and head angles lengths"
        blendshapes_tgt = np.zeros((self.n_ctx, blendshapes.shape[1]), dtype=np.float32)
        head_angles_tgt = np.zeros((self.n_ctx, head_angles.shape[1]), dtype=np.float32)
        formatted_blendshapes = regularize_blendshapes(
            blendshapes[start_idx : frames_needed + start_idx], bad_frames_for_format
        )
        formatted_head_angles = scale_and_center_head_angles(
            head_angles[start_idx : frames_needed + start_idx], bad_frames_for_format
        )
        bad_frames = [
            bf - 1 for bf in bad_frames_for_format if bf - 1 < tgt_len and bf - 1 >= 0
        ]
        blendshapes_tgt[:tgt_len] = formatted_blendshapes[1:]
        head_angles_tgt[:tgt_len] = formatted_head_angles[1:]
        if np.random.rand() < self.zero_padding_prob:
            front_pad_blendshape = np.zeros_like(formatted_blendshapes[0])
            front_pad_head_angle = np.zeros_like(formatted_head_angles[0])
        else:
            front_pad_blendshape = formatted_blendshapes[0]
            front_pad_head_angle = formatted_head_angles[0]
        video_fname = clip_info["original_fname"].item()
        # audio should line up with tgt
        audio_len = tgt_len * N_AUDIO_SAMPLES_PER_VIDEO_FRAME
        audio_start_frame = (start_idx + 1) * N_AUDIO_SAMPLES_PER_VIDEO_FRAME
        audio_end_frame = audio_start_frame + audio_len
        assert np.all(blendshapes_tgt[bad_frames] == 0)
        try:
            audio = load_audio_from_video(video_fname)
            audio = audio[audio_start_frame:audio_end_frame]
            audio_len = len(audio)
            return (
                blendshapes_tgt,
                head_angles_tgt,
                tgt_len,
                audio,
                audio_len,
                bad_frames,
                video_fname,
                front_pad_blendshape,
                front_pad_head_angle,
            )
        except Exception as e:
            print(f"Error processing video {video_fname}: {e}")
            return video_fname


def collate_fn(batch):
    (
        blendshapes,
        head_angles,
        tgt_lens,
        audios,
        audio_lens,
        bad_frames,
        video_fnames,
        front_pad_blendshape,
        front_pad_head_angle,
    ) = zip(*batch)
    tgt_lens = torch.tensor(tgt_lens)
    audio_lens = torch.tensor(audio_lens)
    front_pad_blendshape = torch.tensor(front_pad_blendshape)
    front_pad_head_angle = torch.tensor(front_pad_head_angle)
    # pad blendshapes and head angles
    max_tgt_len = max(tgt_lens)
    blendshapes = torch.stack(
        [
            F.pad(torch.tensor(bs), (0, 0, 0, max_tgt_len - len(bs)))
            for bs in blendshapes
        ]
    )
    head_angles = torch.stack(
        [
            F.pad(torch.tensor(ha), (0, 0, 0, max_tgt_len - len(ha)))
            for ha in head_angles
        ]
    )
    # pad audio
    max_audio_len = max(audio_lens)
    audios = torch.stack(
        [F.pad(torch.tensor(a), (0, max_audio_len - len(a))) for a in audios]
    )
    # create bad frames mask
    bad_frames_masks = torch.zeros(len(blendshapes), max_tgt_len, dtype=torch.bool)
    for i, bad_frames_i in enumerate(bad_frames):
        bad_frames_masks[i, bad_frames_i] = True
    return (
        blendshapes,
        head_angles,
        tgt_lens,
        audios,
        audio_lens,
        bad_frames_masks,
        video_fnames,
        front_pad_blendshape,
        front_pad_head_angle,
    )


def test_dataset():
    data_dir = "data/AVSpeechProcessed/test"
    n_mels = 80
    device = "cuda"
    dataset = HeadDataset(data_dir, n_mels, 30, True, 0, 0, device)
    print("Dataset length:", len(dataset))


if __name__ == "__main__":
    test_dataset()
