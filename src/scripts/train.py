import multiprocessing

# hopefully this fixes the train loop segfaults
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

from re import T
import torch
import argparse
import json
import gc
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from src.data.dataset import HeadDataset, collate_fn
from src.model.ema import EMA
from src.model.inference import PytorchInferencePipeline, render_inference
from src.model.simple import SimpleDecoder, load_from_dict
from src.model.loss import masked_difference_loss
from src.utils.constants import HEAD_ANGLE_NAMES
from src.utils.landmarks import unscale_and_uncenter_head_angles
from src.utils.rendering import Scene
from src.utils.scheduler import WarmupScheduler
from src.utils.video import combine_videos_side_by_side


def calculate_teacher_forcing_prob(
    step: int, start_step: int, end_step: int, start_prob: float, end_prob: float
) -> float:
    """
    Calculates the teacher forcing probability based on the current step.

    Args:
        step: The current step.
        start_step: The step at which teacher forcing probability starts.
        end_step: The step at
        start_prob: The initial teacher forcing probability.
        end_prob: The final teacher forcing probability.

    Returns:
        float: The teacher forcing probability.
    """
    if step < start_step:
        return start_prob
    elif step >= end_step:
        return end_prob
    else:
        return start_prob + (end_prob - start_prob) * (step - start_step) / (
            end_step - start_step
        )


def prepare_tb_logging(path: Path) -> SummaryWriter:
    """
    Creates a tensorboard logger at the given path.

    Args:
        path: The path to the directory where the logs will be stored.

    Returns:
        SummaryWriter: The tensorboard logger.
    """
    logdir_path = Path(path)
    logdir_path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(logdir_path, flush_secs=10)


def make_example_video(
    vecs: torch.tensor, reference_video_path: Path, output_file: Path, scene: Scene
):
    """
    Creates a video from model output.

    Args:
        vecs: The model output.
        reference_video_path: The path to the video whose audio was used to generate the
            model output.
        output_file: The path to save the output video.
        scene: The scene object to use for rendering.
    """
    print(f"Creating example video at {output_file}...")
    vecs = vecs.cpu()
    blendshapes, head_angles = vecs[:, :52], vecs[:, 52:]
    euler_angles = unscale_and_uncenter_head_angles(head_angles)
    scene.render_sequence(blendshapes, euler_angles, output_file)
    combine_videos_side_by_side(reference_video_path, output_file, output_file)


def save_checkpoint(state: dict, save_path: Path):
    """
    Consumes a generic state dictionary. Unpacks state_dict
    for each element of state if required.

    Args:
        state: The state dictionary to save.
        save_path: The path to save the state dictionary
    """

    if "model" in state:
        # we need to call state_dict() on all ranks in case it is calling all_gather
        model = state["model"]

    checkpoint = {}
    for k, v in state.items():
        if hasattr(v, "state_dict"):
            checkpoint[k] = v.state_dict()
        else:
            checkpoint[k] = v
    torch.save(checkpoint, save_path)

    if "model" in state:
        state["model"] = model


def load_checkpoint(state: dict, load_path: Path, device: torch.device):
    """
    Updates a generic state dictionary. Takes the items in 'checkpoint', and pushes them
    into the preloaded state values

    Args:
        state: The state dictionary to update.
        load_path: The path to the checkpoint to load.
        device: The device to load the checkpoint to.
    """
    checkpoint = torch.load(load_path, map_location=device)
    for k, v in state.items():
        if hasattr(v, "load_state_dict"):
            v.load_state_dict(checkpoint[k])
        else:
            state["start_checkpoint_state_dict"][k] = checkpoint[k]
    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loaded checkpoint from {load_path}")

    gc.collect()


def step_model(
    batch: tuple,
    model: SimpleDecoder,
    device: torch.device,
    mask_audio_padding: bool,
    teacher_forcing_prob: float,
    loss_metric: str,
) -> tuple:
    """
    Runs a single step of the model.

    Args:
        batch: The batch of data.
        model: The model to run.
        device: The device to run the model on.
        mask_audio_padding: Whether to mask audio padding in the model's forward method.
        teacher_forcing_prob: The teacher forcing probability.

    Returns:
        tuple: The loss, predictions, and video paths.
    """
    (
        blendshapes_tgts,
        head_angles_tgts,
        tgt_lens,
        audios,
        audio_lens,
        bad_frame_masks,
        video_paths,
        front_pad_blendshape,
        front_pad_head_angle,
    ) = batch
    front_pad_blendshape = front_pad_blendshape.to(device)
    front_pad_head_angle = front_pad_head_angle.to(device)
    blendshapes_tgts = blendshapes_tgts.to(device)
    head_angles_tgts = head_angles_tgts.to(device)
    start_frame = torch.cat((front_pad_blendshape, front_pad_head_angle), dim=-1)
    tgts = torch.cat((blendshapes_tgts, head_angles_tgts), dim=-1)
    audios = audios.to(device)
    tgt_lens = tgt_lens.to(device)
    if mask_audio_padding:
        audio_lens = audio_lens.to(device)
    else:
        audio_lens = None
    bad_frame_masks = bad_frame_masks.to(device)

    preds = model(
        audios,
        tgts,
        start_frame,
        tgt_lens,
        audio_lens,
        bad_frame_masks,
        teacher_forcing_prob=teacher_forcing_prob,
    )

    loss, shapekey_dict = masked_difference_loss(
        preds,
        tgts,
        bad_frame_masks,
        tgt_lens,
        device,
        calculate_shapekey_dict=True,
        metric=loss_metric,
    )
    return loss, preds, video_paths, shapekey_dict


def validate(
    data_config: dict,
    n_ctx: int,
    device: torch.device,
    model: SimpleDecoder,
    step: int,
    epoch: int,
    mask_audio_padding: bool,
    zero_padding_prob: float,
    loss_metric: str,
    num_video_examples: int = 4,
    num_pipeline_examples: int = 2,
) -> tuple[float, dict, list, list]:
    """
    Validates the model on the validation set. Returns:
      - mean_val_loss
      - mean_shapekey_dict
      - pipeline_render_jobs (list of jobs for render_inference)
      - example_render_jobs (list of jobs for make_example_video)
    """
    model.eval()
    dataset = HeadDataset(
        data_config["val"]["data_dir"],
        n_ctx,
        device,
        random_start_idx=False,
        zero_padding_prob=zero_padding_prob,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, collate_fn=collate_fn, **data_config["val"]["dataloader"]
    )
    total_loss = 0
    shapekey_dicts = []

    pipeline_render_jobs = []
    example_render_jobs = []

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")
        for batch_idx, batch in pbar:
            loss, preds, video_paths, shapekey_dict = step_model(
                batch, model, device, mask_audio_padding, 0.0, loss_metric
            )
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
            total_loss += loss.item()

            if shapekey_dict is not None:
                shapekey_dicts.append(shapekey_dict)

            if batch_idx < num_video_examples:
                ref_video_path = Path(video_paths[0])
                if batch_idx < num_pipeline_examples:
                    pipeline_render_jobs.append(
                        {
                            "reference_video_path": ref_video_path,
                            "step": step,
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                        }
                    )
                else:
                    example_render_jobs.append(
                        {
                            "preds": preds[0].cpu(),
                            "reference_video_path": ref_video_path,
                            "step": step,
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                        }
                    )

    mean_val_loss = total_loss / len(dataloader)
    print(f"Mean validation loss: {mean_val_loss:.3f}")

    # Compute average shapekey dictionary
    mean_shapekey_dict = {}
    if shapekey_dicts:
        for shapekey in shapekey_dicts[0]:
            mean_shapekey_dict[shapekey] = sum(
                d[shapekey] for d in shapekey_dicts
            ) / len(shapekey_dicts)

    return mean_val_loss, mean_shapekey_dict, pipeline_render_jobs, example_render_jobs


def train(
    warmup_steps: int,
    data_config: dict,
    device: str,
    model_config: dict,
    lr: float,
    lr_patience: int,
    train_param_dict: dict,
    max_epochs: int,
    val_steps: int,
    log_tb_every: int,
    log_dir: str,
    render_params: dict,
    checkpoint_every: int,
    start_checkpoint: str,
    val_per_epoch: int,
    weight_decay: float,
    max_grad_norm: float,
    mask_audio_padding: bool,
    teacher_forcing: dict,
    zero_padding_prob: float,
    loss_metric: str,
    save_ema: bool,
    fp16: bool,
    val_params: dict,
):
    """
    Trains the model with learning rate warmup.
    """
    scaler = GradScaler(enabled=fp16)
    dataset = HeadDataset(
        data_config["train"]["data_dir"],
        model_config["decoder_dimensions"]["n_ctx"],
        device,
        data_config["train"]["random_start_idx"],
        zero_padding_prob=zero_padding_prob,
    )
    device = torch.device(device)
    model = load_from_dict(model_config, device)
    if save_ema:
        ema = EMA(model, decay=0.9999, device=device)
    param_in_millions = sum(p.numel() for p in model.parameters()) / 1_000_000
    print(f"Model has {param_in_millions:.2f}M parameters")
    dataloader = DataLoader(
        dataset, collate_fn=collate_fn, **data_config["train"]["dataloader"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        base_lr=lr,
        factor=0.5,
        patience=lr_patience,
    )

    state = {
        "model": model,
        "optimizer": optimizer,
        "step": 0,
        "epoch": 0,
        "best_val_loss": float("inf"),
        "train_param_dict": train_param_dict,
        "scheduler": scheduler,
        "start_checkpoint_state_dict": {},
    }

    if start_checkpoint is not None:
        print(f"Checkpoint: {start_checkpoint}")
        load_checkpoint(state, Path(start_checkpoint), device=device)

    if Path(log_dir).exists():
        raise FileExistsError(f"Run directory {log_dir} already exists")

    tb_logger = prepare_tb_logging(log_dir)
    tb_logger.add_text("config", json.dumps(train_param_dict, indent=2))

    checkpoint_dir = f"{log_dir}/checkpoints"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    val_scene = Scene(Path(render_params["shape_keys_path"]), device)

    if val_per_epoch is not None:
        val_interval = len(dataloader) // val_per_epoch
    else:
        val_interval = val_steps

    for epoch in range(max_epochs):
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch}",
        )

        for batch_idx, batch in pbar:
            model.train()
            optimizer.zero_grad()
            with autocast(enabled=fp16):
                teacher_forcing_prob = calculate_teacher_forcing_prob(
                    state["step"],
                    teacher_forcing["start_step"],
                    teacher_forcing["end_step"],
                    teacher_forcing["start_prob"],
                    teacher_forcing["end_prob"],
                )
                loss, _, _, shapekey_dict = step_model(
                    batch,
                    model,
                    device,
                    mask_audio_padding,
                    teacher_forcing_prob,
                    loss_metric,
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )

            if state["step"] % log_tb_every == 0:
                tb_logger.add_scalar("gradients/total_norm", total_norm, state["step"])
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        tb_logger.add_scalar(
                            f"gradients/{name}_norm", grad_norm, state["step"]
                        )
                        tb_logger.add_histogram(
                            f"gradients/{name}_hist", param.grad, state["step"]
                        )
                        tb_logger.add_histogram(
                            f"parameters/{name}_hist", param.data, state["step"]
                        )

            scaler.step(optimizer)
            scaler.update()

            if save_ema:
                ema.update(model)

            scheduler.step(None)

            state["step"] += 1

            pbar.set_postfix({"loss": f"{loss.item():.3f}", "step": state["step"]})

            if state["step"] % log_tb_every == 0:
                tb_logger.add_scalar("train/loss", loss.item(), state["step"])
                tb_logger.add_scalar(
                    "train/lr", scheduler.get_last_lr()[0], state["step"]
                )
                tb_logger.add_scalar("train/epoch", epoch, state["step"])
                tb_logger.add_scalar(
                    "train/teacher_forcing_prob", teacher_forcing_prob, state["step"]
                )
                for shapekey in shapekey_dict:
                    prefix = (
                        "train/shapekey_"
                        if shapekey not in HEAD_ANGLE_NAMES
                        else "train/head_"
                    )
                    tb_logger.add_scalar(
                        f"{prefix}{shapekey}",
                        shapekey_dict[shapekey],
                        state["step"],
                    )

            if state["step"] % val_interval == 0:
                val_loss, mean_shapekey_dict, pipeline_jobs, example_jobs = validate(
                    data_config,
                    model_config["decoder_dimensions"]["n_ctx"],
                    device,
                    model,
                    state["step"],
                    epoch,
                    mask_audio_padding,
                    zero_padding_prob,
                    loss_metric,
                    num_video_examples=val_params["num_video_examples"],
                    num_pipeline_examples=val_params["num_pipeline_examples"],
                )
                # Render in main process to prevent segfaults hopefully?
                output_video_dir = Path(log_dir) / "videos"
                output_video_dir.mkdir(parents=True, exist_ok=True)

                render_pipeline = PytorchInferencePipeline(
                    model,
                    model.n_ctx,
                    render_params["pipeline_crossfade"],
                    1,
                    device,
                )

                for job in pipeline_jobs:
                    output_file = (
                        output_video_dir
                        / f"step_{job['step']}_epoch_{job['epoch']}_pipeline_{job['batch_idx']}.mp4"
                    )
                    render_inference(
                        render_pipeline,
                        val_scene,
                        job["reference_video_path"],
                        render_params["min_audio_samples_per_step"],
                        render_params["max_audio_samples_per_step"],
                        output_file,
                        max_audio_duration=render_params["max_audio_duration"],
                    )
                    print(f"Rendered pipeline video: {output_file}")

                for job in example_jobs:
                    output_file = (
                        output_video_dir
                        / f"step_{job['step']}_epoch_{job['epoch']}_{job['batch_idx']}.mp4"
                    )
                    make_example_video(
                        job["preds"],
                        job["reference_video_path"],
                        output_file,
                        val_scene,
                    )
                    print(f"Rendered example video: {output_file}")
                tb_logger.add_scalar("val/loss", val_loss, state["step"])
                for shapekey in mean_shapekey_dict:
                    prefix = (
                        "val/shapekey_"
                        if shapekey not in HEAD_ANGLE_NAMES
                        else "val/head_"
                    )
                    tb_logger.add_scalar(
                        f"{prefix}{shapekey}",
                        mean_shapekey_dict[shapekey],
                        state["step"],
                    )
                # Pass validation loss for ReduceLROnPlateau phase
                scheduler.step(val_loss)
                if val_loss < state["best_val_loss"]:
                    print(f"Saving new best model with val loss {val_loss}")
                    state["best_val_loss"] = val_loss
                    save_checkpoint(
                        state,
                        f"{checkpoint_dir}/bestval.pt",
                    )
            if state["step"] % checkpoint_every == 0:
                checkpoint_path = (
                    f"{checkpoint_dir}/epoch_{state['epoch']}_step_{state['step']}.pt"
                )
                if save_ema:
                    state["model"] = ema.model
                save_checkpoint(state, checkpoint_path)
                print(f"Saved checkpoint at {checkpoint_path}")

        start_batch = 0
        print(f"Epoch {state['epoch']} completed")
        state["epoch"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/example.json"))
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    train(train_param_dict=config, **config)


if __name__ == "__main__":
    main()
