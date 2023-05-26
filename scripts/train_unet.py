# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py

import argparse
import os
import pickle
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset, load_from_disk
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    UNet2DModel,
)
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "audiodiffusion"))

import audiodiffusion.mel as mel_helper
from diffusers.training_utils import EMAModel
from huggingface_hub import HfFolder, Repository, whoami
from librosa.util import normalize
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm.auto import tqdm

from audiodiffusion.pipeline_audio_diffusion import AudioDiffusionPipeline

logger = get_logger(__name__)


class AudioDiffusion(Dataset):
    def __init__(self, data_folder):
        super().__init__()
        self.images = []
        self.labels = []
        file_names = os.listdir(data_folder)

        for i in range(len(file_names)):
            file_path = os.path.join(data_folder, file_names[i])
            file_data = np.load(file_path)
            image, label = file_data["image"], file_data["text_embed"]
            image = image.transpose(2, 0, 1)
            self.images.append(torch.tensor(image))
            self.labels.append(torch.tensor(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}


def main(args):
    output_dir = os.environ.get("SM_MODEL_DIR", None) or args.output_dir
    logging_dir = os.path.join(output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    dataset = AudioDiffusion("../project/data/development-full/")
    resolution = dataset.images[0].shape[1], dataset.images[0].shape[2]

    augmentations = Compose(
        [
            ToTensor(),
            Normalize([0.5], [0.5]),
        ]
    )

    train_dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2
    )

    # if args.encodings is not None:
    #     encodings = pickle.load(open(args.encodings, "rb"))
    encodings = dataset.labels

    vqvae = None
    if args.vae is not None:
        try:
            vqvae = AutoencoderKL.from_pretrained(args.vae)
        except EnvironmentError:
            vqvae = AudioDiffusionPipeline.from_pretrained(args.vae).vqvae
        # Determine latent resolution
        with torch.no_grad():
            latent_resolution = (
                vqvae.encode(torch.zeros((1, 1) + resolution))
                .latent_dist.sample()
                .shape[2:]
            )

    if args.from_pretrained is not None:
        pipeline = AudioDiffusionPipeline.from_pretrained(args.from_pretrained)
        mel = pipeline.mel
        model = pipeline.unet
        if hasattr(pipeline, "vqvae"):
            vqvae = pipeline.vqvae

    else:
        model = UNet2DConditionModel(
            sample_size=resolution if vqvae is None else latent_resolution,
            in_channels=1 if vqvae is None else vqvae.config["latent_channels"],
            out_channels=1 if vqvae is None else vqvae.config["latent_channels"],
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=encodings[0].shape[-1],
        )

    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_steps)
    else:
        noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs)
        // args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    ema_model = EMAModel(
        getattr(model, "module", model),
        inv_gamma=args.ema_inv_gamma,
        power=args.ema_power,
        max_value=args.ema_max_decay,
    )

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    mel = mel_helper.Mel(
        x_res=resolution[1],
        y_res=resolution[0],
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
    )

    global_step = 0
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        if epoch < args.start_epoch:
            for step in range(len(train_dataloader)):
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1
            if epoch == args.start_epoch - 1 and args.use_ema:
                ema_model.optimization_step = global_step
            continue

        model.train()
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["image"].type(torch.float32)
            labels = batch["label"].type(torch.float32)

            if vqvae is not None:
                vqvae.to(clean_images.device)
                with torch.no_grad():
                    clean_images = vqvae.encode(clean_images).latent_dist.sample()
                # Scale latent images to ensure approximately unit variance
                clean_images = clean_images * 0.18215

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, labels)["sample"]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_model.step(model)
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if (
                (epoch + 1) % args.save_model_epochs == 0
                or (epoch + 1) % args.save_images_epochs == 0
                or epoch == args.num_epochs - 1
            ):
                unet = accelerator.unwrap_model(model)
                if args.use_ema:
                    ema_model.copy_to(unet.parameters())
                pipeline = AudioDiffusionPipeline(
                    vqvae=vqvae,
                    unet=unet,
                    mel=mel,
                    scheduler=noise_scheduler,
                )

            if (
                epoch + 1
            ) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                pipeline.save_pretrained(output_dir)

                # save the model
                if args.push_to_hub:
                    repo.push_to_hub(
                        commit_message=f"Epoch {epoch}",
                        blocking=False,
                        auto_lfs_prune=True,
                    )

            if (epoch + 1) % args.save_images_epochs == 0:
                generator = torch.Generator(device=clean_images.device).manual_seed(42)

                random.seed(42)
                encoding = torch.stack(
                    random.sample(encodings, args.eval_batch_size)
                ).to(clean_images.device)

                # run pipeline in inference (sample random noise and denoise)
                images, (sample_rate, audios) = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    return_dict=False,
                    encoding=encoding,
                )

                # denormalize the images and save to tensorboard
                images = np.array(
                    [
                        np.frombuffer(image.tobytes(), dtype="uint8").reshape(
                            (len(image.getbands()), image.height, image.width)
                        )
                        for image in images
                    ]
                )

                accelerator.trackers[0].writer.add_images("test_samples", images, epoch)
                for _, audio in enumerate(audios):
                    accelerator.trackers[0].writer.add_audio(
                        f"test_audio_{_}",
                        normalize(audio),
                        epoch,
                        sample_rate=sample_rate,
                    )
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument("--output_dir", type=str, default="ddpm-model-64")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--use_auth_token", type=bool, default=False)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", type=bool, default=False)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--from_pretrained", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--scheduler", type=str, default="ddpm", help="ddpm or ddim")
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="pretrained VAE model for latent diffusion",
    )
    parser.add_argument(
        "--encodings",
        type=str,
        default=None,
        help="picked dictionary mapping audio_file to encoding",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    main(args)
