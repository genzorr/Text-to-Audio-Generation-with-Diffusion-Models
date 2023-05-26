Text-to-Audio-Generation-with-Diffusion-Models
Welcome to the project repository!
This is the final project for Skoltech Deep Learning course.

Description
The project aims to solve text to audio generation task using latent diffusion.
We operate with with mel spectrograms of audios. 
At the second step we generate text embeddings using pretrained CLAP model 
At the first step we train VAE to generate audio embeddings
At the third and final step we train u-net based latent duffusion model with cross attention blocks using audio and text embeddings.

```clap_text.py``` contains example of usage of CLAP model to get text embeddings.

```generate.ipynb``` notebook provides an example of usage of a fully trained model (unet + vae).

Weights of trained model are located in model/ folder (uploaded to GDrive: https://drive.google.com/drive/folders/1qw-OSGRCsU3gTdBb3K_E3jarJpyGvYjy?usp=sharing).

lightning_logs/ folder contains logs of vae training process.

VAE training
```sh
python3 scripts/train_vae.py \
--dataset_name /path/to/dev/data \
--save_images_batches 100 \
--max_epochs 10 \
--batch_size 2 \
--gradient_accumulation_steps 12 \
```

Latent diffusion training
```sh
accelerate launch --config_file config/accelerate_local.yaml \
scripts/train_unet.py \
--hop_length 1024 \
--output_dir models/ddpm-ema-audio-64 \
--train_batch_size 1 \
--num_epochs 10 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-4 \
--lr_warmup_steps 500 \
--mixed_precision no \
--vae models/autoencoder-kl \
--save_images_epochs 4
```



Acknowledgments
1) https://github.com/teticio/audio-diffusion/tree/main
2) https://github.com/LAION-AI
3) https://huggingface.co/docs/transformers/model_doc/clap#transformers.ClapTextModel
4) https://huggingface.co/laion/clap-htsat-unfuse
