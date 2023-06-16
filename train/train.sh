#!/bin/bash 
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -N G1C2_melon
#PBS -q pleiades3
#PBS -r n 
#PBS -j oe 

cd $PBS_O_WORKDIR

source activate ai518

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="/home2/s20235025/Melon_data"
export OUTPUT_DIR="/home2/s20235025/Melon_ai518/output"

accelerate launch --mixed_precision="fp16" /home2/s20235025/diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --num_train_epochs=50 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --report_to="all" \
  --output_dir=$OUTPUT_DIR