"""
视频输入微调训练：输入完整 episode 视频，输出 subtask 时间线。

用法:
    python train_video.py
    python train_video.py --data data/train_video.jsonl --output checkpoints_video
"""

import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    Qwen3VLProcessor,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DATA_FILE = "data/train_video.jsonl"
OUTPUT_DIR = "checkpoints_video"

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

BATCH_SIZE = 1
GRAD_ACCUM = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
MAX_FRAMES = 160            # 160帧覆盖全~90s视频, 每0.56s一帧
FPS = 1.78                  # 采样率: 160帧/90s ≈ 1.78fps
# size = min_pixels / max_pixels（总像素预算 = 每帧像素 × 帧数）
# 每帧 ~448x640 = 286720 px × 160帧 ≈ 45.9M
VIDEO_SIZE = {"shortest_edge": 11468800, "longest_edge": 45875200}


class VideoSubtaskDataset(Dataset):
    def __init__(self, data_file, processor):
        self.samples = []
        with open(data_file) as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample["messages"]
        video_path = sample["video"]

        # 分离 user 和 assistant 消息
        user_msg = messages[0]
        assistant_text = messages[1]["content"]

        # 构建完整对话
        full_messages = [
            user_msg,
            {"role": "assistant", "content": assistant_text},
        ]

        text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        # 处理视频输入
        inputs = self.processor(
            text=[text],
            videos=[video_path],
            fps=FPS,
            size=VIDEO_SIZE,
            max_frames=MAX_FRAMES,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # prompt 部分不计算 loss
        prompt_messages = [user_msg]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            videos=[video_path],
            fps=FPS,
            size=VIDEO_SIZE,
            max_frames=MAX_FRAMES,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "labels": labels,
        }

        # 传递视频相关参数
        for key in ("pixel_values_videos", "pixel_values", "image_grid_thw", "mm_token_type_ids",
                     "video_grid_thw", "video_second_per_grid_ts", "second_per_grid_ts"):
            if key in inputs:
                val = inputs[key]
                if key == "mm_token_type_ids":
                    result[key] = val[0]
                elif key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
                    result[key] = val  # 保持原始 shape
                elif key in ("video_second_per_grid_ts", "second_per_grid_ts"):
                    result[key] = val[0] if val.dim() > 0 else val
                else:
                    result[key] = val[0] if val.dim() > 0 else val

        return result


def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=0,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch], batch_first=True, padding_value=0,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [item["labels"] for item in batch], batch_first=True, padding_value=-100,
    )

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # 合并视频相关参数
    for key in ("pixel_values_videos", "pixel_values", "image_grid_thw", "mm_token_type_ids",
                 "video_grid_thw", "video_second_per_grid_ts", "second_per_grid_ts"):
        vals = [item[key] for item in batch if key in item]
        if not vals:
            continue
        if key in ("pixel_values", "pixel_values_videos"):
            result[key] = torch.cat(vals, dim=0)
        elif key in ("image_grid_thw", "video_grid_thw"):
            result[key] = torch.cat(vals, dim=0)  # (total_videos, 3)
        elif key == "mm_token_type_ids":
            result[key] = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=0)
        elif key in ("video_second_per_grid_ts", "second_per_grid_ts"):
            if all(v.dim() == 0 for v in vals):
                result[key] = torch.tensor([v.item() for v in vals])
            else:
                result[key] = torch.cat(vals, dim=0)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_FILE)
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args()

    print(f"加载模型: {MODEL_NAME}")
    processor = Qwen3VLProcessor.from_pretrained(MODEL_NAME, max_pixels=4096*4096)  # 不限图片，视频用 size
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"加载数据: {args.data}")
    dataset = VideoSubtaskDataset(args.data, processor)
    print(f"数据集大小: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    print("开始训练...")
    trainer.train()

    lora_output = os.path.join(args.output, "lora_final")
    model.save_pretrained(lora_output)
    processor.save_pretrained(lora_output)
    print(f"LoRA 权重保存到: {lora_output}")


if __name__ == "__main__":
    main()