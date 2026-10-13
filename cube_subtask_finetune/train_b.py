"""
多帧序列 + IMU 微调训练。

用法:
    python train_b.py                          # 默认用 train_seq.jsonl
    python train_b.py --data data/train_b.jsonl  # 指定数据文件
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen3VLProcessor,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ============ 配置 ============
MODEL_NAME = "/home/liuyao/Telephix/Telephix-annotation/cube_subtask_finetune/model/Qwen3-VL-2B-Instruct"
DATA_FILE = "/home/liuyao/projects/cube_subtask_finetune/data/train_seq.jsonl"
OUTPUT_DIR = "/home/liuyao/projects/cube_subtask_finetune/checkpoints_seq"

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

BATCH_SIZE = 1       # 多图输入显存更大，batch_size=1
GRAD_ACCUM = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
MAX_PIXELS = 120 * 28 * 28
# =============================


class SubtaskDatasetB(Dataset):
    """加载 (IMU图 + 摄像头图) 双图输入。"""

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
        conv = sample["conversations"]

        # 加载图片列表
        image_paths = sample.get("images", [sample["image"]])
        images = [Image.open(p).convert("RGB") for p in image_paths]

        # 构建多图 message
        content = []
        for _ in images:
            content.append({"type": "image"})
        content.append({
            "type": "text",
            "text": conv[0]["value"].replace("<image>\n", "").replace("<image>", ""),
        })

        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": conv[1]["value"]},
        ]

        # 编码
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # prompt 部分不计算 loss
        prompt_messages = [messages[0]]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=images,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "labels": labels,
        }

        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]
        if "mm_token_type_ids" in inputs:
            result["mm_token_type_ids"] = inputs["mm_token_type_ids"][0]

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

    pixel_values_list = []
    image_grid_thw_list = []
    mm_token_type_ids_list = []

    for item in batch:
        if "pixel_values" in item:
            pixel_values_list.append(item["pixel_values"])
        if "image_grid_thw" in item:
            image_grid_thw_list.append(item["image_grid_thw"])
        if "mm_token_type_ids" in item:
            mm_token_type_ids_list.append(item["mm_token_type_ids"])

    if pixel_values_list:
        result["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_grid_thw_list:
            result["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
        if mm_token_type_ids_list:
            result["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                mm_token_type_ids_list, batch_first=True, padding_value=0,
            )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_FILE, help="训练数据 jsonl 路径")
    parser.add_argument("--output", default=OUTPUT_DIR, help="checkpoint 输出目录")
    args = parser.parse_args()

    data_file = args.data
    output_dir = args.output

    print(f"加载模型: {MODEL_NAME}")

    processor = Qwen3VLProcessor.from_pretrained(MODEL_NAME, max_pixels=MAX_PIXELS)
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

    print(f"加载数据: {data_file}")
    dataset = SubtaskDatasetB(data_file, processor)
    print(f"数据集大小: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
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

    lora_output = os.path.join(output_dir, "lora_final")
    model.save_pretrained(lora_output)
    processor.save_pretrained(lora_output)
    print(f"LoRA 权重保存到: {lora_output}")


if __name__ == "__main__":
    main()