"""
使用增强的数据集微调GPT-2模型，以创建更好的莎士比亚聊天机器人
"""

from datasets import load_from_disk, load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
import os


def train_model(use_enhanced_dataset=True):
    # 1. 加载 GPT-2 分词器和模型
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 没有 pad token，使用 eos 替代
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # 2. 加载数据集
    if use_enhanced_dataset and os.path.exists("enhanced_shakespeare_dataset"):
        print("加载增强的莎士比亚数据集...")
        dataset = load_from_disk("enhanced_shakespeare_dataset")
        # 创建训练和验证集的分割
        dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
    else:
        print("加载原始 tiny_shakespeare 数据集...")
        # 如果没有增强数据集，使用原始数据集
        dataset = load_dataset("tiny_shakespeare")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]

    # 3. 分词函数
    def tokenize_function(examples):
        # 增加最大长度以容纳更复杂的对话
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,  # 增加最大长度
        )

    # 4. 处理训练和验证数据集
    print("处理训练数据集...")
    tokenized_train = train_dataset.map(
        tokenize_function, batched=True, remove_columns=train_dataset.column_names
    )

    print("处理验证数据集...")
    tokenized_val = val_dataset.map(
        tokenize_function, batched=True, remove_columns=val_dataset.column_names
    )

    # 5. 设置 DataCollator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. 设置优化的训练参数
    training_args = TrainingArguments(
        output_dir="./gpt2-shakespeare",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=4,  # 根据可用GPU内存增加
        per_device_eval_batch_size=4,
        num_train_epochs=5,  # 增加训练轮次
        weight_decay=0.01,
        learning_rate=5e-5,  # 降低学习率以更好地微调
        logging_dir="./logs",
        save_total_limit=2,
        fp16=False,  # 根据硬件调整
        push_to_hub=False,
        # 添加warmup步骤以稳定训练
        warmup_steps=500,
        # 添加梯度累积以处理更大的批量
        gradient_accumulation_steps=2,
    )

    # 7. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 8. 开始训练
    print("开始训练模型...")
    trainer.train()

    # 9. 保存模型
    model_save_path = "./gpt2-shakespeare-final"
    print(f"训练完成，保存模型到 {model_save_path}")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    return model_save_path


if __name__ == "__main__":
    # 首先确保我们有增强的数据集
    if not os.path.exists("enhanced_shakespeare_dataset"):
        print("未找到增强的数据集，先创建增强数据集...")
        from create_character_dataset import enhance_dataset

        enhance_dataset()

    # 然后训练模型
    model_path = train_model(use_enhanced_dataset=True)
    print(f"模型训练完成并保存在: {model_path}")
