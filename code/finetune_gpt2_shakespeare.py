from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

# 1. 加载 GPT-2 分词器和模型
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT2 没有 pad token，使用 eos 替代
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 2. 加载 tiny_shakespeare 数据集
dataset = load_dataset("tiny_shakespeare")


# 3. 分词函数
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=128
    )


# 4. 分别处理 train 和 validation 数据集
train_dataset = dataset["train"].map(
    tokenize_function, batched=True, remove_columns=["text"]
)
val_dataset = dataset["validation"].map(
    tokenize_function, batched=True, remove_columns=["text"]
)

# 5. 设置 DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. 设置训练参数（注意输出目录）
training_args = TrainingArguments(
    output_dir="./gpt2-shakespeare",
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    fp16=False,  # MPS 设备不支持混合精度
    push_to_hub=False,
)

# 7. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


# 8. 开始训练
def train():
    trainer.train()
    trainer.save_model("./gpt2-shakespeare")


if __name__ == "__main__":
    train()
