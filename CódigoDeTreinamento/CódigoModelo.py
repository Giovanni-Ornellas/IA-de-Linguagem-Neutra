from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Carrega tokenizer e modelo base em português
model_name = "unicamp-dl/ptt5-base-portuguese-vocab"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Carrega dataset CSV
dataset = load_dataset("csv", data_files="neutra_dataset.csv")

# Tokeniza dados
def tokenize(batch):
    input_enc = tokenizer(batch["input"], truncation=True, padding="max_length", max_length=64)
    target_enc = tokenizer(batch["target"], truncation=True, padding="max_length", max_length=64)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2)

# Define argumentos de treino
training_args = TrainingArguments(
    output_dir="./t5_neutro",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
)

# Inicia o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Treina o modelo
trainer.train()
