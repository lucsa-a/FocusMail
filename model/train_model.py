import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import DatasetDict, Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

df = pd.read_csv("./dataset/dataset.csv")

df_train, df_test = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df["Categoria"]
)

label2id = {"Produtivo": 0, "Improdutivo": 1}
id2label = {0: "Produtivo", 1: "Improdutivo"}

df_train["labels"] = df_train["Categoria"].map(label2id)
df_test["labels"] = df_test["Categoria"].map(label2id)

train_dataset = Dataset.from_pandas(df_train[["Texto", "labels"]])
test_dataset = Dataset.from_pandas(df_test[["Texto", "labels"]])
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["Texto"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

save_path = Path("./model/model")
save_path.mkdir(parents=True, exist_ok=True)

trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print(f"Modelo e tokenizer salvos em: {save_path.resolve()}")
