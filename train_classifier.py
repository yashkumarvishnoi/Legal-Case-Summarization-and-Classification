
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import os
from torch.nn import CrossEntropyLoss

# 1. Data Loading
df = pd.read_csv('balanced_legal_cases_dataset.csv')
print("Data loaded successfully. Shape:", df.shape)

# 2. Category Processing
categories = ['administrative', 'criminal', 'civil', 'constitutional', 'family', 'commercial']
df['category'] = df[categories].idxmax(axis=1)
df['label'] = df['category'].map({cat: idx for idx, cat in enumerate(categories)})

# Check class distribution 
print("\nClass Distribution:")
print(df['category'].value_counts(normalize=True))

# 3. Data Cleaning
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    return ' '.join(text.split()).strip()

df['case_text'] = df['case_text'].apply(clean_text)
df = df[df['case_text'].str.len() > 50]

# 4. Train-Validation Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['case_text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# 5. Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained(
    'distilbert-base-uncased',
    model_max_length=512
)

train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=512)

# 6. Dataset Class
class LegalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = LegalDataset(train_encodings, train_labels)
val_dataset = LegalDataset(val_encodings, val_labels)

# 7. Calculate class weights for imbalance handling
class_counts = np.bincount(df['label'])
class_weights = 1. / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum()
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print("\nClass weights:", class_weights)

# 8. Model Initialization
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(categories),
    id2label={i: cat for i, cat in enumerate(categories)},
    label2id={cat: i for i, cat in enumerate(categories)}
)

# 9. Metrics Calculation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
    }
    
    for i, cat in enumerate(categories):
        metrics[f'f1_{cat}'] = f1_score(labels == i, preds == i)
    
    return metrics

# 10. Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,  # Reduced to prevent memory issues
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=42,
    fp16=torch.cuda.is_available(),
    report_to="none",
    save_total_limit=2,
)

# 11. Custom Trainer with Fixed Weight Handling
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(model.device)
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 12. Training
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    class_weights=class_weights
)

print("\nStarting training...")
trainer.train()

# 13. Save Model
output_dir = "./final_model"
os.makedirs(output_dir, exist_ok=True)

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Save config with label mappings
import json
with open(f"{output_dir}/config.json", "r+") as f:
    config = json.load(f)
    config["id2label"] = {i: cat for i, cat in enumerate(categories)}
    config["label2id"] = {cat: i for i, cat in enumerate(categories)}
    f.seek(0)
    json.dump(config, f, indent=2)
    f.truncate()

print("\nTraining complete! Model saved to:", output_dir)