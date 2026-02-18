import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW

from model.debateModel import DebateModel

# ========================
# FIX SEED
# ========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

set_seed(42)

# ========================
# DEVICE
# ========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ========================
# LOAD DATA
# ========================
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/merged.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.15,
    random_state=68,
    stratify=df["task"]
)


# Label mappings
claim_map = {"factual":0, "opinion":1, "misleading":2}
toxic_map = {"clean":0, "toxic":1}

fallacy_labels = sorted(df[df.task=="fallacy"].label.unique())
fallacy_map = {l:i for i,l in enumerate(fallacy_labels)}

# ========================
# TOKENIZER
# ========================
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

MAX_LEN = 128


class DebateDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        text = str(row.text)
        task = row.task
        label = row.label

        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "task": task
        }

        if task == "claim":
            item["label"] = torch.tensor(claim_map[label])

        elif task == "fallacy":
            item["label"] = torch.tensor(fallacy_map[label])

        else:
            item["label"] = torch.tensor(toxic_map[label])

        return item


# ========================
# DATALOADER
# ========================
train_dataset = DebateDataset(train_df)
val_dataset = DebateDataset(val_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)


# ========================
# MODEL
# ========================
model = DebateModel(
    num_claim=3,
    num_fallacy=len(fallacy_map),
    num_toxic=2
).to(device)

# Freeze bottom layers (speed)
for param in model.encoder.base_model.embeddings.parameters():
    param.requires_grad = False

# ========================
# OPTIMIZER
# ========================
optimizer = AdamW(model.parameters(), lr=2e-5)

criterion = torch.nn.CrossEntropyLoss()

# ========================
# TRAIN LOOP
# ========================
EPOCHS = 10
PATIENCE = 2


model.train()

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):

    # ================= TRAIN =================
    model.train()
    train_loss = 0

    for batch in train_loader:

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        task = batch["task"]

        out = model(input_ids, mask)

        if task[0] == "claim":
            loss = criterion(out["claim"], labels)

        elif task[0] == "fallacy":
            loss = criterion(out["fallacy"], labels)

        else:
            loss = criterion(out["toxic"], labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()


    train_loss /= len(train_loader)

    # ================= VALIDATION =================
    model.eval()
    val_loss = 0

    with torch.no_grad():

        for batch in val_loader:

            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            task = batch["task"]

            out = model(input_ids, mask)

            if task[0] == "claim":
                loss = criterion(out["claim"], labels)

            elif task[0] == "fallacy":
                loss = criterion(out["fallacy"], labels)

            else:
                loss = criterion(out["toxic"], labels)

            val_loss += loss.item()


    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # ================= EARLY STOPPING =================
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            model.state_dict(),
            "checkpoints/debate_model.pt"
        )

        print("✅ Model improved. Saved.")

    else:

        patience_counter += 1
        print(f"⚠️ No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("🛑 Early stopping triggered.")
            break


# ========================
# SAVE
# ========================
os.makedirs("checkpoints", exist_ok=True)

torch.save(model.state_dict(), "checkpoints/debate_model.pt")

print("Model saved → checkpoints/debate_model.pt")
