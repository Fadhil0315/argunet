import torch
import numpy as np
from transformers import AutoTokenizer
from model.debateModel import DebateModel
import random
import os


import pandas as pd
df = pd.read_csv("data/processed/merged.csv")
fallacy_labels = sorted(df[df.task=="fallacy"].label.unique())

# ========================
# FIX SEED
# ========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

set_seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load model
model = DebateModel(
    num_claim=3,
    num_fallacy=len(fallacy_labels),
    num_toxic=2
)


model.load_state_dict(
    torch.load("checkpoints/debate_model.pt", map_location=device)
)

model.to(device)
model.eval()

# Label maps (must match train.py)
claim_labels = ["factual", "opinion", "misleading"]
toxic_labels = ["clean", "toxic"]

# Fallacy labels — regenerate same order



def predict(text):

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids, mask)

    claim_pred = torch.argmax(out["claim"], dim=1).item()
    fallacy_pred = torch.argmax(out["fallacy"], dim=1).item()
    toxic_pred = torch.argmax(out["toxic"], dim=1).item()

    return {
        "claim": claim_labels[claim_pred],
        "fallacy": fallacy_labels[fallacy_pred],
        "toxic": toxic_labels[toxic_pred]
    }


# ========================
# TEST LOOP
# ========================
while True:
    text = input("\nEnter text (or 'q' to quit): ")

    if text.lower() == "q":
        break

    result = predict(text)

    print("\nPrediction:")
    for k, v in result.items():
        print(f"{k}: {v}")
