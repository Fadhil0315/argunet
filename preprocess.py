import os
import random
import numpy as np
import pandas as pd
from datasets import load_dataset

# ========================
# FIX RANDOM SEED
# ========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# ========================
# OUTPUT FOLDER
# ========================
os.makedirs("data/processed", exist_ok=True)

all_data = []

# ==================================================
# 1️⃣ CLAIM DATA (LIAR)
# ==================================================
print("Loading LIAR dataset...")

liar_path = "data/raw/claim/liar_dataset/train.tsv"
liar_df = pd.read_csv(liar_path, sep="\t", header=None)


# Columns:
# 1 = label, 2 = statement
liar_df = liar_df[[1, 2]]
liar_df.columns = ["label", "text"]

def map_claim(label):
    if label in ["true", "mostly-true"]:
        return "factual"
    elif label == "half-true":
        return "opinion"
    else:
        return "misleading"

liar_df["label"] = liar_df["label"].apply(map_claim)
liar_df["task"] = "claim"

liar_df = liar_df.sample(n=4000, random_state=42)

all_data.append(liar_df[["text", "task", "label"]])

print("Claim samples:", len(liar_df))


# ==================================================
# 2️⃣ FALLACY DATA (HF)
# ==================================================
print("Loading Fallacy dataset...")

fallacy_ds = load_dataset("tasksource/logical-fallacy")
fallacy_train = fallacy_ds["train"]

fallacy_df = pd.DataFrame({
    "text": fallacy_train["source_article"],
    "label": fallacy_train["logical_fallacies"]
})

fallacy_df["task"] = "fallacy"

fallacy_df = fallacy_df.sample(n=2500, random_state=42)

all_data.append(fallacy_df[["text", "task", "label"]])

print("Fallacy samples:", len(fallacy_df))


# ==================================================
# 3️⃣ TOXIC DATA (Civil Comments)
# ==================================================
print("Loading Toxic dataset...")

toxic_ds = load_dataset("civil_comments")

toxic_train = toxic_ds["train"].shuffle(seed=42).select(range(6000))

toxic_df = pd.DataFrame({
    "text": toxic_train["text"],
    "toxicity": toxic_train["toxicity"]
})

def map_toxic(score):
    return "toxic" if score >= 0.5 else "clean"

toxic_df["label"] = toxic_df["toxicity"].apply(map_toxic)
toxic_df["task"] = "toxic"

all_data.append(toxic_df[["text", "task", "label"]])

print("Toxic samples:", len(toxic_df))


# ==================================================
# MERGE
# ==================================================
merged = pd.concat(all_data, ignore_index=True)

merged = merged.dropna()
merged = merged.sample(frac=1, random_state=42)

print("Total samples:", len(merged))

merged.to_csv("data/processed/merged.csv", index=False)

print("Saved → data/processed/merged.csv")
