import torch
import torch.nn as nn
from transformers import AutoModel
import random
import numpy as np
import os

# ========================
# FIX SEED
# ========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

set_seed(42)


class DebateModel(nn.Module):

    def __init__(self,
                 base_model="distilbert-base-uncased",
                 num_claim=3,
                 num_fallacy=10,
                 num_toxic=2):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(base_model)

        hidden = self.encoder.config.hidden_size

        # Heads
        self.claim_head = nn.Linear(hidden, num_claim)
        self.fallacy_head = nn.Linear(hidden, num_fallacy)
        self.toxic_head = nn.Linear(hidden, num_toxic)

        self.dropout = nn.Dropout(0.2)


    def forward(self, input_ids, attention_mask):

        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = out.last_hidden_state[:, 0, :]

        cls = self.dropout(cls)

        return {
            "claim": self.claim_head(cls),
            "fallacy": self.fallacy_head(cls),
            "toxic": self.toxic_head(cls)
        }
