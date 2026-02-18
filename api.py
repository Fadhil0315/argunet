from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import torch
import numpy as np
import pandas as pd
import random

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from model.debateModel import DebateModel


# ========================
# APP INIT
# ========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ========================
# PRESET TOPICS (20)
# ========================

TOPICS = [
    "Should artificial intelligence be used in schools?",
    "Is climate change primarily caused by human activity?",
    "Should social media be regulated by governments?",
    "Is online education better than traditional education?",
    "Should college education be free?",
    "Is remote work more productive than office work?",
    "Should governments ban single-use plastics?",
    "Is cryptocurrency the future of finance?",
    "Should voting be mandatory?",
    "Is space exploration worth the cost?",
    "Should animals be used for scientific research?",
    "Is nuclear energy a safe alternative?",
    "Should universal basic income be implemented?",
    "Is censorship ever justified?",
    "Should standardized testing be abolished?",
    "Is capitalism the best economic system?",
    "Should smartphones be banned in schools?",
    "Is globalisation beneficial?",
    "Should healthcare be universal?",
    "Is renewable energy sufficient for global needs?"
]


# ========================
# LOAD LABELS
# ========================

df = pd.read_csv("data/processed/merged.csv")
fallacy_labels = sorted(df[df.task == "fallacy"].label.unique())
claim_labels = ["factual", "opinion", "misleading"]
toxic_labels = ["clean", "toxic"]


# ========================
# LOAD MODELS
# ========================

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

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

# Semantic model for relevance
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ========================
# ROOM STORAGE
# ========================

rooms = {}


# ========================
# REQUEST MODELS
# ========================

class JoinRequest(BaseModel):
    room: str
    name: str

class MessageRequest(BaseModel):
    room: str
    name: str
    text: str


# ========================
# HELPER FUNCTIONS
# ========================
def calibrate(score, min_val, power=0.7):
    """
    Softens extreme probabilities.
    - min_val: minimum allowed output
    - power: <1 boosts low values
    """
    score = score ** power
    return max(min_val, score)

def confidence_from_probs(probs):
    eps = 1e-9
    entropy = -np.sum(probs * np.log(probs + eps))
    max_entropy = np.log(len(probs))

    return 1 - (entropy / max_entropy)



def analyze_text(text):

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

    temp=1.8

    claim = torch.softmax(out["claim"]/temp, dim=1)[0].cpu().numpy()
    fallacy = torch.softmax(out["fallacy"]/temp, dim=1)[0].cpu().numpy()
    toxic = torch.softmax(out["toxic"]/temp, dim=1)[0].cpu().numpy()

    return claim, fallacy, toxic


def compute_relevance(topic, text):

    topic_emb = embedder.encode(topic, convert_to_tensor=True)
    text_emb = embedder.encode(text, convert_to_tensor=True)

    score = util.cos_sim(topic_emb, text_emb).item()

    # normalize from [-1,1] to [0,1]
    score = (score + 1) / 2

    return float(score)


def detect_stance(text):

    positive_words = ["support", "benefit", "important", "necessary", "improves", "helps"]
    negative_words = ["harmful", "bad", "dangerous", "worse", "problem", "risk"]

    text = text.lower()

    pos = sum(word in text for word in positive_words)
    neg = sum(word in text for word in negative_words)

    if pos > neg:
        return "for"
    elif neg > pos:
        return "against"
    else:
        return "neutral"


# ========================
# JOIN ROOM
# ========================

@app.post("/join")
def join_room(req: JoinRequest):

    if req.room not in rooms:

        topic = random.choice(TOPICS)

        rooms[req.room] = {
            "players": [],
            "roles": {},
            "topic": topic,
            "messages": [],
            "scores": {},
            "turn": 0
        }

    room = rooms[req.room]

    if req.name not in room["players"]:

        if len(room["players"]) >= 2:
            return {"error": "Room full"}

        room["players"].append(req.name)
        room["scores"][req.name] = 0

        if len(room["players"]) == 1:
            room["roles"][req.name] = "for"
        else:
            room["roles"][req.name] = "against"

    return {
        "players": room["players"],
        "roles": room["roles"],
        "topic": room["topic"],
        "turn": room["turn"]
    }


# ========================
# SEND MESSAGE
# ========================

@app.post("/send")
def send_message(req: MessageRequest):

    claim_conf = 1.0
    fallacy_conf = 1.0
    toxic_conf = 1.0


    if req.room not in rooms:
        return {"error": "Room not found"}

    room = rooms[req.room]

    if len(room["players"]) < 2:
        return {"error": "Waiting for second player"}

    if room["players"][room["turn"]] != req.name:
        return {"error": "Not your turn"}

    claim, fallacy, toxic = analyze_text(req.text)

    claim_conf = confidence_from_probs(claim)
    fallacy_conf = confidence_from_probs(fallacy)
    toxic_conf = confidence_from_probs(toxic)

    raw_claim = float(np.max(claim)) * claim_conf
    raw_fallacy = float(np.max(fallacy)) * fallacy_conf
    raw_toxic = float(np.max(toxic)) * toxic_conf

    



# Convert to "good" scores
    raw_logic = 1 - raw_fallacy
    raw_civility = 1 - raw_toxic


# Calibrate (fix over-harsh model)
    claim_score = calibrate(raw_claim, 0.5, power=0.6)
    logic_score = calibrate(raw_logic, 0.4, power=0.7)
    civility_score = calibrate(raw_civility, 0.5, power=0.6)


    relevance_score = compute_relevance(room["topic"], req.text)

    stance = detect_stance(req.text)
    role = room["roles"][req.name]

    stance_bonus = 1.0
    if stance == role:
        stance_bonus = 1.2
    elif stance != "neutral":
        stance_bonus = 0.6

    total_score = (
    0.35 * claim_score +
    0.30 * logic_score +
    0.20 * civility_score +
    0.15 * relevance_score
    ) * stance_bonus


    # ========================
# Convert to Game Score (Integer)
# ========================

# Clamp just in case
    total_score = max(0.0, min(total_score, 1.0))

# Scale to 1–10 range
    game_score = int(round(total_score * 8 + 1))


    room["scores"][req.name] += game_score

    msg = {
        "name": req.name,
        "text": req.text,
        "stats": {
            "claim": round(claim_score, 2),
            "logic": round(logic_score, 2),
            "civility": round(civility_score, 2),
            "relevance": round(relevance_score, 2)
        },
        "score": game_score
    }

    room["messages"].append(msg)

    room["turn"] = 1 - room["turn"]

    return {
        "message": msg,
        "scores": room["scores"],
        "next_turn": room["players"][room["turn"]]
    }


# ========================
# GET STATE
# ========================

@app.get("/state/{room_id}")
def get_state(room_id: str):

    if room_id not in rooms:
        return {"error": "Room not found"}

    return rooms[room_id]