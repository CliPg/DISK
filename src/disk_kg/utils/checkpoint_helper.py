import os
import json
import pickle

# Set checkpoint directory to project root/checkpoints
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.json")

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return {"file_idx": 0, "entity_block_idx": 0, "relation_block_idx": 0}
    with open(CHECKPOINT_FILE, "r") as f:
        return json.load(f)

def save_checkpoint(pdf_idx, entity_block_idx, relation_block_idx):
    data = {
        "file_idx": pdf_idx,
        "entity_block_idx": entity_block_idx,
        "relation_block_idx": relation_block_idx
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)
