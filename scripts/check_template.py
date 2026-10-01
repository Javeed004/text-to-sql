# scripts/check_template.py
import json
from transformers import AutoTokenizer
import sys
from pathlib import Path

# Add the project root to Python's module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.prompt import build_messages, render_chatml

tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-3B-Instruct")
rows = [json.loads(l) for l in open("data/test.jsonl")][:20]

bad = 0
for r in rows:
    msgs = build_messages(r["schema_text"], r["question"])
    want = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if render_chatml(msgs) != want:
        bad += 1
print("mismatches:", bad)