"""Prompt construction and SQL cleaning for the served text-to-SQL model.

Everything here must stay byte-for-byte consistent with the Phase 0-3 code,
because the accuracy numbers (e.g. 75.5% for Q4_K_M) were measured with this
exact prompt and this exact extraction logic.
"""

import re

# Copied verbatim from Phase 0 build_prompt. Do not edit.
SYSTEM_MSG = (
    "You are a text-to-SQL model. Given a database schema and a question, "
    "output only the SQL query. Do not include explanations or markdown formatting."
)


def build_messages(schema_text: str, question: str) -> list[dict]:
    """Return the system + user messages (inference mode: no assistant turn)."""
    user_msg = f"Schema:\n{schema_text}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]


def render_chatml(messages: list[dict]) -> str:
    """Render messages in Qwen2.5's ChatML format and open the assistant turn.

    Equivalent to tokenizer.apply_chat_template(messages, tokenize=False,
    add_generation_prompt=True) for system + user messages. Verify with
    scripts/check_template.py.
    """
    parts = [
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    ]
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def build_prompt(schema_text: str, question: str) -> str:
    """Convenience: the full prompt string to send to llama-server /completion."""
    return render_chatml(build_messages(schema_text, question))


def clean_sql(raw_output: str) -> str:
    """Strip markdown fences and explanation text down to a single SQL statement.

    Logic is identical to Phase 0's extract_sql. Do not change it without
    re-running the eval harness.
    """
    text = raw_output.strip()

    # strip markdown code fences if present
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # isolate the first SQL-looking statement
    sql_start_match = re.search(
        r"(SELECT|WITH|INSERT|UPDATE|DELETE)\b", text, re.IGNORECASE
    )
    if sql_start_match:
        text = text[sql_start_match.start():]

    # cut off at the first semicolon if present, else take the whole remainder
    if ";" in text:
        text = text.split(";")[0] + ";"

    return text.strip()