import argparse
import json
from pathlib import Path

from server import build_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw gameplay logs to SFT chat JSONL.")
    parser.add_argument("--input", default="data/interactions_raw.jsonl")
    parser.add_argument("--output", default="data/villager_sft_train.jsonl")
    parser.add_argument("--only-ok", action="store_true", help="Skip backend failures.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue

            row = json.loads(line)
            if args.only_ok and not row.get("ok", False):
                continue

            payload = row.get("input", {})
            memory = payload.get("recent_memory", [])
            output = row.get("output", {})
            assistant_json = {
                "line": output.get("line", ""),
                "emotion": output.get("emotion", "annoyed"),
                "action": output.get("action", "stare_at_player"),
                "memory_update": output.get("memory_update", ""),
            }
            messages = build_prompt(payload, memory)
            messages.append({
                "role": "assistant",
                "content": json.dumps(assistant_json, ensure_ascii=False),
            })
            target.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} examples to {output_path}")


if __name__ == "__main__":
    main()
