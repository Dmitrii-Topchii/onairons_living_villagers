import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Create quick course-report plots from Living Villagers logs.")
    parser.add_argument("--input", default="data/interactions_raw.jsonl")
    parser.add_argument("--output-dir", default="data/plots")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    plot_latency(rows, output_dir / "latency_ms.png")
    plot_audio_duration(rows, output_dir / "audio_duration_ms.png")
    plot_context_size(rows, output_dir / "context_facts.png")
    write_summary(rows, output_dir / "summary.json")
    print(f"Wrote plots and summary to {output_dir}")


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def plot_latency(rows: list[dict], output: Path) -> None:
    values = [row.get("latency_ms", 0) for row in rows]
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=min(20, max(5, len(values))))
    plt.title("AI response latency")
    plt.xlabel("Latency, ms")
    plt.ylabel("Interactions")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_audio_duration(rows: list[dict], output: Path) -> None:
    values = [row.get("input", {}).get("audio_duration_ms", 0) for row in rows]
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=min(20, max(5, len(values))))
    plt.title("Player speech segment duration")
    plt.xlabel("Duration, ms")
    plt.ylabel("Segments")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_context_size(rows: list[dict], output: Path) -> None:
    values = [len(row.get("input", {}).get("scene", [])) for row in rows]
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=range(0, max(values) + 2))
    plt.title("Scene facts sent to model")
    plt.xlabel("Facts per interaction")
    plt.ylabel("Interactions")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def write_summary(rows: list[dict], output: Path) -> None:
    backend_counts = Counter(row.get("backend", "unknown") for row in rows)
    stt_counts = Counter(row.get("stt_backend", "unknown") for row in rows)
    timestamps = [parse_timestamp(row.get("timestamp", "")) for row in rows]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    summary = {
        "rows": len(rows),
        "ok_rows": sum(1 for row in rows if row.get("ok", False)),
        "backend_counts": dict(backend_counts),
        "stt_backend_counts": dict(stt_counts),
        "first_timestamp": min(timestamps).isoformat() if timestamps else None,
        "last_timestamp": max(timestamps).isoformat() if timestamps else None,
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
