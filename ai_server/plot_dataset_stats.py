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
    plot_transcript_sources(rows, output_dir / "transcript_sources.png")
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


def plot_transcript_sources(rows: list[dict], output: Path) -> None:
    counts = Counter(row.get("input", {}).get("transcript_source", "unknown") for row in rows)
    labels = list(counts.keys())
    values = list(counts.values())
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.title("Transcript source")
    plt.xlabel("Source")
    plt.ylabel("Interactions")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def write_summary(rows: list[dict], output: Path) -> None:
    backend_counts = Counter(row.get("backend", "unknown") for row in rows)
    model_counts = Counter(row.get("model", "unknown") for row in rows)
    stt_counts = Counter(row.get("stt_backend", "unknown") for row in rows)
    transcript_source_counts = Counter(row.get("input", {}).get("transcript_source", "unknown") for row in rows)
    session_counts = Counter(row.get("session_id", "unknown") for row in rows)
    timestamps = [parse_timestamp(row.get("timestamp", "")) for row in rows]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    latencies = [row.get("latency_ms", 0) for row in rows]
    durations = [row.get("input", {}).get("audio_duration_ms", 0) for row in rows]
    summary = {
        "rows": len(rows),
        "ok_rows": sum(1 for row in rows if row.get("ok", False)),
        "backend_counts": dict(backend_counts),
        "model_counts": dict(model_counts),
        "stt_backend_counts": dict(stt_counts),
        "transcript_source_counts": dict(transcript_source_counts),
        "session_counts": dict(session_counts),
        "latency_ms_avg": mean(latencies),
        "latency_ms_p95": percentile(latencies, 95),
        "audio_duration_ms_avg": mean(durations),
        "first_timestamp": min(timestamps).isoformat() if timestamps else None,
        "last_timestamp": max(timestamps).isoformat() if timestamps else None,
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def percentile(values: list[int], percent: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percent / 100) * (len(ordered) - 1)))
    return ordered[index]


if __name__ == "__main__":
    main()
