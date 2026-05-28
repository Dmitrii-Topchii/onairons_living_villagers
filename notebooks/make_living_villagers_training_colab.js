const fs = require("fs");
const path = require("path");

const notebookPath = path.join(__dirname, "living_villagers_training_colab.ipynb");

function md(text) {
  return {
    cell_type: "markdown",
    metadata: {},
    source: lines(text),
  };
}

function code(text) {
  return {
    cell_type: "code",
    execution_count: null,
    metadata: {},
    outputs: [],
    source: lines(text),
  };
}

function lines(text) {
  return text.replace(/^\n/, "").split("\n").map((line) => `${line}\n`);
}

const cells = [
  md(`
# Living Villagers: SFT/LoRA Training Notebook

This notebook turns Minecraft villager voice-chat interactions into a trainable dialogue dataset, fine-tunes one or more open-weight chat models, plots useful course-report metrics, and exports artifacts for Hugging Face / local inference.

Project story:

1. **Minecraft mod** captures player speech and nearby villager context.
2. **AI server** transcribes speech, queries a teacher model, and logs interactions.
3. **Dataset curation** converts noisy gameplay logs into target villager responses.
4. **LoRA/QLoRA fine-tuning** teaches a compact model the desired annoyed/unhinged villager style.
5. **Analysis plots** show dataset quality, latency, length distributions, loss curves, and SEBP motivation.

Recommended Colab runtime: **A100** if available. T4 can handle the smaller 1.5B/3B experiments.
`),

  md(`
## 0. Runtime Notes

Before running:

- In Colab, choose **Runtime -> Change runtime type -> GPU**.
- If using gated models such as Gemma, accept their terms on Hugging Face first.
- Store your Hugging Face token in Colab secrets or run \`huggingface_hub.login()\`. Do not paste tokens into the notebook before sharing.

This notebook is designed to be honest for a deep learning course:

- It trains LoRA adapters and logs training loss.
- It supports multiple model configs.
- It creates dataset and training plots.
- It includes a **SEBP opportunity analysis** based on padded short-context examples. The actual SEBP custom backward kernel is not implemented here unless you plug in your own implementation.
`),

  code(`
#@title Install dependencies
# This cell is intentionally explicit. If Unsloth changes its Colab install command,
# replace the first line with the current command from https://docs.unsloth.ai/.
!pip -q install -U unsloth
!pip -q install -U "transformers>=4.51.0" datasets trl peft accelerate bitsandbytes safetensors huggingface_hub
!pip -q install -U pandas numpy matplotlib seaborn scikit-learn tqdm
`),

  code(`
#@title Imports and environment check
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Python ready")
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    !nvidia-smi
`),

  md(`
## 0.1 Hugging Face Login

Run this if you need to download gated models, upload datasets, or push LoRA adapters.

Use a **read token** for downloading models/datasets. Use a **write token** only when pushing your own dataset/checkpoint. The token is typed into a hidden prompt and is not saved into the notebook.
`),

  code(`
#@title Optional Hugging Face login
USE_HF_LOGIN = True  #@param {type:"boolean"}

if USE_HF_LOGIN:
    from getpass import getpass
    from huggingface_hub import login, whoami

    token = getpass("Paste Hugging Face token (input hidden): ")
    if token.strip():
        login(token=token.strip(), add_to_git_credential=False)
        try:
            info = whoami()
            print("Logged in as:", info.get("name", "unknown"))
        except Exception as exc:
            print("Login stored, but whoami check failed:", repr(exc))
    else:
        print("No token entered; continuing without Hugging Face login.")
else:
    print("Skipping Hugging Face login.")
`),

  md(`
## 1. Project Paths

Use one of these workflows:

- **Inside cloned repo:** set \`PROJECT_ROOT\` to the repo directory.
- **Uploaded files:** upload \`interactions_raw.jsonl\` or \`villager_sft_mixed_*.jsonl\` manually and update the paths.
- **Hugging Face dataset:** load from a dataset repo after publishing.
`),

  code(`
#@title Configure paths
PROJECT_ROOT = Path("/content/onairons_living_villagers")

# If running this notebook after cloning the repo:
if not PROJECT_ROOT.exists():
    print("PROJECT_ROOT does not exist yet.")
    print("Either clone the repo in the next cell or update PROJECT_ROOT.")

AI_SERVER_DIR = PROJECT_ROOT / "ai_server"
DATA_DIR = AI_SERVER_DIR / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "colab_training"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODELS_DIR = OUTPUTS_DIR / "models"

for directory in [OUTPUTS_DIR, PLOTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("OUTPUTS_DIR:", OUTPUTS_DIR)
`),

  code(`
#@title Optional: clone repo
# Run this only if the repo is not already present in Colab.
REPO_URL = "https://github.com/Dmitrii-Topchii/onairons_living_villagers.git"

if not PROJECT_ROOT.exists():
    !git clone {REPO_URL} {PROJECT_ROOT}
else:
    print("Repo already exists:", PROJECT_ROOT)
`),

  md(`
## 2. Data Loading

The preferred training file is the mixed curated/synthetic JSONL:

\`\`\`text
ai_server/data/training/villager_sft_mixed_<SESSION_ID>.jsonl
\`\`\`

Because \`ai_server/data/\` is usually gitignored, Colab may not have the data after cloning. If so, upload the JSONL file with the file browser or publish it as a Hugging Face dataset and load it from there.
`),

  code(`
#@title Find or set dataset paths
SESSION_ID = "20260528T115926Z"  #@param {type:"string"}

RAW_SESSION_PATH = DATA_DIR / "sessions" / SESSION_ID / "interactions_raw.jsonl"
CURATED_PATH = DATA_DIR / "sessions" / SESSION_ID / "villager_sft_curated.jsonl"
SYNTHETIC_PATH = DATA_DIR / "synthetic" / "villager_sft_synthetic_100.jsonl"
MIXED_DATASET_PATH = DATA_DIR / "training" / f"villager_sft_mixed_{SESSION_ID}.jsonl"

print("RAW_SESSION_PATH:", RAW_SESSION_PATH, RAW_SESSION_PATH.exists())
print("CURATED_PATH:", CURATED_PATH, CURATED_PATH.exists())
print("SYNTHETIC_PATH:", SYNTHETIC_PATH, SYNTHETIC_PATH.exists())
print("MIXED_DATASET_PATH:", MIXED_DATASET_PATH, MIXED_DATASET_PATH.exists())

if not MIXED_DATASET_PATH.exists():
    print("\\nIf the mixed dataset is missing, upload it or run the curation cells below.")
`),

  code(`
#@title Optional: upload dataset JSONL into the expected folder
# Use this in Colab after git clone, because ai_server/data/ is gitignored and will not be in GitHub.
UPLOAD_DATASET_IF_MISSING = True  #@param {type:"boolean"}

if UPLOAD_DATASET_IF_MISSING and not MIXED_DATASET_PATH.exists():
    try:
        from google.colab import files
        print("Upload one of these files:")
        print("  villager_sft_mixed_<SESSION_ID>.jsonl  preferred")
        print("  interactions_raw.jsonl                 raw log, if you want to rebuild curation")
        uploaded = files.upload()
        for filename, content in uploaded.items():
            name = Path(filename).name
            if name.startswith("villager_sft_mixed") and name.endswith(".jsonl"):
                MIXED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
                MIXED_DATASET_PATH.write_bytes(content)
                print("Saved mixed dataset to:", MIXED_DATASET_PATH)
            elif name == "interactions_raw.jsonl":
                RAW_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
                RAW_SESSION_PATH.write_bytes(content)
                print("Saved raw session log to:", RAW_SESSION_PATH)
            else:
                fallback = DATA_DIR / name
                fallback.parent.mkdir(parents=True, exist_ok=True)
                fallback.write_bytes(content)
                print("Saved uploaded file to:", fallback)
    except ModuleNotFoundError:
        print("google.colab is unavailable. Manually copy the dataset JSONL to:")
        print(MIXED_DATASET_PATH)
else:
    print("Dataset upload skipped or dataset already exists.")
`),

  md(`
## 3. Curation Helpers

These cells let the notebook rebuild a training dataset from a raw gameplay log. It intentionally replaces weak teacher outputs with cleaner target replies so the fine-tuned model learns the desired behavior, not the teacher model's mistakes.
`),

  code(`
#@title JSONL utilities
def read_jsonl(path):
    path = Path(path)
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")

def system_prompt():
    return (
        "You are the brain of a Minecraft villager NPC. "
        "You are not a helpful assistant and you must not explain the task. "
        "React like a short, funny, annoyed, slightly unhinged Minecraft villager. "
        "Use the scene facts and recent memory. Be specific, not generic. "
        "If the transcript is unavailable, react to the sound and scene without pretending you understood exact words. "
        "Never mention internal Java/debug strings. "
        "The line must be one short sentence. "
        "Profanity is allowed sometimes, but no slurs, hate, real-world politics, or protected-group insults. "
        "Return strict JSON only with keys: line, emotion, action, memory_update. "
        "Do not wrap the JSON in markdown."
    )

def user_payload_from_input(payload):
    return {
        "player_speech_transcript": payload.get("transcript", ""),
        "transcript_source": payload.get("transcript_source", "unknown"),
        "audio_duration_ms": payload.get("audio_duration_ms", 0),
        "player": payload.get("player", {}),
        "villager": payload.get("villager", {}),
        "scene": payload.get("scene", []),
        "recent_memory": payload.get("recent_memory", []),
        "allowed_actions": [
            "stare_at_player",
            "step_back",
            "look_at_nearby_block",
            "mutter",
            "panic",
            "ignore",
        ],
        "response_contract": {
            "line": "one funny in-character sentence",
            "emotion": "annoyed|suspicious|scared|confused|smug|offended",
            "action": "one allowed action",
            "memory_update": "one concise fact to remember",
        },
    }

def sft_example(payload, response):
    return {
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": json.dumps(user_payload_from_input(payload), ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
        ]
    }
`),

  code(`
#@title Curation rules
def clean_text(x):
    return re.sub(r"\\s+", " ", str(x or "")).strip()

def is_usable_real_row(row):
    payload = row.get("input", {})
    transcript = clean_text(payload.get("transcript", ""))
    if not row.get("ok", False):
        return False
    if row.get("error"):
        return False
    if payload.get("transcript_source") != "faster_whisper":
        return False
    if payload.get("audio_duration_ms", 0) < 700:
        return False
    if len(transcript) < 4:
        return False
    if "inaudible" in transcript.lower():
        return False
    return True

def resp(line, emotion="annoyed", action="stare_at_player", memory_update=None):
    return {
        "line": line,
        "emotion": emotion,
        "action": action,
        "memory_update": memory_update or line,
    }

def has_word(text, *words):
    return any(re.search(rf"\\b{re.escape(word)}\\b", text, flags=re.I) for word in words)

def choose(index, options):
    return options[index % len(options)]

def curated_response(payload, index=0):
    text = clean_text(payload.get("transcript", "")).lower()
    scene = " ".join(payload.get("scene", [])).lower()
    distance = float(payload.get("villager", {}).get("distance", 9) or 9)

    if "too close" in text or "move away" in text or distance < 0.9:
        return resp(choose(index, [
            "Back up before I start charging rent for my personal space.",
            "You are close enough to count my pixels, and I hate it.",
            "Move one block back before I file a complaint with the nearest wall.",
        ]), "offended", "step_back", "Player invaded personal space.")
    if "potato" in text or "tater" in text:
        return resp(choose(index, [
            "That potato is not a personality, but somehow it is beating yours.",
            "Put the potato down like it has done something wrong.",
            "I respect the potato more than this conversation.",
        ]), "smug", "stare_at_player", "Player kept talking about potatoes.")
    if "zombie" in text or "zombie" in scene:
        return resp(choose(index, [
            "Yes, I see the zombie, and no, your yelling is not armor.",
            "If the zombie eats me, I am haunting your hotbar first.",
            "Great, a zombie and you, two problems with legs.",
        ]), "scared", "panic", "Player warned about a zombie.")
    if "creeper" in text or "creeper" in scene:
        return resp(choose(index, [
            "If that creeper sneezes, I am blaming your entire face.",
            "Stop narrating the creeper and start moving, genius.",
            "That creeper has better timing than you, and it explodes for a living.",
        ]), "scared", "panic", "A creeper was nearby.")
    if any(word in text for word in ["house", "bed", "sleep"]):
        return resp(choose(index, [
            "This is my house, not your square little crime scene.",
            "Touch my bed and I will remember it with professional bitterness.",
            "You walked in like doors are suggestions. Awful, but impressive.",
        ]), "offended", "stare_at_player", "Player intruded into villager home.")
    if has_word(text, "break", "broke", "destroy", "destroyed", "hit", "punched", "punch"):
        return resp(choose(index, [
            "Wonderful, you broke it. The village idiot position is filled.",
            "I watched you do that, and somehow the block looked disappointed.",
            "Do you solve every problem by punching geometry?",
        ]), "annoyed", "look_at_nearby_block", "Player damaged something nearby.")
    if any(word in text for word in ["trade", "sell", "buy", "emerald"]):
        return resp(choose(index, [
            "My prices went up the moment you opened your mouth.",
            "One emerald? For that attitude, make it three and an apology.",
            "I trade goods, not emotional support for confused backpacks.",
        ]), "smug", "mutter", "Player asked about trading.")
    if "name" in text or "who are you" in text:
        return resp(choose(index, [
            "My name is none of your business, but you may call me annoyed.",
            "I had a name before you walked in and lowered the tone.",
            "Call me whatever you want; I will be ignoring you professionally.",
        ]), "suspicious", "stare_at_player", "Player asked for villager identity.")
    if any(word in text for word in ["remember", "before", "again"]):
        return resp(choose(index, [
            "I remember enough to regret giving you a second sentence.",
            "Yes, I remember, and my brain is already filing a noise complaint.",
            "Unfortunately, your previous nonsense survived in memory.",
        ]), "annoyed", "mutter", "Player asked about recent memory.")

    return resp(choose(index, [
        "That sentence entered my ears and immediately lowered property values.",
        "I understood the words, which is exactly why I am upset.",
        "You speak like a crafting table fell down stairs.",
        "I have heard cave noises with better arguments.",
        "Say that again slower so I can be disappointed with precision.",
    ]), "annoyed", "stare_at_player", "Player spoke nearby and annoyed the villager.")

def curate_raw_rows(raw_rows):
    examples = []
    seen = set()
    for row in raw_rows:
        if not is_usable_real_row(row):
            continue
        payload = dict(row["input"])
        payload["transcript"] = clean_text(payload.get("transcript", ""))
        payload.pop("audio_pcm_s16le_bytes", None)
        key = payload["transcript"].lower()
        if key in seen:
            continue
        seen.add(key)
        examples.append(sft_example(payload, curated_response(payload, len(examples))))
    return examples
`),

  code(`
#@title Synthetic scenario generator
def make_scenario(transcript, scene, line, emotion, action, memory_update):
    return {
        "transcript": transcript,
        "scene": scene,
        "output": resp(line, emotion, action, memory_update),
    }

def scenario_pool():
    return [
        make_scenario("Is this your house?", ["Player is standing inside a small villager house.", "World time is night."], "Yes, and somehow you made it feel rented and cursed.", "offended", "stare_at_player", "Player entered villager house."),
        make_scenario("Can I sleep in your bed?", ["Player is looking at a villager bed."], "Touch that bed and I will invent taxes just for you.", "offended", "step_back", "Player asked to use villager bed."),
        make_scenario("Am I too close?", ["Nearest villager is 0.5 blocks away."], "You are close enough to fog up my forehead, yes.", "offended", "step_back", "Player stood too close."),
        make_scenario("Do you want this potato?", ["Dropped item nearby: potato."], "That potato has more dignity than this offer.", "smug", "look_at_nearby_block", "Player offered potato."),
        make_scenario("There is a zombie behind you.", ["Danger: hostile mobs nearby: 1 zombie.", "World time is night."], "Then stop narrating my death and help, blockhead.", "scared", "panic", "Player warned about zombie."),
        make_scenario("A creeper is coming.", ["Danger: hostile mobs nearby: 1 creeper."], "Move, you decorative hazard, before we become architecture.", "scared", "panic", "Player warned about creeper."),
        make_scenario("I trampled your crops.", ["Nearby block event: farmland trampled.", "Nearest villager profession is farmer."], "My crops had dreams before your feet wrote nonsense on them.", "offended", "panic", "Player trampled crops."),
        make_scenario("Do you sell anything good?", ["Nearest villager profession is armorer."], "Not to someone who opens with an insult and empty pockets.", "annoyed", "stare_at_player", "Player asked about trades."),
        make_scenario("Do you remember the potato?", ["Recent memory: Player offered a potato.", "Dropped item nearby: potato."], "I remember the potato, and tragically, I remember you too.", "annoyed", "mutter", "Player asked about previous potato."),
        make_scenario("What is your job?", ["Nearest villager profession is farmer."], "Farmer, which means I grow food and harvest regrets like you.", "annoyed", "mutter", "Player asked villager job."),
        make_scenario("Can you hear my voice?", ["Player spoke through Simple Voice Chat."], "Yes, and now my ears want a refund.", "annoyed", "mutter", "Player tested voice chat."),
        make_scenario("Say something funny.", ["Player spoke through Simple Voice Chat."], "Your face loaded before your brain; that was pretty funny.", "smug", "stare_at_player", "Player demanded joke."),
        make_scenario("Do you want this emerald?", ["Dropped item nearby: emerald."], "I want the emerald and several blocks between us.", "smug", "look_at_nearby_block", "Player offered emerald."),
        make_scenario("I broke your window.", ["Nearby block event: glass broken.", "Player is inside villager house."], "I hope your hand gets emotionally splinters.", "offended", "look_at_nearby_block", "Player broke window."),
        make_scenario("Can we be friends?", ["Player spoke through Simple Voice Chat."], "Let us start with distant acquaintances and lower expectations.", "suspicious", "step_back", "Player asked for friendship."),
    ]

def synthetic_payload(scenario, i):
    distance = 0.6 if "0.5 blocks" in " ".join(scenario["scene"]) else 0.8 + (i % 10) * 0.2
    return {
        "event": "player_spoke",
        "transcript": scenario["transcript"],
        "transcript_source": "synthetic",
        "audio_duration_ms": 1000 + (i * 137) % 1700,
        "packet_count": 60 + i % 80,
        "opus_byte_count": 6000 + (i * 73) % 6000,
        "whispering": False,
        "audio_sample_rate_hz": 48000,
        "audio_channels": 1,
        "player": {"uuid": f"synthetic-player-{i:03d}", "name": "SyntheticPlayer", "x": -200 + i % 5, "y": 152, "z": 90 + i % 7},
        "villager": {"uuid": f"synthetic-villager-{i % 12:02d}", "profession": "unemployed", "distance": distance},
        "scene": [
            "Player spoke through Simple Voice Chat.",
            "Speech transcript is synthetic for supervised fine-tuning.",
            f"Nearest villager is {distance:.1f} blocks away.",
            *scenario["scene"],
        ],
        "recent_memory": [s.replace("Recent memory: ", "") for s in scenario["scene"] if s.startswith("Recent memory:")],
    }

def build_synthetic_examples(n=100):
    pool = scenario_pool()
    rows = []
    for i in range(n):
        scenario = pool[i % len(pool)]
        payload = synthetic_payload(scenario, i)
        rows.append(sft_example(payload, scenario["output"]))
    return rows
`),

  code(`
#@title Build/load training dataset
FORCE_REBUILD_FROM_RAW = False  #@param {type:"boolean"}
SYNTHETIC_ROWS = 100  #@param {type:"integer"}

if MIXED_DATASET_PATH.exists() and not FORCE_REBUILD_FROM_RAW:
    examples = read_jsonl(MIXED_DATASET_PATH)
    print("Loaded existing mixed dataset:", MIXED_DATASET_PATH)
else:
    raw_rows = read_jsonl(RAW_SESSION_PATH)
    curated = curate_raw_rows(raw_rows)
    synthetic = build_synthetic_examples(SYNTHETIC_ROWS)
    examples = curated + synthetic
    write_jsonl(CURATED_PATH, curated)
    write_jsonl(SYNTHETIC_PATH, synthetic)
    write_jsonl(MIXED_DATASET_PATH, examples)
    print("Rebuilt dataset")
    print("Curated:", len(curated), CURATED_PATH)
    print("Synthetic:", len(synthetic), SYNTHETIC_PATH)

print("Total examples:", len(examples))
examples[0]
`),

  md(`
## 4. Dataset Quality Plots

These are useful for the course report because they show that the dataset is structured, short-context, and suitable for SFT.
`),

  code(`
#@title Convert dataset to dataframe
def parse_user_content(example):
    return json.loads(example["messages"][1]["content"])

def parse_assistant_content(example):
    return json.loads(example["messages"][2]["content"])

records = []
for i, ex in enumerate(examples):
    user = parse_user_content(ex)
    assistant = parse_assistant_content(ex)
    scene = user.get("scene", [])
    records.append({
        "idx": i,
        "source": user.get("transcript_source", "unknown"),
        "transcript": user.get("player_speech_transcript", ""),
        "line": assistant.get("line", ""),
        "emotion": assistant.get("emotion", "unknown"),
        "action": assistant.get("action", "unknown"),
        "audio_duration_ms": user.get("audio_duration_ms", 0),
        "scene_facts": len(scene),
        "transcript_chars": len(user.get("player_speech_transcript", "")),
        "line_chars": len(assistant.get("line", "")),
        "has_memory": len(user.get("recent_memory", [])) > 0,
    })

df = pd.DataFrame(records)
df.head()
`),

  code(`
#@title Plot dataset composition
sns.set_theme(style="whitegrid", font_scale=1.0)

fig, axes = plt.subplots(2, 3, figsize=(18, 9))

sns.countplot(data=df, x="source", ax=axes[0, 0])
axes[0, 0].set_title("Dataset source")
axes[0, 0].tick_params(axis="x", rotation=20)

sns.countplot(data=df, y="emotion", order=df["emotion"].value_counts().index, ax=axes[0, 1])
axes[0, 1].set_title("Target emotions")

sns.countplot(data=df, y="action", order=df["action"].value_counts().index, ax=axes[0, 2])
axes[0, 2].set_title("Target actions")

sns.histplot(data=df, x="audio_duration_ms", hue="source", bins=20, ax=axes[1, 0])
axes[1, 0].set_title("Speech duration")

sns.histplot(data=df, x="scene_facts", hue="source", bins=12, ax=axes[1, 1])
axes[1, 1].set_title("Scene facts per example")

sns.scatterplot(data=df, x="transcript_chars", y="line_chars", hue="source", ax=axes[1, 2])
axes[1, 2].set_title("Input length vs reply length")

plt.tight_layout()
plot_path = PLOTS_DIR / "dataset_composition.png"
plt.savefig(plot_path, dpi=180)
plot_path
`),

  code(`
#@title Raw gameplay latency plots, if raw logs are present
raw_rows = read_jsonl(RAW_SESSION_PATH)
if raw_rows:
    raw_df = pd.DataFrame([
        {
            "latency_ms": row.get("latency_ms", 0),
            "ok": row.get("ok", False),
            "model": row.get("model", "unknown"),
            "transcript_source": row.get("input", {}).get("transcript_source", "unknown"),
            "audio_duration_ms": row.get("input", {}).get("audio_duration_ms", 0),
        }
        for row in raw_rows
    ])
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    sns.histplot(raw_df, x="latency_ms", hue="transcript_source", bins=24, ax=axes[0])
    axes[0].set_title("Teacher response latency")
    sns.scatterplot(raw_df, x="audio_duration_ms", y="latency_ms", hue="transcript_source", ax=axes[1])
    axes[1].set_title("Speech duration vs latency")
    sns.countplot(raw_df, x="transcript_source", ax=axes[2])
    axes[2].set_title("STT transcript sources")
    axes[2].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plot_path = PLOTS_DIR / "raw_session_latency.png"
    plt.savefig(plot_path, dpi=180)
    display(raw_df.describe(include="all"))
    print("Saved:", plot_path)
else:
    print("No raw gameplay log found; skipping raw latency plots.")
`),

  md(`
## 5. SEBP Opportunity Analysis

Your SEBP method accelerates training/backprop when short examples are padded to a common context length. This notebook does not implement your custom backward kernel, but it computes the padding-induced sparsity that motivates SEBP and creates a report figure.
`),

  code(`
#@title Estimate padding-induced sparsity
# We approximate token counts cheaply using character length / 4.
# Later, after tokenizer load, the notebook computes exact token counts.
df["approx_input_tokens"] = (df["transcript_chars"] + df["line_chars"] + df["scene_facts"] * 18 + 120) / 4
df["approx_input_tokens"] = df["approx_input_tokens"].astype(int).clip(lower=16)

candidate_contexts = [512, 1024, 2048, 4096]
padding_stats = []
for ctx in candidate_contexts:
    real_tokens = df["approx_input_tokens"].clip(upper=ctx)
    pad_tokens = (ctx - real_tokens).clip(lower=0)
    padding_stats.append({
        "context_length": ctx,
        "mean_real_tokens": real_tokens.mean(),
        "mean_padding_tokens": pad_tokens.mean(),
        "padding_fraction": pad_tokens.mean() / ctx,
        "sebp_upper_bound_speedup": ctx / max(real_tokens.mean(), 1),
    })

padding_df = pd.DataFrame(padding_stats)
display(padding_df)

fig, ax1 = plt.subplots(figsize=(9, 5))
sns.barplot(data=padding_df, x="context_length", y="padding_fraction", ax=ax1, color="#4c78a8")
ax1.set_title("Short-context padding creates SEBP opportunity")
ax1.set_ylabel("Mean padding fraction")
ax1.set_xlabel("Training context length")
ax1.set_ylim(0, 1)
for container in ax1.containers:
    ax1.bar_label(container, fmt="%.2f")
plt.tight_layout()
plot_path = PLOTS_DIR / "sebp_padding_opportunity.png"
plt.savefig(plot_path, dpi=180)
plot_path
`),

  md(`
## 6. Train/Eval Split
`),

  code(`
#@title Split dataset
TRAIN_PATH = OUTPUTS_DIR / "train.jsonl"
EVAL_PATH = OUTPUTS_DIR / "eval.jsonl"

train_examples, eval_examples = train_test_split(
    examples,
    test_size=0.15,
    random_state=SEED,
    stratify=df["source"] if df["source"].nunique() > 1 else None,
)

write_jsonl(TRAIN_PATH, train_examples)
write_jsonl(EVAL_PATH, eval_examples)

print("Train:", len(train_examples), TRAIN_PATH)
print("Eval:", len(eval_examples), EVAL_PATH)
`),

  md(`
## 7. Model Experiment Configs

Start with one model, then add more if time allows. Good options:

- \`Qwen/Qwen2.5-1.5B-Instruct\`: fastest baseline.
- \`Qwen/Qwen2.5-3B-Instruct\`: stronger, still manageable.
- \`Qwen/Qwen3-4B-Instruct-2507\`: strongest Qwen target if available and supported.
- \`google/gemma-2-2b-it\` or newer Gemma small instruct model: optional comparison, may require accepting license terms.

For Colab A100, 4-bit LoRA should be comfortable for these sizes.
`),

  code(`
#@title Model configs
MODEL_CONFIGS = [
    {
        "run_name": "qwen25_15b_lora",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_seq_length": 1024,
        "lora_r": 16,
        "lr": 2e-4,
        "epochs": 3,
        "batch_size": 2,
        "grad_accum": 4,
        "max_grad_norm": 0.5,
    },
    {
        "run_name": "qwen25_3b_lora",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "max_seq_length": 1024,
        "lora_r": 16,
        "lr": 2e-4,
        "epochs": 3,
        "batch_size": 1,
        "grad_accum": 8,
        "max_grad_norm": 0.5,
    },
    {
        "run_name": "qwen3_4b_lora",
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "max_seq_length": 1024,
        "lora_r": 16,
        "lr": 1.5e-4,
        "epochs": 3,
        "batch_size": 1,
        "grad_accum": 8,
        "max_grad_norm": 0.5,
    },
    {
        "run_name": "gemma_small_lora",
        "model_id": "google/gemma-2-2b-it",
        "max_seq_length": 1024,
        "lora_r": 16,
        "lr": 1.5e-4,
        "epochs": 3,
        "batch_size": 1,
        "grad_accum": 8,
        "max_grad_norm": 0.5,
    },
]

# Keep this short for the first run. Add more run names after the pipeline works.
RUN_EXPERIMENTS = ["qwen25_15b_lora"]  #@param {type:"raw"}
CONFIG_BY_NAME = {cfg["run_name"]: cfg for cfg in MODEL_CONFIGS}
selected_configs = [CONFIG_BY_NAME[name] for name in RUN_EXPERIMENTS]
selected_configs
`),

  code(`
#@title Training helpers
from datasets import load_dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel

class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.rows = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            row = {"step": int(state.global_step), **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}}
            self.rows.append(row)

def load_chat_dataset(tokenizer, train_path, eval_path):
    dataset = load_dataset("json", data_files={"train": str(train_path), "eval": str(eval_path)})

    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return dataset.map(format_example, remove_columns=dataset["train"].column_names)

def target_modules_for_model(model_id):
    # Qwen/Gemma LLM linear module names are compatible with this common target list.
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def train_one_config(cfg):
    run_dir = MODELS_DIR / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Training:", cfg)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_id"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=target_modules_for_model(cfg["model_id"]),
        lora_alpha=cfg["lora_r"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    dataset = load_chat_dataset(tokenizer, TRAIN_PATH, EVAL_PATH)
    callback = LossHistoryCallback()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        packing=False,
        args=TrainingArguments(
            output_dir=str(run_dir),
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            num_train_epochs=cfg["epochs"],
            learning_rate=cfg["lr"],
            warmup_ratio=0.05,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=10,
            save_steps=50,
            max_grad_norm=cfg["max_grad_norm"],
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            report_to="none",
            seed=SEED,
        ),
        callbacks=[callback],
    )

    started = time.time()
    train_result = trainer.train()
    elapsed = time.time() - started

    model.save_pretrained(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))

    metrics = dict(train_result.metrics)
    metrics["elapsed_seconds"] = elapsed
    metrics["run_name"] = cfg["run_name"]
    metrics["model_id"] = cfg["model_id"]
    metrics["train_examples"] = len(dataset["train"])
    metrics["eval_examples"] = len(dataset["eval"])
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    loss_df = pd.DataFrame(callback.rows)
    loss_df.to_csv(run_dir / "loss_history.csv", index=False)
    print("Saved:", run_dir)
    return {"cfg": cfg, "run_dir": run_dir, "model": model, "tokenizer": tokenizer, "metrics": metrics, "loss_df": loss_df}
`),

  code(`
#@title Run training experiments
TRAIN_NOW = True  #@param {type:"boolean"}

training_results = []
if TRAIN_NOW:
    for cfg in selected_configs:
        result = train_one_config(cfg)
        training_results.append(result)
        # Free memory between runs if you train several models.
        torch.cuda.empty_cache()
else:
    print("TRAIN_NOW is False; skipping training.")
`),

  code(`
#@title Plot training/eval loss curves
loss_frames = []

if training_results:
    for result in training_results:
        loss_df = result["loss_df"].copy()
        if not loss_df.empty:
            loss_df["run_name"] = result["cfg"]["run_name"]
            loss_frames.append(loss_df)
else:
    for cfg in selected_configs:
        csv_path = MODELS_DIR / cfg["run_name"] / "loss_history.csv"
        if csv_path.exists():
            loss_df = pd.read_csv(csv_path)
            loss_df["run_name"] = cfg["run_name"]
            loss_frames.append(loss_df)

if loss_frames:
    losses = pd.concat(loss_frames, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    if "loss" in losses:
        sns.lineplot(data=losses.dropna(subset=["loss"]), x="step", y="loss", hue="run_name", marker="o", ax=axes[0])
        axes[0].set_title("Training loss")
    if "eval_loss" in losses:
        sns.lineplot(data=losses.dropna(subset=["eval_loss"]), x="step", y="eval_loss", hue="run_name", marker="o", ax=axes[1])
        axes[1].set_title("Eval loss")
    plt.tight_layout()
    plot_path = PLOTS_DIR / "training_loss_curves.png"
    plt.savefig(plot_path, dpi=180)
    display(losses.tail())
    print("Saved:", plot_path)
else:
    print("No loss history found yet.")
`),

  md(`
## 8. Qualitative Evaluation

This creates a small before/after style sanity check. It also scores basic constraints: valid JSON, one short line, allowed emotion/action, and no obvious internal/debug language.
`),

  code(`
#@title Inference and quality scoring helpers
ALLOWED_EMOTIONS = {"annoyed", "suspicious", "scared", "confused", "smug", "offended"}
ALLOWED_ACTIONS = {"stare_at_player", "step_back", "look_at_nearby_block", "mutter", "panic", "ignore"}

TEST_SCENES = [
    {
        "transcript": "Is this your house?",
        "scene": ["Player is inside a villager house.", "Villager has a bed nearby.", "Nearest villager is 0.9 blocks away."],
    },
    {
        "transcript": "There is a creeper behind you.",
        "scene": ["Danger: hostile mobs nearby: 1 creeper.", "World time is night."],
    },
    {
        "transcript": "Do you want this potato?",
        "scene": ["Dropped item nearby: potato.", "Nearest villager profession is farmer."],
    },
    {
        "transcript": "Am I standing too close?",
        "scene": ["Nearest villager is 0.5 blocks away."],
    },
    {
        "transcript": "Do you remember what I gave you?",
        "scene": ["Recent memory: Player offered a potato.", "Dropped item nearby: potato."],
    },
]

def payload_for_test(scene, i=0):
    return {
        "event": "player_spoke",
        "transcript": scene["transcript"],
        "transcript_source": "manual_eval",
        "audio_duration_ms": 1500,
        "player": {"uuid": "eval-player", "name": "EvalPlayer"},
        "villager": {"uuid": "eval-villager", "profession": "unemployed", "distance": 1.0},
        "scene": ["Player spoke through Simple Voice Chat.", *scene["scene"]],
        "recent_memory": [s.replace("Recent memory: ", "") for s in scene["scene"] if s.startswith("Recent memory:")],
    }

def parse_response_text(text):
    text = text.strip()
    match = re.search(r"\\{.*\\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        return json.loads(text), True
    except Exception:
        return {"line": text[:180], "emotion": "unknown", "action": "unknown", "memory_update": ""}, False

def quality_score(parsed, valid_json):
    line = str(parsed.get("line", ""))
    emotion = str(parsed.get("emotion", ""))
    action = str(parsed.get("action", ""))
    score = 0
    score += int(valid_json)
    score += int(5 <= len(line) <= 180)
    score += int(line.count(".") + line.count("!") + line.count("?") <= 2)
    score += int(emotion in ALLOWED_EMOTIONS)
    score += int(action in ALLOWED_ACTIONS)
    score += int("java" not in line.lower() and "debug" not in line.lower())
    return score

def generate_with_model(model, tokenizer, payload, max_new_tokens=80):
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": json.dumps(user_payload_from_input(payload), ensure_ascii=False)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
`),

  code(`
#@title Evaluate trained adapters on fixed prompts
eval_rows = []

if training_results:
    for result in training_results:
        for i, scene in enumerate(TEST_SCENES):
            payload = payload_for_test(scene, i)
            text = generate_with_model(result["model"], result["tokenizer"], payload)
            parsed, valid_json = parse_response_text(text)
            eval_rows.append({
                "run_name": result["cfg"]["run_name"],
                "prompt": scene["transcript"],
                "raw": text,
                "line": parsed.get("line", ""),
                "emotion": parsed.get("emotion", ""),
                "action": parsed.get("action", ""),
                "valid_json": valid_json,
                "quality_score": quality_score(parsed, valid_json),
            })

eval_df = pd.DataFrame(eval_rows)
if not eval_df.empty:
    display(eval_df[["run_name", "prompt", "line", "emotion", "action", "valid_json", "quality_score"]])
    eval_df.to_csv(OUTPUTS_DIR / "qualitative_eval.csv", index=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=eval_df, x="run_name", y="quality_score")
    plt.title("Heuristic dialogue quality score")
    plt.ylim(0, 6)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plot_path = PLOTS_DIR / "heuristic_quality_score.png"
    plt.savefig(plot_path, dpi=180)
    print("Saved:", plot_path)
else:
    print("No trained models in memory; run training first.")
`),

  md(`
## 9. Export and Quantization

For a course submission, the LoRA adapter is enough. For players, the nicer path is:

1. Merge adapter into base model.
2. Convert/quantize to GGUF.
3. Load GGUF in LM Studio / llama.cpp / Ollama.
4. Point the Minecraft AI server at the local model server.

Unsloth can export GGUF for supported models. If a model is unsupported, push the adapter to Hugging Face and convert later with llama.cpp.
`),

  code(`
#@title Save merged model / GGUF for local inference
EXPORT_GGUF = False  #@param {type:"boolean"}
GGUF_QUANTIZATION = "q4_k_m"  #@param ["q4_k_m", "q5_k_m", "q8_0"]

if EXPORT_GGUF and training_results:
    for result in training_results:
        cfg = result["cfg"]
        model = result["model"]
        tokenizer = result["tokenizer"]
        export_dir = MODELS_DIR / cfg["run_name"] / f"gguf_{GGUF_QUANTIZATION}"
        export_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained_gguf"):
            model.save_pretrained_gguf(str(export_dir), tokenizer, quantization_method=GGUF_QUANTIZATION)
            print("Saved GGUF:", export_dir)
        else:
            print("This model object does not expose save_pretrained_gguf. Export adapter and convert separately:", cfg["run_name"])
else:
    print("GGUF export skipped.")
`),

  code(`
#@title Push adapter to Hugging Face Hub
PUSH_TO_HUB = False  #@param {type:"boolean"}
HF_NAMESPACE = "OnAIron"  #@param {type:"string"}

if PUSH_TO_HUB:
    from huggingface_hub import login
    login()
    for result in training_results:
        cfg = result["cfg"]
        repo_id = f"{HF_NAMESPACE}/living-villagers-{cfg['run_name']}"
        result["model"].push_to_hub(repo_id)
        result["tokenizer"].push_to_hub(repo_id)
        print("Pushed:", repo_id)
else:
    print("Hub push skipped.")
`),

  md(`
## 10. Course Report Artifacts

Use these generated files in the report/presentation:

- \`dataset_composition.png\`
- \`raw_session_latency.png\`
- \`sebp_padding_opportunity.png\`
- \`training_loss_curves.png\`
- \`heuristic_quality_score.png\`
- \`metrics.json\`
- \`qualitative_eval.csv\`

Suggested result table:

| Run | Base model | Train rows | Eval rows | Final train loss | Eval loss | Notes |
|---|---|---:|---:|---:|---:|---|
| qwen25_15b_lora | Qwen2.5 1.5B Instruct | ... | ... | ... | ... | fastest |
| qwen25_3b_lora | Qwen2.5 3B Instruct | ... | ... | ... | ... | better quality |
| qwen3_4b_lora | Qwen3 4B Instruct | ... | ... | ... | ... | strongest local target |

Research angle:

- Our dialogue examples are short-context.
- Padding to fixed context length creates structured sparsity in output gradients.
- SEBP can exploit this during transformer fine-tuning.
- We show the padding fraction and compare normal LoRA training settings; a future/custom SEBP kernel can replace the normal backward pass for speedup.
`),

  code(`
#@title Build report summary JSON
summary = {
    "dataset": {
        "examples": len(df),
        "source_counts": df["source"].value_counts().to_dict(),
        "emotion_counts": df["emotion"].value_counts().to_dict(),
        "action_counts": df["action"].value_counts().to_dict(),
        "mean_audio_duration_ms": float(df["audio_duration_ms"].mean()),
        "mean_scene_facts": float(df["scene_facts"].mean()),
    },
    "sebp_padding_opportunity": padding_df.to_dict(orient="records"),
    "model_runs": [],
}

for cfg in selected_configs:
    metrics_path = MODELS_DIR / cfg["run_name"] / "metrics.json"
    if metrics_path.exists():
        summary["model_runs"].append(json.loads(metrics_path.read_text()))

summary_path = OUTPUTS_DIR / "course_report_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(summary_path)
print(json.dumps(summary, indent=2)[:2000])
`),
];

const notebook = {
  cells,
  metadata: {
    colab: {
      provenance: [],
      gpuType: "A100",
    },
    kernelspec: {
      display_name: "Python 3",
      name: "python3",
    },
    language_info: {
      name: "python",
    },
  },
  nbformat: 4,
  nbformat_minor: 5,
};

fs.mkdirSync(path.dirname(notebookPath), { recursive: true });
fs.writeFileSync(notebookPath, JSON.stringify(notebook, null, 2), "utf8");
console.log(`Wrote ${notebookPath}`);
