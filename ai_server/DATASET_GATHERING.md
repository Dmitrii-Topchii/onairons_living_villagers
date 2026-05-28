# Dataset Gathering Protocol

This is the quick, repeatable workflow for collecting fine-tuning data from real Minecraft interactions.

## 1. Start LM Studio

Load the fastest model that gives acceptable dialogue. Current best local choice:

```text
qwen2-1.5b-instruct
```

Keep the LM Studio local server running at:

```text
http://127.0.0.1:1234/v1
```

## 2. Start One Dataset Session

From `ai_server`:

```powershell
.\run_dataset_session.ps1
```

The script prints a session id like:

```text
20260527T192300Z
```

Each session writes raw rows to:

```text
data/sessions/<SESSION_ID>/interactions_raw.jsonl
```

## 3. Collect Scenes In Minecraft

For the first clean dataset, collect 50-150 short interactions. Speak clearly, one thought at a time, and let the villager finish before asking again.

Useful scene buckets:

- Neutral chat: ask name, job, village gossip, trade questions.
- Intrusion: walk into a house, stand too close, stare at the villager.
- Danger: night, zombie nearby, creeper nearby, villager panic.
- Object event: drop potatoes, bread, emeralds, blocks near the villager.
- Bad behavior: hit a nearby mob, trample crops, break a block near the villager.
- Memory: ask follow-up questions about something the villager just said.

Good examples for fine-tuning have:

- real transcript source: `faster_whisper`
- successful model response: `ok=true`
- player speech duration above about 700 ms
- short, in-character line
- scene facts that actually matter

## 4. Finalize The Session

Stop the server with `Ctrl+C`, then run:

```powershell
.\finalize_dataset_session.ps1 -SessionId <SESSION_ID>
```

Outputs:

```text
data/sessions/<SESSION_ID>/villager_sft_train.jsonl
data/sessions/<SESSION_ID>/plots/
data/sessions/<SESSION_ID>/plots/summary.json
```

Use the plots in the course report to show latency, speech duration, transcript source quality, and scene-context size.

## 5. Training Story

For the course report, describe three levels:

- Baseline: mock response backend.
- Pretrained model: Qwen through LM Studio with Whisper STT and scene context.
- Fine-tuned model: SFT/LoRA on collected villager interactions.

For the research angle, compare normal fine-tuning with SEBP-style accelerated backward passes on short padded dialogue examples.
