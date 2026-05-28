# Living Villagers AI Server

## Mock Mode

```powershell
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

This is the default mode. It proves the Minecraft -> Python -> Minecraft loop, logs dataset rows, and lets the villager speak through the Simple Voice Chat audio channel with the Java-side mumble voice.

## Hugging Face Transformers Mode

Install the correct `torch` build for the machine first, then:

```powershell
pip install -r requirements-transformers.txt
$env:LV_AI_BACKEND="transformers"
$env:LV_MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Linux/bash:

```bash
export LV_AI_BACKEND=transformers
export LV_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

## OpenAI-Compatible Model Server Mode

Use this when Huawei/A100 runs a separate model server with `/v1/chat/completions`.

```powershell
$env:LV_AI_BACKEND="openai"
$env:LV_OPENAI_BASE_URL="http://YOUR_MODEL_SERVER:8001/v1"
$env:LV_MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Linux/bash:

```bash
export LV_AI_BACKEND=openai
export LV_OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export LV_MODEL_ID=Qwen/Qwen3-30B-A3B-Instruct-2507
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

## Optional Speech-to-Text

The Java mod now sends decoded 48 kHz mono PCM with each accepted speech segment. By default the server keeps using the mocked transcript. To enable local Whisper transcription:

```bash
pip install -r requirements-stt.txt
export LV_STT_BACKEND=faster_whisper
export LV_STT_MODEL_ID=base.en
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

For a stronger model, try `small.en` or `medium.en` after the basic loop works.

Every interaction is logged automatically to:

```text
ai_server/data/interactions_raw.jsonl
```

The log keeps transcript/context/metadata and intentionally does not store the raw base64 audio blob.

## Dataset Export

For clean data gathering, use one session folder per Minecraft playtest:

```powershell
.\run_dataset_session.ps1
```

After stopping the server, finalize that session:

```powershell
.\finalize_dataset_session.ps1 -SessionId 20260527T192300Z
```

See `DATASET_GATHERING.md` for the practical collection protocol.

After playing in Minecraft, convert raw logs into supervised fine-tuning examples:

```powershell
python export_sft_dataset.py --input data/interactions_raw.jsonl --output data/villager_sft_train.jsonl --only-ok
```

For cleaner fine-tuning data after real STT/model inference is connected:

```bash
python export_sft_dataset.py --input data/interactions_raw.jsonl --output data/villager_sft_train.jsonl --only-ok --skip-mock --require-real-transcript --min-duration-ms 900
```

## Course Report Plots

```bash
pip install -r requirements-analysis.txt
python plot_dataset_stats.py --input data/interactions_raw.jsonl --output-dir data/plots
```

This creates quick plots for latency, speech duration, and number of scene facts per interaction.

## Unsloth LoRA Fine-Tuning

Use this on CUDA/A100 first. Unsloth is the right quick path for Qwen LoRA/QLoRA; Huawei Ascend/NPU fine-tuning may need a different runtime depending on your server stack.

```powershell
pip install -r requirements-unsloth.txt
python train_lora_unsloth.py --model Qwen/Qwen3-4B-Instruct-2507 --data data/villager_sft_train.jsonl --output outputs/qwen3-4b-living-villagers-lora
```

Start fine-tuning with the dense 4B model first. For stronger A100 experiments, use Qwen3-30B-A3B for inference first, then follow the current Unsloth MoE guide for fine-tuning because MoE models need special handling.

Useful model choices:

- `Qwen/Qwen3-4B-Instruct-2507`: best first LoRA target.
- `Qwen/Qwen3-30B-A3B-Instruct-2507`: stronger inference target; MoE fine-tuning needs current Unsloth MoE workflow.
- Gemma 4: try after Qwen works; you may need to accept Google's model terms on Hugging Face/Kaggle before downloading.
