# Onairon's Living Villagers

Working design notes for a future Fabric mod where villagers can hear nearby players through proximity chat, understand local world context, and respond with short, funny, reactive voice lines.

This document is meant to be migrated into the actual project later. It was drafted inside the existing creeper drones workspace only as planning material.

## Core Idea

Most AI villager mods are basically:

```text
player speaks
-> giant LLM prompt
-> long model monologue
-> TTS reads paragraph
```

This mod should be different:

```text
world event or player speech
-> compact perception packet
-> villager reaction planner
-> short line + physical action + memory update
-> proximity voice response
```

The goal is not "villagers narrate paragraphs." The goal is villagers that behave like reactive, social, rude, funny little Minecraft people.

Example:

```text
Player tramples crops.

Farmer:
- snaps head toward player
- walks closer
- says: "That was food. Was. Past tense."
- refuses trade for 30 seconds
- remembers: player trampled crops
```

## Name Candidates

Preferred:

```text
Mod Name: Onairon's Living Villagers
Mod ID: onairon_living_villagers
Package: com.onairon.livingvillagers
```

Other possible names:

```text
Onairon's Living Entities
Onairon's Smart Villagers
Onairon's Villager Civilization
Village Minds
Living Villages
```

Recommended approach: start with **Onairon's Living Villagers** because it is clear, searchable, and focused. If the project expands to mobs and other entities, use **Onairon's Living Entities** as the broader umbrella later.

## Fabric Template Settings

Recommended Fabric template generator choices:

```text
Mod Name:
Onairon's Living Villagers

Use custom id:
yes

Custom Mod ID:
onairon_living_villagers

Package Name:
com.onairon.livingvillagers

Minecraft Version:
Use the target version that Simple Voice Chat supports.
For current experiments this may be 1.21.11, but verify Simple Voice Chat compatibility first.
```

Advanced options:

```text
Kotlin Programming Language: no
Mojang Mappings: yes
Data Generation: no for the first prototype
Split client and common sources: yes
Kotlin Build Script: no
```

Reasoning:

- Java is the least-friction path for Fabric and Simple Voice Chat examples.
- Split client/common sources helps avoid dedicated-server crashes from accidental client-only calls.
- Data generation is unnecessary until the mod adds assets, items, blocks, recipes, or generated resources.
- Mojang mappings are reasonable for newer Fabric versions, but this can be revisited if a dependency/template strongly favors Yarn.

## Simple Voice Chat Dependency Model

There are two separate things:

1. **Simple Voice Chat runtime mod**
   - Installed as its own `.jar` in the Minecraft/server `mods` folder.
   - Needed for actual proximity voice chat.
   - Players usually need it installed on their clients too.

2. **Simple Voice Chat API dependency**
   - Added to our mod's Gradle build.
   - Lets our code compile against classes such as `VoicechatPlugin`.
   - Our mod registers a `voicechat` entrypoint in `fabric.mod.json`.

Do not bundle Simple Voice Chat inside this mod. Treat it like a separate required dependency.

Typical runtime layout:

```text
server/mods/
  fabric-api.jar
  voicechat-fabric-....jar
  onairon-living-villagers.jar
```

Typical development concept:

```text
build.gradle
  -> depends on Simple Voice Chat API

fabric.mod.json
  -> depends on voicechat_api
  -> has a voicechat entrypoint

Minecraft mods folder
  -> contains the actual Simple Voice Chat mod jar
```

## Local Development Setup

Use a local dedicated Fabric server first, even when testing alone.

```text
Minecraft client
  -> local dedicated Fabric server on localhost
      -> Fabric API
      -> Simple Voice Chat
      -> Onairon's Living Villagers
          -> Python AI server
```

This avoids ambiguity around singleplayer/integrated-server behavior and tests the real server-mediated proximity chat path.

Singleplayer support can be considered later. If Simple Voice Chat is awkward in singleplayer, a fallback could use direct mic capture plus Minecraft positional sound playback, but that is not the main target.

## High-Level Architecture

```text
Fabric Minecraft server
  - listens to world events
  - receives Simple Voice Chat mic packets
  - tracks nearby villagers
  - builds compact context packets
  - sends AI jobs to Python server
  - applies actions and plays generated voice

Python AI server
  - speech-to-text
  - villager context processing
  - LLM response generation
  - text-to-speech
  - audio effects / villager voice style
  - optional memory storage
```

Principle:

```text
Minecraft owns reality.
AI owns interpretation.
Scheduler owns latency.
```

## Voice Pipeline

Target pipeline:

```text
Player speaks near villager
-> Simple Voice Chat mic packets
-> Fabric addon buffers audio
-> voice activity detection / end-of-speech
-> Python STT
-> context builder
-> LLM villager brain
-> TTS
-> villager voice effects
-> Simple Voice Chat positional playback from villager
```

Important: do not send every tiny mic packet directly to the LLM. Buffer audio into utterances first.

Possible first implementation:

```text
1. Capture microphone packets from Simple Voice Chat.
2. Detect a speech segment.
3. Find nearby villagers.
4. Pick the most relevant villager.
5. Send audio + context to Python.
6. Get back text + generated audio.
7. Play audio from villager position.
```

## Villager Perception

Villagers do not need real computer vision at first. Minecraft already has structured world state.

Possible perception facts:

```text
Nearby player position
Nearby entities
Nearby blocks
Profession/job site
Owned/nearby crops
Recent block changes
Recent crop trampling
Recent chest opening
Recent attacks
Current time/weather
Nearby hostile mobs
Player held item
Player reputation
Village gossip/memory
```

Keep the context packet small and ranked by relevance.

Bad:

```text
Send the whole world and ask the model what matters.
```

Good:

```text
Send the 10-30 most relevant facts and a clear reaction goal.
```

## Reaction Planner

The LLM should return structured output, not just plain text.

Example:

```json
{
  "line": "That was food. Was. Past tense.",
  "emotion": "offended",
  "action": "stare_at_player",
  "memory_update": "Player trampled my crops.",
  "trade_lock_seconds": 30,
  "urgency": 0.72,
  "should_continue": false
}
```

Hard rules for stage one:

```text
Max 1-2 sentences.
No generic assistant tone.
No exposition.
No long paragraphs.
No explaining Minecraft mechanics.
React to what just happened.
Prefer jokes, insults, fear, bargaining, gossip, or panic.
Output actions, not only text.
```

## First Funny Reactions To Build

High-value first triggers:

```text
crop trampled near farmer
villager hit
player opens nearby chest
player rings bell repeatedly
player holds weapon near villager
zombie or raid danger nearby
player steals/uses bed
player throws emerald
player stares silently
player trades badly
player blocks villager path
```

Profession instincts:

```text
farmer: crops, composters, food, weather
librarian: bookshelves, silence, knowledge, lecterns
blacksmith/toolsmith/weaponsmith: anvils, lava, metal, weapons
cleric: undead, death, potions, ominous comments
butcher: animals, food, suspicious meat jokes
nitwit: confidently wrong reactions
```

## Voice Style

The goal is not to clone any specific creator's voice. The goal is an original "unhinged Minecraft villager" sound:

```text
nasal
compressed
raspy
short comedic rhythm
slightly angry
occasional "hrm"
villager-ish timing
not a normal assistant voice
```

Possible TTS/STT stack:

```text
STT:
faster-whisper or Whisper

LLM:
Qwen / Gemma / other open-weight model

TTS:
Kokoro, Piper, OpenVoice, or another open/free local TTS option

Audio effects:
pitch shift
formant/nasal EQ
compression
light distortion
short slapback delay
insert villager-ish grunts
```

The writing style matters as much as the voice model.

Example lines:

```text
"Wonderful. The boots have opinions now."
"Please exit the wheat, you walking disaster."
"That was food. Was. Past tense."
"Hrm. I am choosing peace with great difficulty."
"Could you maybe stop crushing the crops I need to survive?"
```

## AI Server Endpoints

Early prototype can use one endpoint:

```text
POST /pipeline/villager-turn
```

Input:

```json
{
  "audio_format": "pcm16",
  "audio": "...",
  "player": {
    "uuid": "...",
    "name": "Player"
  },
  "villager": {
    "uuid": "...",
    "profession": "farmer",
    "mood": "annoyed"
  },
  "scene": [
    "Player is standing inside wheat field.",
    "Three farmland blocks were trampled in the last 10 seconds.",
    "Villager owns nearby composter.",
    "Time is morning."
  ],
  "memory": [
    "Player previously stole wheat."
  ]
}
```

Output:

```json
{
  "transcript": "Hey, got any food?",
  "line": "You have a strange way of asking for food after stomping through my field.",
  "emotion": "annoyed",
  "action": "look_at_player",
  "memory_update": "Player asked for food after trampling crops.",
  "audio_format": "opus_or_pcm",
  "audio": "..."
}
```

Later services can be split:

```text
POST /voice/transcribe
POST /villager/respond
POST /voice/synthesize
POST /memory/update
```

## Hardware / Research Direction

The mod can become a real distributed inference research platform.

Task classes:

```text
STT: streaming, latency-sensitive
LLM: heavy, batching-friendly
TTS: medium, latency-sensitive
embeddings/memory: small and frequent
summarization: background
world-event ranking: small classifier or rules
```

Possible hardware split:

```text
A100 / CUDA GPUs:
main LLM inference
large model experiments
batching

NPUs:
STT if supported
TTS if supported
embeddings
small classifiers
reaction ranking
background summarization

CPU:
fallback
world event processing
lightweight rules
```

Research metrics:

```text
p50/p95 response latency
first-token latency
audio roundtrip latency
villagers served concurrently
queue time by task type
GPU/NPU utilization
quality vs latency tradeoff
cost per dialogue turn
failure/fallback rate
```

Possible paper framing:

```text
Heterogeneous Low-Latency Inference for Real-Time Embodied NPCs in Open-World Games

VillagerSim: A Minecraft-Based Benchmark for Voice-Interactive Multi-Agent AI Systems
```

Potential venues:

```text
MLSys
SC
HPDC
IPDPS
EuroSys
USENIX ATC
CHI
UIST
IUI
AIIDE
FDG
AAMAS
INTERSPEECH
ICASSP
```

## Milestones

### Milestone 1: Mod Skeleton

```text
Fabric mod loads.
Simple Voice Chat dependency is declared.
VoicechatPlugin entrypoint initializes.
Server logs that Living Villagers voice plugin loaded.
```

### Milestone 2: Voice Detection

```text
Simple Voice Chat mic packets are observed.
Player UUID is identified.
Nearby villagers are found.
Debug log shows which villager can "hear" the player.
```

### Milestone 3: Audio Segment

```text
Mic packets are buffered into a short speech segment.
Segment has a clear start/end.
Segment can be sent to Python as PCM/WAV/Opus.
```

### Milestone 4: Text-Only AI Reaction

```text
Python receives transcript or mock transcript.
Minecraft sends scene context.
LLM returns short structured JSON.
Villager performs action and sends text chat/debug response.
```

### Milestone 5: TTS Playback

```text
Python generates villager voice audio.
Fabric mod receives audio.
Audio is played from villager position through Simple Voice Chat or fallback sound path.
```

### Milestone 6: Real Reactions

```text
Villagers react to crop trampling, chest opening, hits, bell spam, zombie danger.
Reactions are short and funny.
Villagers update per-player memory/reputation.
```

### Milestone 7: Village Memory

```text
Villagers remember player behavior.
Villagers gossip or share reputation.
Village mood changes over time.
```

### Milestone 8: Distributed Backend

```text
Separate STT, LLM, TTS, memory, and scheduler workers.
Run across GPU/NPU/CPU resources.
Collect latency and utilization metrics.
```

## First Vertical Slice

The first truly satisfying demo should be:

```text
Player joins local Fabric server.
Player walks into a farmer's crop field.
Player says something through proximity chat.
Player tramples crops.
Farmer turns, walks closer, and says through proximity voice:
"That was food. Was. Past tense."
Farmer remembers the incident.
```

That is the smallest version that proves the core identity of the mod.

