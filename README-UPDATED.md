# VisMem-C: Visual Memory-Centric Conversational Dataset

A complete pipeline for generating multi-session conversational datasets with integrated visual content, from dialogue generation through image synthesis to final dataset assembly.

## System Architecture

VisMem-C is a **three-stage pipeline** that transforms scripted events into a fully annotated conversational dataset with S3-hosted images:

```
Stage 1: Conversation Generation          Stage 2: Image Generation              Stage 3: LoCoMo Output
(generate_conversations.py)               (regenerate_images.py)                (build_vismen_c.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Events + Prompts        Agent Transcripts         Images on S3           Final Dataset
(events.json)           (agent_a.json)           (Updated events.json)  (vismen-c.json)
    ↓                        ↓                            ↓                    ↓
┌─────────────┐         ┌──────────────┐         ┌───────────────┐    ┌─────────────┐
│   Events    │         │  Agents A/B  │         │   Imagen/    │    │ LoCoMo      │
│ with text & │────→    │  Multi-turn  │────→    │  OpenAI      │───→│ Format      │
│ prompts     │         │  dialogue    │         │  + S3        │    │ Output      │
└─────────────┘         └──────────────┘         └───────────────┘    └─────────────┘
                        agent_a.json              vismen-c.json
                        agent_b.json              (final)
```

---

## Overview

VisMem-C produces conversational datasets designed for **visual memory tasks**:

**What it generates:**
- **Multi-session conversations** between two agents discussing various topics
- **Visual events** seamlessly integrated into dialogue (photos, receipts, documents)
- **Image synthesis** using Google Vertex AI Imagen or OpenAI DALL-E
- **S3-hosted images** with public URLs automatically embedded in datasets
- **Structured metadata** including QA pairs, observations, session summaries, and event-based facts
- **LoCoMo-compatible output** ready for downstream evaluation systems

**Use cases:**
- Training models on multimodal long-context understanding
- Evaluating visual grounding in conversational contexts
- Testing memory recall with visual references
- Analyzing how agents integrate visual information into dialogue

---

## Project Structure

```
vismem-c/
├── README.md                           # This file - full system documentation
├── QUICKSTART.md                       # 5-minute quick start guide
├── ARCHITECTURE.md                     # Detailed architectural overview
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore patterns
├── LICENSE                             # License (add your choice)
│
├── scripts/
│   ├── regenerate_images.py            # Stage 2: Image generation + S3 upload pipeline
│   ├── build_vismen_c.py               # Stage 3: LoCoMo format assembly
│   ├── global_methods.py               # Shared utilities (LLM calls, embeddings, retry logic)
│   └── generate_conversations.py       # Stage 1: Multi-session conversation generation
│
├── backends/
│   ├── __init__.py                     # Backend factory (get_backend)
│   ├── base.py                         # Abstract base class for image generators
│   ├── google_imagen.py                # Google Vertex AI Imagen backend
│   └── openai_backend.py               # OpenAI DALL-E backend (optional)
│
├── data/
│   └── metadata/
│       ├── events.json                 # Visual events with image prompts (→ S3 URLs)
│       ├── agent_a.json                # Agent A multi-session transcript
│       ├── agent_b.json                # Agent B multi-session transcript
│       ├── qa.json                     # Human-annotated QA pairs
│       └── generated_images/           # Local image cache (generated during Stage 2)
│
└── outputs/
    └── vismen-c.json                   # Final LoCoMo-formatted dataset
```

---

## Pipeline Stages

### Stage 1: Conversation Generation

**Script:** `scripts/generate_conversations.py`

**What it does:**
- Takes events.json with text and image prompts (but no URLs yet)
- Generates multi-turn dialogues between two agents using OpenAI API
- Creates agent_a.json and agent_b.json with conversation transcripts
- Each turn includes dia_id (dialogue turn identifier) for linking to events

**Key Issues & Fixes:**

⚠️ **Issue: OpenAI API v1.0+ compatibility**
```
APIRemovedInV1: You tried to access openai.ChatCompletion, 
but this is no longer supported in openai>=1.0.0
```

**Solution:** Pin to OpenAI v0.28 (legacy API)
```bash
pip install openai==0.28
```

Or update `global_methods.py` to use new API:
```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
completion = client.chat.completions.create(...)
```

**Usage:**
```bash
python generative_agents/generate_conversations.py \
  --out-dir data/metadata \
  --prompt-dir prompts \
  --persona --session --summary \
  --num-sessions 30 \
  --max-turns-per-session 18
```

**Output:** `agent_a.json`, `agent_b.json` with conversation transcripts

---

### Stage 2: Image Generation & S3 Upload

**Script:** `scripts/regenerate_images.py`

**What it does:**
- Reads events.json (with prompts but missing URLs)
- For each event with an image:
  - Extracts the image prompt
  - Calls the image backend (Imagen, DALL-E, etc.)
  - Generates high-quality image locally
  - Uploads to AWS S3 with semantic naming
  - Updates events.json with public S3 URL
- Provides retry logic for safety filter issues
- Supports partial re-runs and dry-mode testing

**Architecture:**

```
ImageRegenerationPipeline (orchestrator)
    ├─ load_json(path) → dict
    ├─ get_image_prompt(event) → prompt string (with fallback chain)
    ├─ generate_image(prompt, idx) → local file path (calls backend)
    ├─ upload_to_s3(local_path, idx, dia_id) → public S3 URL
    ├─ process_events(start, num) → updates events.json in-place
    └─ print_summary() → statistics report
```

**Backend Abstraction:**

```
ImageGenerator (abstract base)
    ├─ generate(prompt, count, output_dir) → [{'filepath': ..., 'generation_time': ...}]
    ├─ get_backend_name() → str
    
Implementation:
    GoogleImagenBackend (uses Google Vertex AI Imagen @006)
        ├─ Retry logic with prompt simplification
        ├─ Progressive safety filter relaxation
        ├─ Automatic prompt enhancement
        
    OpenAIBackend (uses OpenAI DALL-E 3) [optional]
```

**Key Issues & Fixes:**

**Issue: Google Cloud authentication failure**
```
403 Permission denied on resource project your-default-project-id.
[reason: "CONSUMER_INVALID"
```

**Cause:** Using placeholder project ID `'your-default-project-id'` in code

**Solution:** Set actual GCP project ID
```bash
# Option 1: Environment variable
export GOOGLE_CLOUD_PROJECT=your-actual-project-id

# Option 2: Update google_imagen.py
project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('GCP_PROJECT_ID') or 'aadhi-project-123'

# Find your project ID
gcloud config get-value project
```

 **Issue: Generation getting skipped**
```
Events skipped: 3
Images generated: 0
Images uploaded: 0
```

**Cause:** Three skip conditions in `process_events()`:
1. No `img` metadata in event
2. URL already set to S3
3. External URL exists + `--replace-external` not passed

**Solution:** Add `--replace-external` flag
```bash
python scripts/regenerate_images.py \
  --out-dir data/metadata \
  --backend imagen \
  --bucket vismem-c-images-aadhi \
  --replace-external
```

**Usage:**

```bash
# Dry run (simulate without uploading)
python scripts/regenerate_images.py \
  --out-dir data/metadata \
  --backend imagen \
  --bucket vismem-c-images-aadhi \
  --dry-run

# Generate images and upload to S3
python scripts/regenerate_images.py \
  --out-dir data/metadata \
  --backend imagen \
  --bucket vismem-c-images-aadhi \
  --replace-external

# Process only events 5-14 (useful for resuming)
python scripts/regenerate_images.py \
  --out-dir data/metadata \
  --backend imagen \
  --bucket vismem-c-images-aadhi \
  --start-event 5 --num-events 10 \
  --log-level DEBUG
```

**Output:** Updated `events.json` with S3 URLs
```json
{
  "events": [
    {
      "dia_id": "D1:1",
      "text": "I'm on vacation!",
      "img": {
        "prompt": "A photo of Big Ben in London, England",
        "url": "https://vismem-c-images-aadhi.s3.amazonaws.com/vismen/D1_1.png"
      }
    }
  ]
}
```

---

### Stage 3: LoCoMo Format Assembly

**Script:** `scripts/build_vismen_c.py`

**What it does:**
- Consumes complete agent transcripts (agent_a.json, agent_b.json)
- Looks up image metadata from events.json using dia_id
- Normalizes conversation turns with image URLs
- Extracts per-session facts and observations
- Merges human-annotated QA pairs
- Produces final LoCoMo-compatible dataset

**Architecture:**

```
LoCoMoBuilder
    ├─ load_json(path) → dict
    ├─ build_event_lookup(events_path) → {dia_id → event_meta}
    ├─ normalize_turn(turn, event_meta) → standardized turn with img_url + caption
    ├─ extract_session_ids(agent) → [1, 2, 3, ...]
    ├─ convert_sessions(agent_a, event_lookup) → normalized conversation
    ├─ build_event_summary(agent_a, conversation) → per-session event facts
    └─ load_qa_entries(qa_file) → [{question, answer, category, ...}]
```

**Key Issues & Fixes:**

 **Issue: File not found**
```
python: can't open file '/Users/aadhi/Desktop/vismem-c-main/build_vismen_c.py': 
[Errno 2] No such file or directory
```

**Solution:** Use correct script path
```bash
# WRONG: python build_vismen_c.py (file not in root)
# CORRECT: python scripts/build_vismen_c.py
python scripts/build_vismen_c.py \
  --agent-a data/metadata/agent_a.json \
  --agent-b data/metadata/agent_b.json \
  --events data/metadata/events.json \
  --qa-file data/metadata/qa.json \
  --out-file data/vismen-c.json \
  --sample-id my-dataset
```

**Issue: Can't find '__main__' module in 'scripts/'**
```
python scripts/ build_vismen_c.py  # Extra space after scripts/
/Users/aadhi/miniconda3/envs/vismem-c/bin/python: 
can't find '__main__' module in '/Users/aadhi/Desktop/vismem-c-main/scripts/'
```

**Solution:** Remove space between path and script name
```bash
# WRONG: python scripts/ build_vismen_c.py
# CORRECT: python scripts/build_vismen_c.py
python scripts/build_vismen_c.py \
  --agent-a data/metadata/agent_a.json \
  --agent-b data/metadata/agent_b.json \
  --events data/metadata/events.json \
  --qa-file data/metadata/qa.json \
  --out-file data/vismen-c.json
```

⚠️ **Issue: Arguments on separate lines treated as separate commands**
```bash
python scripts/build_vismen_c.py
--agent-a data/metadata/agent_a.json    # zsh: command not found: --agent-a
--agent-b data/metadata/agent_b.json    # zsh: command not found: --agent-b
```

**Solution:** Use backslash `\` for line continuation
```bash
python scripts/build_vismen_c.py \
  --agent-a data/metadata/agent_a.json \
  --agent-b data/metadata/agent_b.json \
  --events data/metadata/events.json \
  --qa-file data/metadata/qa.json \
  --out-file data/vismen-c.json \
  --sample-id my-dataset
```

**Output:** `vismen-c.json` (LoCoMo format)
```json
[{
  "sample_id": "vismen-c-demo",
  "conversation": {
    "speaker_a": "Justin",
    "speaker_b": "AI Assistant",
    "session_1_date_time": "11:48 am on 21 May, 2022",
    "session_1": [
      {
        "speaker": "Justin",
        "dia_id": "D1:1",
        "text": "I'm on vacation again!",
        "img_url": ["https://vismem-c-images-aadhi.s3.amazonaws.com/vismen/D1_1.png"],
        "blip_caption": "A photo of Big Ben in London, England",
        "query": "A photo of Big Ben in London, England"
      }
    ]
  },
  "qa": [...],
  "observation": { "session_1_observation": {...} },
  "session_summary": { "session_1_summary": "..." },
  "event_summary": { "events_session_1": {...} }
}]
```

---

## Data Schemas

### events.json

**Input format** (from Stage 1):
```json
{
  "events": [
    {
      "dia_id": "D1:1",
      "text": "I'm on vacation! [shares photo]",
      "img": {
        "prompt": "A photo of Big Ben in London, England",
        "caption": "Big Ben clock tower",
        "description": "Historic landmark in London"
      }
    }
  ]
}
```

**Output format** (from Stage 2 - after `regenerate_images.py`):
```json
{
  "events": [
    {
      "dia_id": "D1:1",
      "text": "I'm on vacation!",
      "img": {
        "prompt": "A photo of Big Ben in London, England",
        "caption": "Big Ben clock tower",
        "url": "https://vismem-c-images-aadhi.s3.amazonaws.com/vismen/D1_1.png"
      }
    }
  ]
}
```

### agent_a.json / agent_b.json

**Schema:**
```json
{
  "name": "Justin",
  "session_1_date_time": "11:48 am on 21 May, 2022",
  "session_1": [
    {
      "dia_id": "D1:1",
      "speaker": "Justin",
      "text": "I'm on vacation again!",
      "img_url": ["https://...png"]
    },
    {
      "dia_id": "D1:2",
      "speaker": "AI Assistant",
      "text": "That sounds wonderful! Where are you?"
    }
  ],
  "session_1_facts": {
    "Justin": [["Justin is on vacation", "D1:1"]],
    "AI Assistant": [["AI responded positively", "D1:2"]]
  },
  "session_1_summary": "Justin shares vacation stories with the AI assistant..."
}
```

### qa.json

**Schema:**
```json
{
  "qa": [
    {
      "question": "How many cities has Justin visited abroad in 2024?",
      "answer": "2",
      "evidence": [],
      "category": 1
    },
    {
      "question": "Which city did Justin visit in June, 2024?",
      "answer": "Paris",
      "evidence": [],
      "category": 1
    }
  ]
}
```

---

## Setup

### Prerequisites

- **Python 3.10+** (Python 3.9 support ends April 2026)
- **Google Cloud Project** with Vertex AI API enabled
- **AWS Account** with S3 bucket created
- **Git**
- **Conda** (recommended) or venv

### Installation

**1. Clone Repository:**
```bash
git clone https://github.com/yourusername/vismem-c.git
cd vismem-c
```

**2. Create Conda Environment:**
```bash
# Using conda (recommended)
conda create -n vismem-c python=3.10
conda activate vismem-c

# OR using venv
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Key dependency fixes:**
```bash
# If using old OpenAI API in generate_conversations.py
pip install openai==0.28

# For Google Cloud
pip install google-cloud-aiplatform

# For AWS
pip install boto3

# For LLM backends
pip install openai  # Latest version
```

**4. Configure Google Cloud:**
```bash
# Authenticate with GCP
gcloud auth application-default login

# Set your project ID
export GOOGLE_CLOUD_PROJECT=your-gcp-project-id

# Verify project
gcloud config get-value project
```

**5. Configure AWS:**
```bash
# Configure AWS credentials
aws configure

# You will be prompted for:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., us-west-2)

# Verify setup
aws s3 ls
```

**6. Create S3 Bucket (if needed):**
```bash
aws s3 mb s3://vismem-c-images-aadhi --region us-west-2

# Make bucket public for image URLs
aws s3api put-bucket-policy --bucket vismem-c-images-aadhi \
  --policy '{
    "Version": "2012-10-17",
    "Statement": [{
      "Sid": "PublicRead",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::vismem-c-images-aadhi/*"
    }]
  }'
```

---

## Quick Start (End-to-End)

```bash
# 1. Generate conversations (Stage 1)
python generative_agents/generate_conversations.py \
  --out-dir data/metadata \
  --prompt-dir prompts \
  --num-sessions 30 \
  --max-turns-per-session 18

# 2. Generate images and upload to S3 (Stage 2)
python scripts/regenerate_images.py \
  --out-dir data/metadata \
  --backend imagen \
  --bucket vismem-c-images-aadhi \
  --replace-external

# 3. Build final LoCoMo dataset (Stage 3)
python scripts/build_vismen_c.py \
  --agent-a data/metadata/agent_a.json \
  --agent-b data/metadata/agent_b.json \
  --events data/metadata/events.json \
  --qa-file data/metadata/qa.json \
  --out-file data/vismen-c.json \
  --sample-id vismen-c-demo

# 4. Verify output
grep -n "s3.amazonaws.com" data/vismen-c.json | head -5
```

---

## Verification

### Check S3 URLs are in Output

```bash
# Verify events.json has S3 URLs after Stage 2
grep -c "s3.amazonaws.com" data/metadata/events.json

# Verify final dataset has S3 URLs after Stage 3
grep -c "s3.amazonaws.com" data/vismen-c.json

# Pretty print one turn to inspect structure
python -c "
import json
data = json.load(open('data/vismen-c.json'))
turn = data[0]['conversation']['session_0'][0]
print(json.dumps(turn, indent=2))
"
```

**Expected output:**
```json
{
  "speaker": "Alice",
  "dia_id": "event_0",
  "text": "Here is my receipt",
  "img_url": [
    "https://vismem-c-images-aadhi.s3.amazonaws.com/vismen/event_0_timestamp.png"
  ],
  "blip_caption": "A receipt from Apple Store",
  "query": "A receipt from Apple Store"
}
```

---

## Output Locations

| Stage | Script | Output File | Location |
|-------|--------|-------------|----------|
| 1 | `generate_conversations.py` | agent_a.json | `data/metadata/agent_a.json` |
| 1 | `generate_conversations.py` | agent_b.json | `data/metadata/agent_b.json` |
| 2 | `regenerate_images.py` | events.json (updated) | `data/metadata/events.json` |
| 2 | `regenerate_images.py` | Generated images | `data/metadata/generated_images/` |
| 2 | `regenerate_images.py` | S3 objects | `s3://bucket-name/vismen/` |
| 3 | `build_vismen_c.py` | **vismen-c.json** | **`data/vismen-c.json`** |

---

## Troubleshooting

### Image Generation Issues

**"Empty response from Imagen API"**
- Safety filter blocked content
- Solution: Simplify prompt or use `--replace-external --log-level DEBUG`
- Check GCP quota: https://console.cloud.google.com/quotas

**"Backend returned no image"**
- Image generation failed silently
- Solution: Enable DEBUG logging to see API response

**"403 Permission denied on resource project your-default-project-id"**
- Using placeholder GCP project ID
- Solution: Set actual project ID via environment variable or code

**"S3 upload failed: NoSuchBucket"**
- Bucket name incorrect or doesn't exist
- Solution: Verify bucket: `aws s3 ls`

### Credential Issues

**"Failed to initialize S3"**
- AWS credentials not configured
- Solution: Run `aws configure` and verify `~/.aws/credentials`

**"Vertex AI init failed: Credentials not found"**
- Google Cloud credentials not set
- Solution: Run `gcloud auth application-default login`

**"ModuleNotFoundError: No module named 'vertexai'"**
- Missing dependency
- Solution: `pip install google-cloud-aiplatform`

### File/Command Issues

**"File not found: [script].py"**
- Wrong script path
- Solution: Use full path like `scripts/regenerate_images.py`

**"Can't find '__main__' module in 'scripts/'"**
- Space between path and script name
- Solution: Remove extra space: `scripts/build_vismen_c.py` not `scripts/ build_vismen_c.py`

**"command not found: --agent-a"**
- Arguments on separate lines without backslash continuation
- Solution: Use backslash `\` at end of each line

**"Generation getting skipped"**
- External URLs exist and `--replace-external` not passed
- Solution: Add `--replace-external` flag

---

## Performance & Costs

### Costs

| Component | Cost | Notes |
|-----------|------|-------|
| Google Imagen | $0.02-0.10 per image | Varies by size/quality |
| AWS S3 Storage | $0.023 per GB/month | Minimal for typical datasets |
| **Total for 100 images** | ~$2-10 | Plus S3 storage |

### Performance

| Task | Time | Notes |
|------|------|-------|
| Image generation | 5-10s per image | Includes API latency |
| S3 upload | 1-2s per image | Network dependent |
| Batch processing (10 images) | ~90-120s | Parallel possible with caution |

---

## Configuration Reference

### regenerate_images.py Options

```bash
python scripts/regenerate_images.py \
  --out-dir PATH                    # Path to metadata directory (required)
  --backend NAME                    # Backend: imagen, openai (default: imagen)
  --bucket BUCKET_NAME              # S3 bucket name (required)
  --start-event INDEX               # Start from event INDEX (default: 0)
  --num-events COUNT                # Process COUNT events (default: all)
  --dry-run                         # Simulate without uploading
  --log-level LEVEL                 # DEBUG, INFO, WARNING, ERROR (default: INFO)
  --replace-external                # Replace external URLs with S3 URLs
```

### build_vismen_c.py Options

```bash
python scripts/build_vismen_c.py \
  --agent-a PATH                    # Path to agent_a.json
  --agent-b PATH                    # Path to agent_b.json
  --events PATH                     # Path to events.json
  --qa-file PATH                    # Path to qa.json
  --out-file PATH                   # Output file path (default: data/vismen-c.json)
  --sample-id ID                    # Sample identifier (default: vismen-c)
```

---

---

## License

[Add your license - MIT, Apache 2.0, etc.]

---

## Changelog

### v1.0.0 (2025-12-01)
- ✅ Initial release with full three-stage pipeline
- ✅ Google Vertex AI Imagen backend with retry logic
- ✅ AWS S3 integration with public URL generation
- ✅ LoCoMo format conversion with S3 URL injection
- ✅ Comprehensive troubleshooting guide
- ✅ Complete documentation with real-world issues and solutions
- ✅ Support for Miniconda environments
- ✅ OpenAI API v0.28 compatibility notes
- ✅ Dry-run and partial processing capabilities
