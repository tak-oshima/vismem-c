# VisMem-C: Complete System Documentation Summary

**Last Updated:** December 1, 2025

## What You Have

A **production-ready three-stage pipeline** for generating visual conversational datasets:

| Stage | Script | Input | Output | Purpose |
|-------|--------|-------|--------|---------|
| **1** | `generate_conversations.py` | events.json (prompts only) | agent_a.json, agent_b.json | Generate multi-turn dialogue |
| **2** | `regenerate_images.py` | events.json (prompts, no URLs) | events.json (with S3 URLs) | Generate images + S3 upload |
| **3** | `build_vismen_c.py` | agent_a/b.json + events.json + qa.json | vismen-c.json | Assemble LoCoMo format |

---

## Documentation Files Created

### For Repository Setup
- ✅ **README-SYSTEM.md** [77] - Complete system-level overview (use as main README)
- ✅ **ARCHITECTURE.md** [78] - Deep technical dive with design patterns
- ✅ **QUICKSTART.md** [75] - 5-minute setup guide
- ✅ **requirements.txt** [73] - Python dependencies
- ✅ **.gitignore** [74] - Credential protection
- ✅ **PUSH_CHECKLIST.md** [76] - Step-by-step GitHub push guide

### Using Existing Code Files
You already have these in your repo:
- `scripts/regenerate_images.py` - Image generation orchestrator
- `scripts/build_vismen_c.py` - LoCoMo format assembly
- `scripts/global_methods.py` - LLM utilities & retry logic
- `backends/base.py` - Abstract image backend
- `backends/google_imagen.py` - Google Vertex AI Imagen backend

---

## Quick Reference: How Files Fit Together

### Data Flow

```
Stage 1: Conversation Generation
└─ Input: events.json (text + image prompts, no URLs yet)
└─ Uses: global_methods.py (run_chatgpt, run_claude, etc.)
└─ Output: agent_a.json (Justin's transcript), agent_b.json (AI's transcript)

Stage 2: Image Generation & Upload (YOUR FOCUS)
└─ Input: events.json (with prompts)
└─ Uses: regenerate_images.py + backends/google_imagen.py
└─ Process: 
   ├─ Extract prompt from event
   ├─ Call backend.generate() → local image
   ├─ Upload to S3 → public URL
   └─ Update events.json with URL
└─ Output: events.json (now with S3 URLs)

Stage 3: LoCoMo Assembly
└─ Input: agent_a.json, agent_b.json, events.json (with URLs!), qa.json
└─ Uses: build_vismen_c.py
└─ Process:
   ├─ Look up event metadata for each dialogue turn
   ├─ Normalize turns (merge agent transcript + image metadata)
   ├─ Extract facts and summaries
   └─ Assemble into LoCoMo format
└─ Output: vismen-c.json (final dataset)
```

### Code Dependencies

```
regenerate_images.py
├─ backends/base.py (abstract base class)
├─ backends/google_imagen.py (Imagen implementation)
├─ AWS S3 (boto3)
└─ Google Vertex AI (vertexai)

build_vismen_c.py
├─ Reads: agent_a.json, agent_b.json
├─ Reads: events.json (expects img.url from Stage 2)
├─ Reads: qa.json (unchanged)
└─ Produces: vismen-c.json (LoCoMo format)

global_methods.py (used by Stage 1)
├─ OpenAI ChatGPT API
├─ Anthropic Claude API
├─ Google Gemini API
└─ Retry logic + JSON parsing
```

---

## Stage 2: Image Generation in Detail

### regenerate_images.py Architecture

**Main Class: ImageRegenerationPipeline**

```python
class ImageRegenerationPipeline:
    def __init__(self, out_dir, backend_name, bucket, dry_run=False)
        # Initialize S3 client
        # Initialize image backend (Imagen, DALL-E, etc.)
        # Load events.json
    
    def process_events(start_event=0, num_events=None)
        # Main loop: for each event
        #   - Extract image prompt
        #   - Generate image via backend
        #   - Upload to S3
        #   - Update events.json with URL
        # Save updated events.json
    
    def get_image_prompt(event) → str
        # Extract prompt with fallback chain:
        # 1. event['img']['prompt']
        # 2. event['img']['caption']
        # 3. event['img']['description']
        # 4. event['text']
    
    def generate_image(prompt, idx) → (filepath, time)
        # Call backend.generate()
        # Returns local file path
    
    def upload_to_s3(local_path, idx, dia_id) → S3_URL
        # Upload file to S3
        # Use semantic naming: vismen/{dia_id}.png
        # Return public HTTPS URL
    
    def print_summary()
        # Print statistics
```

**Usage:**

```bash
# Dry run (test without uploading)
python scripts/regenerate_images.py \
  --out-dir data/metadata \
  --backend imagen \
  --bucket vismem-c-images-aadhi \
  --dry-run

# Actual run
python scripts/regenerate_images.py \
  --out-dir data/metadata \
  --backend imagen \
  --bucket vismem-c-images-aadhi \
  --replace-external

# Resume from event 5
python scripts/regenerate_images.py \
  --out-dir data/metadata \
  --backend imagen \
  --bucket vismem-c-images-aadhi \
  --start-event 5 --num-events 10
```

### Backend Abstraction

**Base Class: ImageGenerator (backends/base.py)**

```python
class ImageGenerator(ABC):
    @abstractmethod
    def generate(prompt, count, output_dir) → List[Dict]
        # Returns: [{'filepath': str, 'generation_time': float}]
    
    @abstractmethod
    def get_backend_name() → str
        # Returns: "Google Imagen" or similar
```

**Concrete Implementations:**

1. **GoogleImagenBackend** (backends/google_imagen.py)
   - Model: Google Vertex AI `imagegeneration@006`
   - Features:
     - 3-level retry logic (handle safety filters)
     - Prompt enhancement for quality
     - Semantic S3 naming by dia_id
   - Setup: `gcloud auth application-default login`

2. **OpenAIBackend** (optional)
   - Model: OpenAI DALL-E 3
   - Setup: `export OPENAI_API_KEY=...`

**Factory Pattern:**

```python
# backends/__init__.py
def get_backend(backend_name, config):
    if backend_name in ['imagen', 'google_imagen']:
        return GoogleImagenBackend(config)
    elif backend_name in ['openai', 'dalle']:
        return OpenAIBackend(config)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
```

---

## Stage 3: LoCoMo Assembly in Detail

### build_vismen_c.py Architecture

**Main Functions:**

```python
def extract_session_ids(agent) → List[int]
    # Parse session_1, session_2, ... from agent JSON
    # Return: [1, 2, 3, ...]

def build_event_lookup(events_path) → Dict[dia_id → event_meta]
    # Create {dia_id → event_metadata} for O(1) lookup

def normalize_turn(turn, event_meta) → normalized_turn
    # Input turn: {speaker, dia_id, text, img_url}
    # Output: same + blip_caption, query from event_meta

def convert_sessions(agent_a, event_lookup) → (conversation, observations, summaries)
    # For each session:
    #   - Normalize all turns
    #   - Extract facts (observations)
    #   - Extract summary
    # Return: structured conversation dict

def build_event_summary(agent_a, conversation) → event_summary_dict
    # Aggregate observations per session

def load_qa_entries(qa_path) → List[Dict]
    # Load human-annotated QA pairs

def main()
    # Orchestrate all steps
    # Produce final vismen-c.json in LoCoMo format
```

**Usage:**

```bash
python scripts/build_vismen_c.py \
  --agent-a data/metadata/agent_a.json \
  --agent-b data/metadata/agent_b.json \
  --events data/metadata/events.json \
  --qa-file data/metadata/qa.json \
  --out-file outputs/vismen-c.json \
  --sample-id vismen-c-demo
```

---

## Data Schemas

### events.json

**Input (before Stage 2):**
```json
{
  "events": [
    {
      "dia_id": "D1:1",
      "text": "I'm on vacation! [shares photo]",
      "img": {
        "prompt": "A photo of Big Ben in London",
        "caption": "Big Ben clock",
        "description": "Historic landmark"
        // NO "url" field
      }
    }
  ]
}
```

**Output (after Stage 2):**
```json
{
  "events": [
    {
      "dia_id": "D1:1",
      "text": "I'm on vacation!",
      "img": {
        "prompt": "A photo of Big Ben in London",
        "url": "https://vismem-c-images-aadhi.s3.amazonaws.com/vismen/D1_1.png"
        // URL added by regenerate_images.py
      }
    }
  ]
}
```

### agent_a.json / agent_b.json

```json
{
  "name": "Justin",
  "session_1_date_time": "11:48 am on 21 May, 2022",
  "session_1": [
    {
      "speaker": "Justin",
      "dia_id": "D1:1",
      "text": "I'm on vacation!",
      "img_url": ["https://...png"]
    },
    {
      "speaker": "AI Assistant",
      "dia_id": "D1:2",
      "text": "That sounds wonderful!"
    }
  ],
  "session_1_facts": {
    "Justin": [["Justin is on vacation", "D1:1"]],
    "AI Assistant": [["AI responded positively", "D1:2"]]
  },
  "session_1_summary": "Justin shares vacation stories..."
}
```

### qa.json

```json
{
  "qa": [
    {
      "question": "How many cities has Justin visited abroad in 2024?",
      "answer": "2",
      "evidence": [],
      "category": 1
    }
  ]
}
```

### vismen-c.json (Final Output - LoCoMo Format)

```json
[{
  "sample_id": "vismen-c-demo",
  "qa": [...],
  "conversation": {
    "speaker_a": "Justin",
    "speaker_b": "AI Assistant",
    "session_1_date_time": "11:48 am on 21 May, 2022",
    "session_1": [
      {
        "speaker": "Justin",
        "dia_id": "D1:1",
        "text": "I'm on vacation!",
        "img_url": ["https://vismem-c-images-aadhi.s3.amazonaws.com/vismen/D1_1.png"],
        "blip_caption": "A photo of Big Ben in London",
        "query": "A photo of Big Ben in London"
      }
    ],
    "session_1_facts": {...},
    "session_1_summary": "..."
  },
  "observation": {...},
  "session_summary": {...},
  "event_summary": {...}
}]
```

---

## Configuration & Setup

### Environment Variables

```bash
# Google Cloud
export GCP_PROJECT_ID=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# AWS
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_DEFAULT_REGION=us-west-2

# OpenAI (if using DALL-E)
export OPENAI_API_KEY=your-key

# Anthropic (if using Claude)
export ANTHROPIC_API_KEY=your-key
```

### Installation

```bash
# 1. Clone
git clone https://github.com/yourusername/vismem-c.git
cd vismem-c

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify
python3 -c "from backends import get_backend; print('✓ OK')"
```

---

## Workflow Example

### Complete End-to-End

```bash
# Assume: events.json already exists with prompts + agent_a.json, agent_b.json created

# Stage 2: Generate images and upload to S3
python scripts/regenerate_images.py \
  --out-dir data/metadata \
  --backend imagen \
  --bucket vismem-c-images-aadhi \
  --replace-external

# (At this point: events.json now has S3 URLs)

# Stage 3: Assemble LoCoMo format
python scripts/build_vismen_c.py \
  --agent-a data/metadata/agent_a.json \
  --agent-b data/metadata/agent_b.json \
  --events data/metadata/events.json \
  --qa-file data/metadata/qa.json \
  --out-file outputs/vismen-c.json

# Verify
python3 -c "
import json
with open('outputs/vismen-c.json') as f:
    data = json.load(f)
    print(f'✓ Created: {len(data)} samples')
    print(f'✓ Sample ID: {data[0][\"sample_id\"]}')
    print(f'✓ Conversation has {len(data[0][\"conversation\"])} sessions')
"
```

---

## Key Design Decisions

### Why Three Stages?

1. **Separation of Concerns**
   - Stage 1: LLM dialogue generation (uses API calls, retry logic)
   - Stage 2: Image generation + storage (handles external APIs, networking)
   - Stage 3: Data assembly (deterministic, no external APIs)

2. **Parallelization Potential**
   - Each stage can be cached/rerun independently
   - Easy to resume if Stage 2 fails partway

3. **Testability**
   - Test each stage in isolation
   - Dry-run mode for Stage 2 without S3 costs

### Backend Abstraction

Why a pluggable backend system?

- **Multiple providers:** Google Imagen, OpenAI DALL-E, Stable Diffusion, etc.
- **Easy to add:** Just subclass `ImageGenerator` and implement `generate()`
- **No rewrites:** Core pipeline logic stays unchanged
- **Cost optimization:** Switch providers without code changes

### URL Priority in Stage 3

When normalizing dialogue turns:

1. **Priority 1:** S3 URL from events.json (generated by Stage 2)
2. **Priority 2:** External URL from agent transcripts (fallback if S3 fails)

This ensures Stage 2 updates are used when available, but doesn't break if S3 upload fails.

---

## Common Patterns in Code

### Pattern 1: Factory with Fallback

```python
# backends/__init__.py
try:
    from backends.google_imagen import GoogleImagenBackend
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False
```

**Why:** Graceful degradation if optional deps missing

### Pattern 2: Structured Logging

```python
logger.info(f"[✓] Image uploaded: {s3_url}")
logger.warning(f"[!] Skipping event (no image)")
logger.error(f"[✗] Upload failed: {error}")
```

**Why:** Easy to grep logs by status icon

### Pattern 3: Type Hints

```python
def normalize_turn(turn: Turn, event_meta: Dict[str, object]) -> Turn:
    ...
```

**Why:** Self-documenting code + IDE support

### Pattern 4: Error Handling with Context

```python
try:
    response = api_call()
except ClientError as e:
    logger.error(f"[✗] S3 upload failed: {e}")
    return None  # Graceful failure
except Exception as e:
    logger.error(f"[✗] Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    return None
```

**Why:** Specific error handling + full traceback for debugging

---

## Next Steps to Deploy

### 1. Prepare for GitHub Push
- See: PUSH_CHECKLIST.md [76]

### 2. Update README
- Replace generic README with README-SYSTEM.md [77]
- Include ARCHITECTURE.md [78] link for deep dive
- Reference QUICKSTART.md [75] for setup

### 3. Document in Code
- Add docstrings to functions (some already have them)
- Add type hints throughout
- Consider adding unit tests

### 4. Test Pipeline
```bash
# Dry-run test
python scripts/regenerate_images.py \
  --out-dir data/metadata \
  --backend imagen \
  --bucket test-bucket \
  --dry-run

# Verify events.json unchanged
git diff data/metadata/events.json
# Should show no changes
```

### 5. Production Deployment
```bash
# Full run
python scripts/regenerate_images.py \
  --out-dir data/metadata \
  --backend imagen \
  --bucket vismem-c-images-aadhi

# Assemble
python scripts/build_vismen_c.py \
  --agent-a data/metadata/agent_a.json \
  --agent-b data/metadata/agent_b.json \
  --events data/metadata/events.json \
  --qa-file data/metadata/qa.json \
  --out-file outputs/vismen-c.json

# Verify
ls -lh outputs/vismen-c.json
wc -l outputs/vismen-c.json  # Should be few lines (1 sample)
```

---

## Files Summary

### Documentation (Ready to Commit)
- ✅ README-SYSTEM.md [77] - **Use as main README**
- ✅ ARCHITECTURE.md [78] - Deep technical reference
- ✅ QUICKSTART.md [75] - Setup guide
- ✅ PUSH_CHECKLIST.md [76] - GitHub push guide
- ✅ requirements.txt [73] - Dependencies
- ✅ .gitignore [74] - Secrets protection

### Code (Already Exists in Your Repo)
- ✅ scripts/regenerate_images.py - Stage 2 orchestrator
- ✅ scripts/build_vismen_c.py - Stage 3 assembler
- ✅ scripts/global_methods.py - Utilities
- ✅ backends/base.py - Abstract base
- ✅ backends/google_imagen.py - Imagen implementation

### Sample Data (Should Exist in Your Repo)
- ✅ data/metadata/events.json - Input events
- ✅ data/metadata/agent_a.json - Agent A transcript
- ✅ data/metadata/agent_b.json - Agent B transcript
- ✅ data/metadata/qa.json - QA pairs

---

## Final Repository Structure

```
vismem-c/
├── README-SYSTEM.md              ← Main README (comprehensive system overview)
├── ARCHITECTURE.md               ← Technical deep dive
├── QUICKSTART.md                 ← 5-minute setup
├── PUSH_CHECKLIST.md             ← GitHub deployment checklist
├── requirements.txt              ← Python dependencies
├── .gitignore                    ← Secrets protection
├── LICENSE                       ← Add your choice (MIT, Apache 2.0, etc.)
│
├── scripts/
│   ├── regenerate_images.py      ← Stage 2: Image generation + S3
│   ├── build_vismen_c.py         ← Stage 3: LoCoMo assembly
│   ├── global_methods.py         ← Shared utilities
│   └── generate_conversations.py ← Stage 1 (external, not in repo)
│
├── backends/
│   ├── __init__.py               ← Factory pattern
│   ├── base.py                   ← Abstract base class
│   └── google_imagen.py          ← Google Vertex AI Imagen
│
├── data/
│   └── metadata/
│       ├── events.json           ← Input events (after Stage 1)
│       ├── agent_a.json          ← Agent A transcript (Stage 1 output)
│       ├── agent_b.json          ← Agent B transcript (Stage 1 output)
│       ├── qa.json               ← Human-annotated QA pairs
│       └── generated_images/     ← Images generated in Stage 2 (git ignored)
│
└── outputs/
    └── vismen-c.json             ← Final dataset (git ignored)
```

---

## Ready to Ship! ✅

All documentation is complete and system-ready. Follow PUSH_CHECKLIST.md [76] for GitHub deployment.

