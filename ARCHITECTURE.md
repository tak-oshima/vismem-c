# VisMem-C Architecture Deep Dive

Detailed technical architecture and design rationale for the three-stage VisMem-C pipeline.

## System Components Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VisMem-C System                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Stage 1: Conversation Generation (generate_conversations.py)      │
│  └─ Input: events.json (text + prompts, no URLs)                   │
│  └─ Output: agent_a.json, agent_b.json (multi-turn transcripts)    │
│  └─ Uses: global_methods.py (LLM APIs, retry logic)                │
│                                                                     │
│  Stage 2: Image Generation & S3 Upload (regenerate_images.py)      │
│  └─ Input: events.json (with prompts, no URLs)                     │
│  └─ Output: events.json (updated with S3 URLs)                     │
│  └─ Uses: backends/ (pluggable image providers)                    │
│     ├─ GoogleImagenBackend (Vertex AI Imagen @006)                 │
│     └─ OpenAIBackend (DALL-E 3)                                    │
│                                                                     │
│  Stage 3: LoCoMo Format Assembly (build_vismen_c.py)               │
│  └─ Input: agent_a.json, agent_b.json, events.json, qa.json        │
│  └─ Output: vismen-c.json (LoCoMo-compatible dataset)              │
│  └─ Function: Normalize, merge, and assemble final dataset         │
│                                                                     │
│  Shared: Utilities (global_methods.py)                             │
│  └─ LLM API clients with retry/backoff                             │
│  └─ Embedding generation                                           │
│  └─ Error handling and JSON parsing                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Conversation Generation

**Script:** `generate_conversations.py` (external - not in this repo)

### Input Data Model

```python
# events.json
{
  "events": [
    {
      "dia_id": "D1:1",                   # Unique dialogue turn ID
      "text": "I'm on vacation!",         # Speaker's text
      "img": {
        "prompt": "A photo of Big Ben",   # Image generation prompt
        "caption": "Big Ben clock",       # Display caption
        "description": "Historic London landmark"
      }
      # Note: NO 'url' field yet (populated in Stage 2)
    }
  ]
}
```

### Generation Process

```
events.json
    ↓
global_methods.run_chatgpt()  / run_claude() / run_gemini()
    ├─ Prompt engineering: "Generate a multi-turn conversation about..."
    ├─ JSON parsing with retry (up to 10x on parse errors)
    ├─ Backoff logic: exponential delay on rate limits
    └─ Response validation
    ↓
agent_a.json, agent_b.json
    ├─ session_1, session_2, ... (multi-session structure)
    ├─ Each turn has dia_id (references events.json)
    ├─ Text, speaker, optional img_url (still external)
    └─ Per-session facts and summaries
```

### global_methods.py Utility Functions

**API Setup:**
```python
set_openai_key()       # Load OPENAI_API_KEY from env
set_gemini_key()       # Load GOOGLE_API_KEY from env
set_anthropic_key()    # Load ANTHROPIC_API_KEY from env
```

**LLM Calls with Retry:**
```python
run_json_trials(
    prompt="...",
    model="gpt-4",
    max_retries=10,      # Try up to 10 times on JSON parse errors
    temperature=0.7,
    max_tokens=2000
)
# Returns: parsed dict (automatically extracts JSON from response)
```

**Embeddings:**
```python
embedding = get_openai_embedding("text to embed")
# Returns: 1536-dim vector (ada-002 model)
```

**Current Issues (Code Quality Notes):**
- Uses deprecated `openai.ChatCompletion.create()` (should use `client.chat.completions.create()`)
- No type hints on function signatures
- Broad `except Exception` handling (should be more specific)
- Hardcoded model names like "gpt-3.5-turbo" (should be parameterized)

---

## Stage 2: Image Generation & S3 Upload

**Script:** `regenerate_images.py`

### Architecture Overview

```
ImageRegenerationPipeline (main orchestrator)
│
├─ __init__(out_dir, backend_name, bucket, dry_run, replace_external)
│  ├─ Initialize S3 client (boto3)
│  ├─ Initialize image backend via get_backend()
│  ├─ Validate events.json exists
│  └─ Initialize statistics tracker
│
├─ process_events(start_event=0, num_events=None)
│  └─ Main loop:
│     for event in events[start_event:end_event]:
│       1. Check if URL exists (skip if S3 or external + no --replace-external)
│       2. Extract prompt via get_image_prompt()
│       3. Generate image via generate_image()
│       4. Upload to S3 via upload_to_s3()
│       5. Update event['img']['url'] in memory
│       6. Save events.json after loop
│
├─ get_image_prompt(event)
│  └─ Fallback chain:
│     1. Try event['img']['prompt']
│     2. Try event['img']['caption']
│     3. Try event['img']['description']
│     4. Try event['text'] (full text as prompt)
│     5. Return None if all fail
│
├─ generate_image(prompt, event_idx)
│  └─ Call backend.generate():
│     - Passes prompt, count=1, output_dir
│     - Backend handles retry logic (for Imagen safety filters)
│     - Returns: [{'filepath': local_path, 'generation_time': seconds}]
│     - On error: logs exception, returns None
│
├─ upload_to_s3(local_path, event_idx, dia_id)
│  └─ S3 naming strategy:
│     if dia_id: use f"vismen/{safe_dia_id}.png"  (semantic naming)
│     else: use f"vismen/event_{event_idx}_{timestamp}.png"  (fallback)
│  └─ ExtraArgs: {'ContentType': 'image/png'}
│  └─ Returns: public HTTPS URL
│
└─ print_summary()
   └─ Print statistics: total, generated, uploaded, skipped, errors
```

### Backend Abstraction

**Base Class (ImageGenerator):**
```python
from abc import ABC, abstractmethod

class ImageGenerator(ABC):
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        count: int = 1, 
        output_dir: str = 'outputs/images',
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate images from prompt.
        
        Args:
            prompt: Text description
            count: Number of images to generate
            output_dir: Local directory to save files
            **kwargs: Provider-specific parameters
            
        Returns:
            List of dicts: [{'filepath': str, 'generation_time': float, ...}]
        """
        pass
    
    @abstractmethod
    def get_backend_name(self) -> str:
        """Return provider name for logging."""
        pass
```

**Factory Pattern (get_backend):**
```python
def get_backend(backend_name: str, config: dict):
    backend_name = backend_name.lower()
    
    if backend_name in ['imagen', 'google_imagen']:
        from backends.google_imagen import GoogleImagenBackend
        return GoogleImagenBackend(config)
    
    elif backend_name in ['openai', 'dalle']:
        from backends.openai_backend import OpenAIBackend
        return OpenAIBackend(config)
    
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
```

### GoogleImagenBackend Implementation

**Initialization:**
```python
class GoogleImagenBackend(ImageGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Read credentials from env or file
        project_id = os.getenv("GCP_PROJECT_ID")
        location = "us-central1"
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        # Load model
        self.model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        
        # Configuration
        self.enhance_prompt = config.get('enhance_prompt', True)
```

**Retry Logic:**
```python
def generate(self, prompt, count, output_dir, **kwargs):
    """Generate with 3-level retry strategy."""
    
    # Variant 1: Original prompt, less strict safety
    response = self.model.generate_images(
        prompt=prompt,
        number_of_images=count,
        safety_filter_level="block_some"
    )
    if response.images:
        return self._save_images(response, output_dir)
    
    # Variant 2: Simplified prompt, less strict safety
    simplified = self.simplify_prompt(prompt)
    response = self.model.generate_images(
        prompt=simplified,
        safety_filter_level="block_some"
    )
    if response.images:
        return self._save_images(response, output_dir)
    
    # Variant 3: Simplified, no safety filter
    response = self.model.generate_images(
        prompt=simplified,
        safety_filter_level="block_none"
    )
    if response.images:
        return self._save_images(response, output_dir)
    
    # All variants failed
    logger.error(f"All generation attempts failed")
    return []
```

**Prompt Enhancement:**
```python
def enhance_prompt_text(self, prompt):
    """Add quality keywords to prompt."""
    enhancements = {
        'receipt': "professional product photography, clear text",
        'cafe': "professional architectural photography, warm ambiance",
        'food': "food photography, appetizing, professional lighting",
        # ... more keywords ...
    }
    
    for keyword, enhancement in enhancements.items():
        if keyword in prompt.lower():
            return f"{prompt}, {enhancement}"
    
    return f"{prompt}, high quality professional photography"
```

### Data Flow Diagram

```
events.json (input)
    ↓ [for each event with img.prompt]
get_image_prompt()
    ↓
backend.generate()
    ├─ Try Imagen (block_some)
    ├─ If fail: Try simplified (block_some)
    └─ If fail: Try simplified (block_none)
    ↓ [returns local filepath]
s3_client.upload_file()
    ↓
S3: https://bucket.s3.amazonaws.com/vismen/{dia_id}.png
    ↓
event['img']['url'] = S3_URL
    ↓
save events.json
    ↓
events.json (output with S3 URLs)
```

### CLI Options & Behavior

```bash
python regenerate_images.py \
  --out-dir data/metadata              # Required
  --backend imagen                      # Required (default: imagen)
  --bucket vismem-c-images-aadhi        # Required
  --start-event 0                       # Optional (default: 0)
  --num-events 3                        # Optional (default: all)
  --dry-run                             # Optional (no uploads)
  --log-level DEBUG                     # Optional (default: INFO)
  --replace-external                    # Optional (override external URLs)
```

**Dry-run behavior:**
```python
if self.dry_run:
    # Generate images locally
    # Print what WOULD be uploaded
    # Don't upload to S3
    # Don't modify events.json
```

**Resume behavior:**
```bash
# Process events 0-2
python regenerate_images.py --start-event 0 --num-events 3

# Resume from event 3 (if previous run halted)
python regenerate_images.py --start-event 3 --num-events N
```

---

## Stage 3: LoCoMo Format Assembly

**Script:** `build_vismen_c.py`

### Architecture

```
LoCoMoBuilder (orchestrator)
│
├─ __init__(agent_a, agent_b, events_path, qa_path)
│  ├─ Load agent_a.json
│  ├─ Load agent_b.json
│  ├─ Build event_lookup: {dia_id → event_metadata}
│  └─ Load qa.json entries
│
├─ extract_session_ids(agent)
│  └─ Parse agent dict for session_1, session_2, etc.
│  └─ Return sorted list of session numbers
│
├─ build_event_lookup(events_path)
│  └─ Create {dia_id → event_metadata} dict
│  └─ O(1) lookup during normalization
│
├─ normalize_turn(turn, event_meta)
│  └─ Input turn structure:
│     {
│       "dia_id": "D1:1",
│       "speaker": "Justin",
│       "text": "I'm on vacation!",
│       "img_url": ["https://external.com/image.png"]  ← OLD URL
│     }
│  └─ Output normalized turn:
│     {
│       "speaker": "Justin",
│       "dia_id": "D1:1",
│       "text": "I'm on vacation!",
│       "img_url": ["https://s3.amazonaws.com/vismen/D1_1.png"],  ← S3 URL
│       "blip_caption": "A photo of Big Ben",
│       "query": "A photo of Big Ben"
│     }
│  └─ Strategy:
│     1. Check event_meta for S3 URL → use it (priority)
│     2. If no S3 URL: fall back to original img_url
│     3. Extract caption/query from event_meta
│
├─ convert_sessions(agent_a, event_lookup)
│  └─ For each session_N in agent_a:
│     ├─ Normalize all turns (update URLs)
│     ├─ Extract session facts
│     ├─ Extract session summary
│     └─ Build normalized session structure
│
├─ build_event_summary(agent_a, conversation)
│  └─ Aggregate per-session observations
│  └─ Create {events_session_1: {...}, events_session_2: {...}}
│
└─ main()
   └─ Orchestrate all steps
   └─ Produce final vismen-c.json
```

### Schema Transformation

**Input Schemas:**

agent_a.json:
```json
{
  "name": "Justin",
  "session_1": [
    {
      "dia_id": "D1:1",
      "speaker": "Justin",
      "text": "I'm on vacation!",
      "img_url": ["https://old.com/image.png"]
    }
  ],
  "session_1_facts": {
    "Justin": [["Justin is on vacation", "D1:1"]]
  }
}
```

events.json (after Stage 2):
```json
{
  "events": [
    {
      "dia_id": "D1:1",
      "img": {
        "prompt": "A photo of Big Ben",
        "url": "https://s3.amazonaws.com/vismen/D1_1.png"
      }
    }
  ]
}
```

**Output Schema:**

vismen-c.json (LoCoMo):
```json
[{
  "sample_id": "vismen-c-demo",
  "conversation": {
    "speaker_a": "Justin",
    "speaker_b": "AI Assistant",
    "session_1_date_time": "...",
    "session_1": [
      {
        "speaker": "Justin",
        "dia_id": "D1:1",
        "text": "I'm on vacation!",
        "img_url": ["https://s3.amazonaws.com/vismen/D1_1.png"],  ← Updated
        "blip_caption": "A photo of Big Ben",
        "query": "A photo of Big Ben"
      }
    ],
    "session_1_facts": {...},
    "session_1_summary": "..."
  },
  "qa": [...],
  "observation": {...},
  "session_summary": {...},
  "event_summary": {...}
}]
```

### URL Priority Logic

```python
def normalize_turn(turn, event_meta):
    normalized = {
        "speaker": turn.get("speaker"),
        "dia_id": turn.get("dia_id"),
        "text": turn.get("text", "")
    }
    
    # PRIORITY 1: Use S3 URL from events.json
    if event_meta:
        img_info = event_meta.get("img", {}) or {}
        s3_url = img_info.get("url")
        if s3_url:
            normalized["img_url"] = [s3_url]  ← S3 (preferred)
            prompt = img_info.get("prompt")
            if prompt:
                normalized["blip_caption"] = prompt
                normalized["query"] = prompt
            return normalized
    
    # PRIORITY 2: Fall back to original img_url
    images = turn.get("img_url")
    if images:
        if isinstance(images, str):
            images = [images]
        normalized["img_url"] = images  ← External or old URL
    
    return normalized
```

### CLI Options

```bash
python build_vismen_c.py \
  --agent-a data/metadata/agent_a.json       # Required
  --agent-b data/metadata/agent_b.json       # Required
  --events data/metadata/events.json         # Required
  --qa-file data/metadata/qa.json            # Required
  --out-file outputs/vismen-c.json           # Optional
  --sample-id vismen-c-demo                  # Optional
```

---

## Data Models & Schemas

### events.json Full Schema

**Before Stage 2:**
```json
{
  "events": [
    {
      "dia_id": "D1:1",
      "text": "I'm on vacation! [shares photo]",
      "img": {
        "id": 1,
        "prompt": "A photo of Big Ben in London, England",
        "caption": "Big Ben clock tower",
        "description": "Historic landmark in London"
        // No "url" field yet
      }
    },
    {
      "dia_id": "D2:9",
      "text": "Just took this photo on my vacation!",
      "img": {
        "prompt": "A photo of the Eiffel Tower in Paris, France",
        "caption": "Eiffel Tower",
        "description": "Iconic Paris landmark"
      }
    }
  ]
}
```

**After Stage 2:**
```json
{
  "events": [
    {
      "dia_id": "D1:1",
      "text": "I'm on vacation!",
      "img": {
        "id": 1,
        "prompt": "A photo of Big Ben in London, England",
        "caption": "Big Ben clock tower",
        "url": "https://vismem-c-images-aadhi.s3.amazonaws.com/vismen/D1_1.png"
        // ↑ Added after Stage 2
      }
    }
  ]
}
```

### agent_a.json Full Schema

```json
{
  "name": "Justin",
  "session_1_date_time": "11:48 am on 21 May, 2022",
  "session_1": [
    {
      "speaker": "Justin",
      "dia_id": "D1:1",
      "text": "I'm on vacation again!",
      "img_url": ["https://vismem-c-images-aadhi.s3.amazonaws.com/vismen/D1_1.png"]
    },
    {
      "speaker": "AI assistant",
      "dia_id": "D1:2",
      "text": "That sounds wonderful!"
    }
  ],
  "session_1_facts": {
    "Justin": [
      ["Justin is on vacation.", "D1:1"],
      ["Justin took a photo of Big Ben.", "D1:1"]
    ],
    "AI assistant": [
      ["AI responded positively to Justin's vacation.", "D1:2"]
    ]
  },
  "session_1_summary": "Justin shares vacation photos and stories with the AI assistant...",
  "session_2_date_time": "3:33 am on 24 May, 2022",
  "session_2": [...],
  "session_2_facts": {...},
  "session_2_summary": "..."
}
```

### qa.json Schema

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
      "evidence": ["D2:9"],
      "category": 1,
      "difficulty": "easy"
    }
  ]
}
```

---

## Error Handling & Logging

### Logging Levels

| Level | Use Case | Examples |
|-------|----------|----------|
| DEBUG | Low-level diagnostics | API responses, retry attempts, file I/O details |
| INFO | Normal operation | Backend initialized, image generated, file saved |
| WARNING | Unexpected but recoverable | URL already exists, safety filter relaxed |
| ERROR | Failures requiring attention | Generation failed, S3 upload failed, file not found |

**Example output:**
```
[2025-12-01 10:52:12,802] INFO: [✓] Backend 'imagen' initialized
[2025-12-01 10:52:12,803] INFO: [*] Processing events 0–2 (total: 3)
[2025-12-01 10:52:12,804] INFO: [1] Generating: A photo of Big Ben...
[2025-12-01 10:52:20,743] ERROR: [✗] Backend returned no image for event 1
[2025-12-01 10:52:20,746] INFO: [✓] Saved: data/metadata/events.json
```

### Exception Handling Strategy

**Generate stage:**
```python
try:
    response = self.model.generate_images(...)
except Exception as e:
    logger.error(f"[✗] Generation failed: {e}")
    logger.error(f"[DEBUG] Exception type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    return None
```

**Upload stage:**
```python
try:
    self.s3_client.upload_file(...)
except ClientError as e:
    logger.error(f"[✗] S3 upload failed: {e}")
    return None
except Exception as e:
    logger.error(f"[✗] Unexpected error: {e}")
    traceback.print_exc()
    return None
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Load events.json | O(n) | n = number of events |
| Build event_lookup | O(n) | Single pass, dict insertion O(1) |
| Process N events | O(n * t) | t = time per image (5-10s) |
| Build LoCoMo | O(m) | m = total turns across sessions |
| S3 upload | O(1) per file | Network I/O bound |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| events dict | O(n) | n events + metadata |
| event_lookup | O(n) | Same size as events dict |
| Generated images (disk) | O(n * size) | size ~500KB-2MB per image |
| Final vismen-c.json | O(m) | m = total turns |

### Throughput

**Single-threaded (sequential):**
- 10 images: ~120-180 seconds (accounting for API latency)
- 100 images: ~20-30 minutes
- 1000 images: ~3-5 hours

**With parallel processing (caution):**
- Respect GCP Imagen API quotas (typically 1-5 req/min)
- Respect AWS S3 rate limits (typically 3,500 PUT/s per partition)
- Can spawn multiple processes with `--start-event N --num-events M`

---

## Design Patterns & Best Practices

### Pattern 1: Backend Abstraction (Strategy Pattern)

**Why:** Multiple image providers (Imagen, DALL-E, Stable Diffusion) have different APIs

**How:**
```python
# Abstract base
class ImageGenerator(ABC):
    @abstractmethod
    def generate(...): pass

# Concrete implementations
class GoogleImagenBackend(ImageGenerator): ...
class OpenAIBackend(ImageGenerator): ...

# Factory
def get_backend(name, config):
    if name == 'imagen': return GoogleImagenBackend(config)
    if name == 'openai': return OpenAIBackend(config)
```

### Pattern 2: Retry with Backoff (Resilience Pattern)

**Why:** APIs may fail temporarily (rate limits, transient errors)

**How:**
```python
for attempt in range(max_retries):
    try:
        response = api_call()
        if response.success:
            return response
    except RateLimitError:
        wait_time = 2 ** attempt  # Exponential backoff
        sleep(wait_time)
    except Exception:
        if attempt == max_retries - 1:
            raise
```

### Pattern 3: Factory Pattern (Creational Pattern)

**Why:** Decouple caller from concrete implementation

**How:**
```python
# Instead of:
if backend_name == 'imagen':
    backend = GoogleImagenBackend()
elif backend_name == 'openai':
    backend = OpenAIBackend()

# Use:
backend = get_backend(backend_name, config)
```

### Pattern 4: Dictionary Lookup (Semantic Naming)

**Why:** Easily locate images by dialogue ID or event index

**How:**
```python
# Build lookup
event_lookup = {event['dia_id']: event for event in events}

# Use for O(1) access
event_meta = event_lookup.get('D1:1')
```

---

## Configuration Reference

### Environment Variables

```bash
# Google Cloud
export GCP_PROJECT_ID=your-gcp-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# AWS
# OR use ~/.aws/credentials
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-west-2

# OpenAI (if using DALL-E backend)
export OPENAI_API_KEY=...

# Anthropic (if using Claude)
export ANTHROPIC_API_KEY=...
```

### File Permissions

```bash
# Scripts should be executable
chmod +x scripts/*.py

# Verify
ls -la scripts/
# Should show:
# -rwxr-xr-x  ... scripts/regenerate_images.py
# -rwxr-xr-x  ... scripts/build_vismen_c.py
```

---

## Future Enhancements

1. **Caching:** Store generated image metadata to avoid re-processing
2. **Parallel Processing:** Process multiple events simultaneously (respecting API quotas)
3. **Incremental Builds:** Update only new/modified events
4. **Alternative Backends:** Stable Diffusion, Midjourney, Azure Computer Vision
5. **Image Validation:** Check quality/safety before uploading
6. **Cost Tracking:** Monitor API spending in real-time
7. **Dataset Versioning:** Track dataset versions and changes
8. **Batch Operations:** Process multiple datasets in sequence

---

## Testing Strategy

### Unit Tests

```python
# Test event prompt extraction
def test_get_image_prompt():
    event = {"img": {"prompt": "test"}}
    assert pipeline.get_image_prompt(event) == "test"

# Test fallback chain
event = {"img": {}, "text": "fallback"}
assert pipeline.get_image_prompt(event) == "fallback"
```

### Integration Tests

```python
# Test full pipeline (dry-run mode)
pipeline = ImageRegenerationPipeline(..., dry_run=True)
pipeline.process_events(num_events=1)
assert pipeline.stats['images_generated'] == 1

# Test S3 naming
url = pipeline.upload_to_s3("image.png", 0, "D1:1")
assert "D1_1.png" in url
```

### E2E Tests

```bash
# Full dry-run test
python regenerate_images.py ... --dry-run

# Verify events.json not modified
git diff data/metadata/events.json
# Should show no changes
```

