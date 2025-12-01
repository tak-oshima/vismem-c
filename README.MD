# VisMem-C: Long Conversation (with Image) Data Generation

Long-horizon agents routinely forget what they see. The original [LoCoMo benchmark](https://github.com/snap-research/locomo) captured month-long conversations between two humans, but the associated images were incidental and never grounded downstream evaluation—models were scored only on text. VisMem-C re-tools that pipeline so every planned event is paired with controlled image generation and plays out between a persona-driven user and an AI assistant, enabling true multimodal tests of whether models update and retain visual memories (including structured content such as receipts or signage) over weeks of dialogue.

This guide walks through synthesizing that data end to end. By editing `events.json`, you can script the exact story moments—and the images you expect the agents to reference—before spinning up personas, conversations, session memories, and LoCoMo-compatible outputs. You can also tailor the user persona directly in `agent_a.json` (keep `agent_b.json` as the fixed assistant). We harden the user prompts so the human persona sticks to those scripted events instead of inventing new life updates that would corrupt downstream QA.


## Quick Start

Follow these steps from the project root:

1. **Create the runtime environment** (feel free to use `requirements.txt` instead of Conda if you prefer pip).
   ```bash
   conda env create -f environment.yml
   conda activate vismem-c
   ```

2. **Export the required OpenAI key** (edit `scripts/env.sh` if you prefer sourcing).
   ```bash
   export OPENAI_API_KEY=your-openai-key
   ```
   Add any other provider keys only if you plan to run their corresponding evaluation scripts.

3. **Lay out the scripted events** that must occur in the conversation. Edit `data/metadata/events.json` (or your chosen metadata directory) to describe each planned turn—text plus image metadata. Example schema:
   ```json
   {
     "events": [
       {
         "text": "<dialogue line that should appear; include [shares ...] if an image is shown>",
         "img": {
           "id": <unique integer>,
           "url": "<link to the generated image>",
           "prompt": "<text prompt used for image generation>"
         }
       }
       // ...
     ]
   }
   ```
   The generator assigns dialogue IDs and timestamps automatically, so you only need to set the content you want guaranteed in the sessions.

4. **Generate the conversation and memory artifacts.** Adjust the flags to control persona sampling, number of sessions, summaries, etc.
   ```bash
   python generative_agents/generate_conversations.py \
     --out-dir data/metadata \
     --prompt-dir prompts \
     --persona --session --summary \
     --num-sessions 30 \
     --max-turns-per-session 18
   ```
   Key outputs inside `data/metadata`:
   - `agent_a.json` / `agent_b.json` (persona + multi-session transcript + facts)
   - `embeddings.pkl` (session-level fact embeddings)
   - `sessions.html` (HTML view of the dialogue)
   - Updated `events.json` (if provided)

5. **Convert to a LoCoMo-style sample** for downstream evaluation. The QA file (`data/metadata/qa.json`) remains human annotated; these scripts simply copy it into the generated sample.
   ```bash
   python scripts/build_vismen_c.py \
     --agent-a data/metadata/agent_a.json \
     --agent-b data/metadata/agent_b.json \
     --events data/metadata/events.json \
     --qa-file data/metadata/qa.json \
     --out-file outputs/vismen-c.json \
     --sample-id vismen-c-demo
   ```



## Attribution and License

This repository is a fork of **LoCoMo: Data and Code for the ACL 2024 paper "Evaluating Very Long-Term Conversational Memory of LLM Agents"** originally developed by Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang.

- Original repository: https://github.com/snap-research/locomo
- Original paper (preprint): arXiv:2402.17753

Unless otherwise noted, the data and code in this repository are derived from the original LoCoMo project and are made available under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license, as specified in `LICENSE.txt`.

This fork is used solely for **non-commercial academic purposes (class project)** and is not affiliated with or endorsed by Snap Inc. or the original authors.


