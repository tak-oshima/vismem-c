source scripts/env.sh

python3 generative_agents/generate_conversations.py \
    --out-dir ./data/multimodal_dialog/example/ \
    --prompt-dir ./prompt_examples \
    --session --summary --num-sessions 3 \
    --persona \
    --max-turns-per-session 20
