source scripts/env.sh

python3 generative_agents/generate_conversations.py \
    --out-dir ./data/metadata/ \
    --prompt-dir ./prompts \
    --session --summary --num-sessions 50 \
    --persona \
    --max-turns-per-session 20
