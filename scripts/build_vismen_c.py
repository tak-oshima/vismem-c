#!/usr/bin/env python3
"""Convert generated agent transcripts into a LoCoMo-style sample."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


Turn = Dict[str, object]
FactEntry = Sequence[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-a",
        default="data/metadata/agent_a.json",
        help="Path to agent_a.json produced by generate_conversations.py",
    )
    parser.add_argument(
        "--agent-b",
        default="data/metadata/agent_b.json",
        help="Path to agent_b.json produced by generate_conversations.py",
    )
    parser.add_argument(
        "--events",
        default="data/metadata/events.json",
        help="Optional events.json file used during conversation generation.",
    )
    parser.add_argument(
        "--qa-file",
        default="data/metadata/qa.json",
        help="Path to a qa.json file whose `qa` entries should be copied verbatim.",
    )
    parser.add_argument(
        "--out-file",
        default="data/vismen-c.json",
        help="Destination LoCoMo-like JSON file.",
    )
    parser.add_argument(
        "--sample-id",
        default="vismen-c",
        help="Identifier stored under the sample_id field.",
    )
    return parser.parse_args()


def extract_session_ids(agent: Dict[str, object]) -> List[int]:
    session_ids = set()
    for key in agent:
        if not key.startswith("session_"):
            continue
        suffix = key[len("session_") :]
        digits = ""
        for char in suffix:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            session_ids.add(int(digits))
    return sorted(session_ids)


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r") as handle:
        return json.load(handle)


def build_event_lookup(events_path: str) -> Dict[str, Dict[str, object]]:
    if not os.path.exists(events_path):
        return {}
    events_blob = load_json(events_path)
    events = events_blob.get("events", [])
    return {entry.get("dia_id"): entry for entry in events if entry.get("dia_id")}


def normalize_turn(turn: Turn, event_meta: Dict[str, object]) -> Turn:
    normalized = {
        "speaker": turn.get("speaker"),
        "dia_id": turn.get("dia_id"),
        "text": turn.get("text", ""),
    }

    images = turn.get("img_url")
    if images:
        if isinstance(images, str):
            images = [images]
        normalized["img_url"] = images
        if event_meta:
            img_info = event_meta.get("img", {}) or {}
            prompt = img_info.get("prompt")
            if prompt:
                normalized["blip_caption"] = prompt
                normalized["query"] = prompt
    return normalized


def convert_sessions(
    agent_a: Dict[str, object],
    event_lookup: Dict[str, Dict[str, object]],
) -> Tuple[Dict[str, object], Dict[str, Dict[str, List[FactEntry]]], Dict[str, str]]:
    session_ids = extract_session_ids(agent_a)
    speaker_a = agent_a.get("name", "Speaker A")
    conversation = {
        "speaker_a": speaker_a,
        "speaker_b": None,
    }
    observations: Dict[str, Dict[str, List[FactEntry]]] = {}
    summaries: Dict[str, str] = {}

    for session_id in session_ids:
        date_key = f"session_{session_id}_date_time"
        turns_key = f"session_{session_id}"
        facts_key = f"session_{session_id}_facts"
        summary_key = f"session_{session_id}_summary"

        date_value = agent_a.get(date_key)
        turns = agent_a.get(turns_key, [])
        conversation[date_key] = date_value

        normalized_turns = []
        for turn in turns:
            event_meta = event_lookup.get(turn.get("dia_id"))
            normalized_turns.append(normalize_turn(turn, event_meta))
        conversation[turns_key] = normalized_turns

        facts = agent_a.get(facts_key, {})
        obs_key = f"session_{session_id}_observation"
        obs_payload: Dict[str, List[FactEntry]] = {}
        for speaker, entries in facts.items():
            cleaned_entries = []
            for entry in entries:
                if isinstance(entry, Sequence) and len(entry) >= 2:
                    cleaned_entries.append([entry[0], entry[1]])
            obs_payload[speaker] = cleaned_entries
        observations[obs_key] = obs_payload

        summaries[summary_key] = agent_a.get(summary_key, "")

    return conversation, observations, summaries


def build_event_summary(
    agent_a: Dict[str, object],
    conversation: Dict[str, object],
) -> Dict[str, Dict[str, object]]:
    session_ids = extract_session_ids(agent_a)
    speaker_a = agent_a.get("name", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")
    summary = {}
    for session_id in session_ids:
        obs_key = f"session_{session_id}_facts"
        date_key = f"session_{session_id}_date_time"
        events_key = f"events_session_{session_id}"
        facts = agent_a.get(obs_key, {})
        speaker_a_events = [fact for fact, _ in facts.get(speaker_a, [])]
        speaker_b_events = [fact for fact, _ in facts.get(speaker_b, [])]
        summary[events_key] = {
            speaker_a: speaker_a_events,
            speaker_b: speaker_b_events,
            "date": agent_a.get(date_key, ""),
        }
    return summary


def load_qa_entries(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"QA file not found: {path}")
    data = load_json(path)
    if isinstance(data, dict) and "qa" in data:
        return data["qa"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized QA schema in {path}")


def main() -> None:
    args = parse_args()
    agent_a = load_json(args.agent_a)
    agent_b = load_json(args.agent_b)
    events = build_event_lookup(args.events)

    conversation, observations, summaries = convert_sessions(agent_a, events)
    conversation["speaker_b"] = agent_b.get("name", "Speaker B")

    event_summary = build_event_summary(agent_a, conversation)
    qa_entries = load_qa_entries(args.qa_file)

    sample = {
        "sample_id": args.sample_id,
        "qa": qa_entries,
        "conversation": conversation,
        "observation": observations,
        "session_summary": summaries,
        "event_summary": event_summary,
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump([sample], handle, indent=2)


if __name__ == "__main__":
    main()

