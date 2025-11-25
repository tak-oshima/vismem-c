import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import logging
import argparse
import os, json, sys
import random
from collections import defaultdict
from copy import deepcopy
from datetime import date, timedelta, datetime
from generative_agents.conversation_utils import *
from generative_agents.html_utils import convert_to_chat_html
from generative_agents.event_utils import *
from generative_agents.memory_utils import *
from global_methods import run_chatgpt, set_openai_key

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--out-dir', required=True, type=str, help="Path to directory containing agent files and downloaded images for a conversation")
    parser.add_argument('--prompt-dir', required=True, type=str, help="Path to the dirctory containing in-context examples")
    
    parser.add_argument('--start-session', type=int, default=1, help="Start iterating from this index; first session is 1")
    parser.add_argument('--num-sessions', type=int, default=20, help="Maximum number of sessions in the conversation")
    parser.add_argument('--max-turns-per-session', type=int, default=20, help="Maximum number of total turns in each session")

    parser.add_argument('--persona', action="store_true", help="Set flag to sample a new persona from MSC and generate details")
    parser.add_argument('--session', action="store_true", help="Set flag to generate sessions based on the generated/existing personas")
    parser.add_argument('--overwrite-persona', action='store_true', help="Overwrite existing persona summaries saved in the agent files")
    parser.add_argument('--overwrite-session', action='store_true', help="Overwrite existing sessions saved in the agent files")
    parser.add_argument('--summary', action="store_true", help="Set flag to generate and use summaries in the conversation generation prompt")

    parser.add_argument('--emb-file', type=str, default='embeddings.pkl', help="Name of the file used to save embeddings for the fine-grained retrieval-based memory module")
    parser.add_argument('--reflection', action="store_true", help="Set flag to use reflection module at the end of each session and include in the conversation generation prompt for context")

    args = parser.parse_args()
    return args


def is_ai_agent(speaker):
    if not isinstance(speaker, dict):
        return False

    if speaker.get('is_ai') or speaker.get('is_ai_agent'):
        return True

    role_keys = ['role', 'type', 'speaker_type', 'persona_type']
    for key in role_keys:
        value = speaker.get(key)
        if isinstance(value, str) and value.lower() in ['assistant', 'agent', 'ai', 'bot', 'model']:
            return True

    persona_summary = speaker.get('persona_summary', '')
    lowered = persona_summary.lower()
    ai_indicators = ['language model', 'ai assistant', 'virtual assistant', 'chatbot', 'artificial intelligence']
    return any(indicator in lowered for indicator in ai_indicators)


def save_agents(agents, args):
    agent_a, agent_b = agents
    logging.info("Saving updated Agent A to %s" % args.agent_a_file)
    with open(args.agent_a_file, 'w') as f:
        json.dump(agent_a, f, indent=2)
    logging.info("Saving updated Agent B to %s" % args.agent_b_file)
    with open(args.agent_b_file, 'w') as f:
        json.dump(agent_b, f, indent=2)


def load_agents(args):

    agent_a = json.load(open(args.agent_a_file))
    agent_b = json.load(open(args.agent_b_file))
    return agent_a, agent_b


def load_events_metadata(events_path):
    if not os.path.exists(events_path):
        logging.info("No events file found at %s; skipping event prefill.", events_path)
        return None

    with open(events_path, 'r') as f:
        data = json.load(f)

    events = data.get('events', [])
    if not isinstance(events, list):
        logging.warning("Events file %s has unexpected format. Expected 'events' to be a list.", events_path)
        events = []

    for event in events:
        event.pop('session_date_time', None)
        event.pop('dia_id', None)

    return {
        "path": events_path,
        "data": data,
        "events": events,
    }


def assign_events_to_sessions(events, num_sessions, max_turns_per_session, start_session):
    sessions = list(range(start_session, num_sessions + 1))
    odd_turns = [turn for turn in range(1, max_turns_per_session + 1, 2)]

    if not odd_turns:
        raise ValueError("max_turns_per_session must be at least 1 to schedule events.")

    available_slots = [(session, turn) for session in sessions for turn in odd_turns]

    if len(events) > len(available_slots):
        raise ValueError(
            "Not enough odd-numbered turn slots to schedule all events. "
            f"{len(events)} events but only {len(available_slots)} available slots."
        )

    random.shuffle(available_slots)

    assignments = []
    for idx, _ in enumerate(events):
        session, turn = available_slots[idx]
        assignments.append({
            "index": idx,
            "session": session,
            "turn": turn,
        })

    session_map = defaultdict(list)
    for assignment in assignments:
        session_map[assignment["session"]].append((assignment["turn"], assignment["index"]))

    for session_id in session_map:
        session_map[session_id].sort(key=lambda item: item[0])

    return assignments, session_map


def build_event_turn(event, agent_name, session_id, turn_number, args):
    text = event.get("text", "") or ""
    clean_text = replace_captions(text, args) if text else ""
    img = event.get("img") or {}
    img_url = img.get("url") if isinstance(img, dict) else None

    turn = {
        "speaker": agent_name,
        "text": text,
        "clean_text": clean_text,
        "dia_id": f"D{session_id}:{turn_number}",
    }

    turn["img_url"] = [img_url] if img_url else []

    return turn


def prepare_event_prefill(args, start_session):
    events_path = os.path.join(args.out_dir, 'events.json')
    metadata = load_events_metadata(events_path)

    if not metadata:
        return None

    events = metadata["events"]
    if not events:
        logging.info("Events file %s is empty; nothing to prefill.", events_path)
        return {
            "metadata": metadata,
            "assignments": [],
            "session_map": defaultdict(list),
        }

    try:
        assignments, session_map = assign_events_to_sessions(
            events,
            args.num_sessions,
            args.max_turns_per_session,
            start_session,
        )
    except ValueError as exc:
        logging.error("Failed to assign events to sessions: %s", exc)
        raise

    return {
        "metadata": metadata,
        "assignments": assignments,
        "session_map": session_map,
    }


def get_random_time():
    start_time = timedelta(hours=9, minutes=0, seconds=0)
    end_time = timedelta(hours=21, minutes=59, seconds=59)
    random_seconds = random.randint(start_time.total_seconds(), end_time.total_seconds())
    hours = random_seconds//3600
    minutes = (random_seconds - (hours*3600))//60
    return timedelta(hours=hours, minutes=minutes, seconds=0)


def datetimeStr2Obj(dateStr):
    if 'am' in dateStr:
        datetimeObj = datetime.strptime(dateStr, "%H:%M am on %d %B, %Y")
    else:
        datetimeObj = datetime.strptime(dateStr, "%H:%M pm on %d %B, %Y")
    return datetimeObj

def datetimeObj2Str(datetimeObj):
    time_mod = 'am' if datetimeObj.hour <= 12 else 'pm'
    hour = datetimeObj.hour if datetimeObj.hour <= 12 else datetimeObj.hour-12
    min = str(datetimeObj.minute).zfill(2)
    return str(hour) + ':' + min + ' ' + time_mod + ' on ' + str(datetimeObj.day) + ' ' + datetimeObj.strftime("%B") + ', ' + str(datetimeObj.year)


def dateObj2Str(dateObj):
    return dateObj.strftime("%d") + ' ' + dateObj.strftime("%B") + ', ' + dateObj.strftime("%Y")


def get_random_date():
    # initializing dates ranges
    test_date1, test_date2 = date(2022, 1, 1), date(2023, 6, 1)
    # getting days between dates
    dates_bet = test_date2 - test_date1
    total_days = dates_bet.days
    delta_days = random.choice(range(1, total_days))
    random_date = test_date1 + timedelta(days=int(delta_days))
    return random_date


def get_session_summary(session, speaker_1, speaker_2, curr_date, previous_summary=""):
    session_query = ''
    for c in session:
        session_query += "%s: %s\n" % (c["speaker"], c["text"])
        if "image" in c:
            session_query += "[%s shares %s]\n" % (c["speaker"], c["image"])

    if previous_summary:

        query = SESSION_SUMMARY_PROMPT % (speaker_1['name'], speaker_2['name'], previous_summary, curr_date,
                                               speaker_1['name'], speaker_2['name'], session_query, speaker_1['name'], speaker_2['name'])
    else:
        query = SESSION_SUMMARY_INIT_PROMPT % (speaker_1['name'], speaker_2['name'], curr_date, session_query)

    query += '\n\n'
    # should summarize persona, previous conversations with respect to speaker.
    output = run_chatgpt(query, 1, 150, 'chatgpt')
    output = output.strip()
    return output


def get_all_session_summary(speaker, curr_sess_id):
    summary = "\n"
    for sess_id in range(1, curr_sess_id):
        sess_date = speaker['session_%s_date_time' % sess_id]
        sess_date = sess_date[2] + ' ' + sess_date[1] + ', ' + sess_date[0]
        summary += sess_date + ': ' + speaker["session_%s_summary" % sess_id] + '\n'
    return summary


def extract_session_ids(agent):
    session_ids = set()
    for key in agent.keys():
        if not key.startswith('session_'):
            continue
        suffix = key[len('session_'):]
        digits = ''
        for char in suffix:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            session_ids.add(int(digits))
    return session_ids


def sync_session_datetimes(agent_a, agent_b):
    updated = False
    session_ids = extract_session_ids(agent_a) | extract_session_ids(agent_b)
    for sess_id in session_ids:
        key = f'session_{sess_id}_date_time'
        if key in agent_a and key in agent_b:
            continue
        if key in agent_a:
            agent_b[key] = agent_a[key]
            updated = True
        elif key in agent_b:
            agent_a[key] = agent_b[key]
            updated = True
    return updated


def catch_date(date_str):
    date_format1 = '%d %B, %Y'
    date_format2 = '%d %B %Y'
    try:
        return datetime.strptime(date_str, date_format1)
    except:
        return datetime.strptime(date_str, date_format2)


def get_agent_query(speaker_1, speaker_2, curr_sess_id=0):
    summary = get_all_session_summary(speaker_1, curr_sess_id)
    speaker_is_ai = is_ai_agent(speaker_1)
    if speaker_is_ai:
        query = AGENT_CONV_PROMPT % (
            speaker_2['name'],
        )
    else:
        query = USER_CONV_PROMPT % (
            speaker_1['name'],
            speaker_1['persona_summary'],
            summary,
        )
    
    return query


def prepare_session_state(agent_a, agent_b, initial_session=None):
    if not initial_session:
        return [], 0, f"{agent_a['name']}: ", 0, False, False, {}

    turn_overrides = {}
    enriched_session = []
    for turn in initial_session:
        dia_id = turn.get('dia_id')
        if dia_id and ':' in dia_id:
            try:
                _, turn_number = dia_id.split(':')
                turn_number = int(turn_number)
                turn_overrides[turn_number] = deepcopy(turn)
                continue
            except (ValueError, TypeError):
                pass
        enriched_session.append(deepcopy(turn))

    session = enriched_session
    conv_lines = []
    break_at_next_a = False
    break_at_next_b = False

    for turn in session:
        speaker = turn.get('speaker', agent_a['name'])
        text = turn.get('text') or ''
        line = f"{speaker}: {text}"
        conv_lines.append(line)

        turn_text = (turn.get('text') or '').strip()
        if turn_text.endswith('[END]'):
            if speaker == agent_a['name']:
                break_at_next_a = True
            elif speaker == agent_b['name']:
                break_at_next_b = True

    if session:
        last_speaker = session[-1].get('speaker', agent_a['name'])
        if last_speaker == agent_a['name']:
            curr_speaker = 1
        elif last_speaker == agent_b['name']:
            curr_speaker = 0
        else:
            curr_speaker = 0
    else:
        curr_speaker = 0

    next_speaker = agent_a['name'] if curr_speaker == 0 else agent_b['name']
    conv_so_far = '\n'.join(conv_lines) + f"\n\n{next_speaker}: "

    return session, curr_speaker, conv_so_far, len(session), break_at_next_a, break_at_next_b, turn_overrides


def apply_turn_side_effects(turn, session, conv_so_far, agent_a, agent_b):
    session.append(turn)

    speaker = turn["speaker"]
    text = turn.get("text") or ""

    print("############ ", speaker, ': ', text)

    conv_so_far = conv_so_far + text + '\n'

    conv_so_far += f"\n{agent_b['name']}: " if speaker == agent_a['name'] else f"\n{agent_a['name']}: "

    break_next_a = False
    break_next_b = False
    turn_text = (turn.get('text') or '').strip()
    if turn_text.endswith('[END]'):
        if speaker == agent_a['name']:
            break_next_a = True
        else:
            break_next_b = True

    next_speaker_flag = 0 if speaker == agent_b['name'] else 1
    return conv_so_far, break_next_a, break_next_b, next_speaker_flag


def get_session(agent_a, agent_b, args, curr_sess_id=0, initial_session=None):
    # load embeddings for retrieveing relevat observations from previous conversations
    if curr_sess_id == 1:
        embeddings = None
    else:
        embeddings = pkl.load(open(args.emb_file, 'rb'))

    # always start with Agent A
    session, curr_speaker, conv_so_far, start_turn_idx, break_at_next_a, break_at_next_b, turn_overrides = prepare_session_state(agent_a, agent_b, initial_session)

    if start_turn_idx >= args.max_turns_per_session or (break_at_next_a and break_at_next_b):
        return session

    stop_dialog_count = args.max_turns_per_session if args.max_turns_per_session <= 10 else random.choice(list(range(10, args.max_turns_per_session))) # choose a random turn number to include instructions for ending the session
    for i in range(start_turn_idx, args.max_turns_per_session):

        if break_at_next_a and break_at_next_b:
            break

        current_turn_number = i + 1

        if current_turn_number in turn_overrides:
            turn = deepcopy(turn_overrides[current_turn_number])
            if 'dia_id' not in turn:
                turn['dia_id'] = f'D{curr_sess_id}:{current_turn_number}'
            conv_so_far, break_a, break_b, next_speaker_flag = apply_turn_side_effects(turn, session, conv_so_far, agent_a, agent_b)
            break_at_next_a = break_at_next_a or break_a
            break_at_next_b = break_at_next_b or break_b
            curr_speaker = next_speaker_flag
            continue

        if curr_speaker == 0:
            agent_query = get_agent_query(agent_a, agent_b, curr_sess_id=curr_sess_id)
        else:
            agent_query = get_agent_query(agent_b, agent_a, curr_sess_id=curr_sess_id)
        
        output = run_chatgpt(agent_query + conv_so_far, 1, 100, 'chatgpt', temperature=1.2)
        output = output.strip().split('\n')[0]
        output = clean_dialog(output, agent_a['name'] if curr_speaker == 0 else agent_b['name'])
        output = {"text": output}

        output["speaker"] = agent_a["name"] if curr_speaker == 0 else agent_b['name']
        text_replaced_caption = replace_captions(output["text"], args).strip()
        output["clean_text"] = text_replaced_caption if text_replaced_caption else ""
        
        output["dia_id"] = 'D%s:%s' % (curr_sess_id, i+1)
        session.append(output)

        print("############ ", agent_a['name'] if curr_speaker == 0 else agent_b['name'], ': ', output["text"])
        
        conv_so_far = conv_so_far + output["text"] + '\n'

        if output['text'].endswith('[END]'):
            if curr_speaker == 0:
                break_at_next_a = True
            else:
                break_at_next_b = True

        conv_so_far += f"\n{agent_b['name']}: " if curr_speaker == 0 else f"\n{agent_a['name']}: "
        curr_speaker = int(not curr_speaker)

    return session


def main():
    # get arguments
    args = parse_args()

    set_openai_key()

    args.emb_file = os.path.join(args.out_dir, args.emb_file)

    # create dataset directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    logging.info("Dataset directory: %s" % args.out_dir)

    args.agent_a_file = os.path.join(args.out_dir, 'agent_a.json')
    args.agent_b_file = os.path.join(args.out_dir, 'agent_b.json')

    
    # Step 1: Get personalities for the agents; get a randomly selected sample from the MSC dataset and expand the few-liner personas into detailed personas.
    if args.persona:
        agent_a, agent_b = get_msc_persona(args)
        if agent_a is not None and agent_b is not None:
            save_agents([agent_a, agent_b], args)


    # Step 2: 
    if args.session:

        agent_a, agent_b = load_agents(args)
        if sync_session_datetimes(agent_a, agent_b):
            save_agents([agent_a, agent_b], args)

        events_plan = prepare_event_prefill(args, args.start_session)
        events_metadata = None
        event_session_map = {}
        events_list = []
        if events_plan:
            events_metadata = events_plan["metadata"]
            event_session_map = events_plan["session_map"]
            events_list = events_metadata["events"]

        # default start index is 1; if resuming conversation from a leter session, indicate in script arguments using --start-session
        for j in range(args.start_session, args.num_sessions+1):

            print("******************* SESSION %s ******************" % j)

            if j>1 and ('session_%s_date_time' % (j-1)) in agent_a:
                prev_date_time_string = agent_a['session_%s_date_time' % (j-1)]
                prev_date_time = datetimeStr2Obj(prev_date_time_string)
            else:
                prev_date_time, prev_date_time_string = None, None

            curr_session_key = 'session_%s' % j
            curr_date_time_key = 'session_%s_date_time' % j

            if args.overwrite_session or curr_date_time_key not in agent_a:
                curr_time = get_random_time()
                if prev_date_time is not None:
                    curr_date = prev_date_time + timedelta(days=random.choice([1, 2]))
                else:
                    random_date = get_random_date()
                    curr_date = datetime.combine(random_date, datetime.min.time())
                curr_date_time = curr_date + curr_time

                curr_date_time_string = datetimeObj2Str(curr_date_time)
                agent_a[curr_date_time_key] = curr_date_time_string
                agent_b[curr_date_time_key] = curr_date_time_string
                save_agents([agent_a, agent_b], args)
            else:
                curr_date_time_string = agent_a[curr_date_time_key]
                curr_date_time = datetimeStr2Obj(curr_date_time_string)

            session_prefill_turns = []
            if j in event_session_map:
                for turn_number, event_index in event_session_map[j]:
                    event_obj = events_list[event_index]
                    turn_payload = build_event_turn(event_obj, agent_a['name'], j, turn_number, args)
                    session_prefill_turns.append(turn_payload)
                    event_obj["session_date_time"] = curr_date_time_string
                    event_obj["dia_id"] = turn_payload["dia_id"]

                session_prefill_turns.sort(key=lambda turn: int(turn["dia_id"].split(":")[1]))
                agent_a[curr_session_key] = session_prefill_turns
                save_agents([agent_a, agent_b], args)

            initial_session = None
            if not args.overwrite_session and curr_session_key in agent_a:
                initial_session = deepcopy(agent_a[curr_session_key])

            session = get_session(agent_a, agent_b, args,
                                  curr_sess_id=j,
                                  initial_session=initial_session)
            
            agent_a[curr_session_key] = session
            agent_b[curr_session_key] = session

            save_agents([agent_a, agent_b], args)

            if events_metadata and j in event_session_map:
                with open(events_metadata["path"], 'w') as f:
                    json.dump(events_metadata["data"], f, indent=2)

            if 'session_%s_facts' % j not in agent_a or args.overwrite_session:

                facts = get_session_facts(args, agent_a, agent_b, j)

                agent_a['session_%s_facts' % j] = facts
                agent_b['session_%s_facts' % j] = facts

                print(" --------- Session %s Summary for Agent A---------" % (j))
                print(facts)

                save_agents([agent_a, agent_b], args)

            if args.reflection and ('session_%s_reflection' % j not in agent_a or args.overwrite_session):

                reflections = get_session_reflection(args, agent_a, agent_b, j)

                agent_a['session_%s_reflection' % j] = reflections['a']
                agent_b['session_%s_reflection' % j] = reflections['b']

                print(" --------- Session %s Reflection for Agent A---------" % (j))
                print(reflections)

                save_agents([agent_a, agent_b], args)

            if args.summary and ('session_%s_summary' % j not in agent_a or args.overwrite_session):

                summary = get_session_summary(agent_a['session_%s' % j], agent_a, agent_b, agent_a['session_%s_date_time' % j], 
                                              previous_summary=None if j==1 else agent_a['session_%s_summary' % (j-1)])

                agent_a['session_%s_summary' % j] = summary
                agent_b['session_%s_summary' % j] = summary

                save_agents([agent_a, agent_b], args)

        if events_metadata:
            with open(events_metadata["path"], 'w') as f:
                json.dump(events_metadata["data"], f, indent=2)

    agent_a, agent_b = load_agents(args)
    convert_to_chat_html(agent_a, agent_b, outfile=os.path.join(args.out_dir, 'sessions.html'), use_events=False, img_dir=args.out_dir)


if __name__ == "__main__":
    main()