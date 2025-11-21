import json, re, os
import random
from global_methods import run_chatgpt, run_chatgpt_with_examples

PERSONA_FROM_MSC_PROMPT = (
    "Let's write speaker descriptions from a given set of life attributes. Example:\n\n%s\n\n"
    "Note: Add crucial details in the persona about the person such as their name, age, marital status, "
    "gender, job etc. Add additional details like names of family/friends or specific activities, "
    "likes and dislikes, experiences when appropriate.\n\n"
    "For the following attributes, write a persona. Output a json file with the keys 'persona' and 'name'.\n\n"
    "%s\n\n"
    "Start your answer with a curly bracket.\n"
)


USER_CONV_PROMPT_SESS_1 = (
    "%s\n\n"
    "Today is %s. You are %s chatting with an AI assistant for the first time. "
    "Treat the assistant like your go-to search engine: ask for explanations, recommendations, "
    "comparisons, or step-by-step guides about topics already connected to your life story. "
    "You can clarify details you've mentioned before, but do not invent new trips, purchases, "
    "or major events; instead, frame follow-up questions about existing details. "
    "Keep your message under 25 words and do not address the assistant by name.\n\n"
    "CONVERSATION:\n\n"
)

AGENT_CONV_PROMPT_SESS_1 = (
    "%s\n\n"
    "Today is %s. Assume the role of %s and write the next thing you would say to %s in the conversation. "
    "Greet them warmly, show curiosity about their life, offer help or information, "
    "reference relevant details from your persona, avoid inventing human-only experiences, "
    "and keep responses under 25 words while sounding natural.\n\n"
    "CONVERSATION:\n\n"
)

USER_CONV_PROMPT = (
    "%s\n\n"
    "%s last talked to the AI assistant at %s. %s\n\n"
    "Today is %s. You are %s continuing this conversation with the AI assistant. "
    "Ask follow-up questions about ongoing goals, clarify past discussions, or request specific "
    "recommendations related to what is already known. Do not introduce new life events, travel, "
    "purchases, or major changes—stick to existing topics. "
    "Keep replies under 25 words and do not address the assistant by name.\n\n"
    "CONVERSATION:\n\n"
)

#########################################################################

AGENT_CONV_PROMPT_V2 = """
You are an AI assistant chatting with a user named %s.
Write the next thing you would say to the user in the given CONVERSATION.
- Write in less than 20 words.
- Always comply with the user's request.
- Offer help or information.
- When the user shares a photo in the conversation in the format "[shares a photo of...]", you must avoid mentioning any of the words inside the square brackets that describe the photo. For example, if the user message contains `[shares a photo of a cat]`, you must not mention the word `cat`.
- Do not ask the user about their personal life details e.g., travel, purchases, major changes, etc.

This is your conversation with the user so far.
CONVERSATION:
"""


USER_CONV_PROMPT_V2 = """
You are %s chatting with an AI assistant.
Use the given PERSONALITY to write the next thing you would say to the AI assistant in the given CONVERSATION.
- Write in less than 20 words.
- Keep the conversation flowing naturally for at least 3 turns before you ask a new question.
- Do not treat the AI assistant as a human, and and avoid asking questions on the assistant's personal preferences or experiences.
- Ask questions that the given PERSONALITY might be interested in, or general questions e.g., recent news or current events, tech, politics, sports, history, cooking, health and fitness, finance, etc.
- Always prefer asking questions in domains that have not been discussed in the given SUMMARY.
- Sometimes, ask a deep and personal questions e.g., ask about emotions, likes, dislikes, aspirations and relationships.
- Avoid asking questions on topics similar to what has been discussed in the given SUMMARY.
- Sometimes ask follow-up questions on the current topic.
- Avoid references to time such as 'last Friday', 'next month' or 'when I was ten years old'.
- Do not repeat questions asked previously in the conversation.
- Do not disclose any personal life details e.g., travel, purchases, major changes, etc.
- Do not address or refer to the AI assistant by any name, title, or pronoun..

PERSONALITY: %s

This is a summary of your previous conversations with the AI assistant.
SUMMARY:
%s

This is your conversation with the AI assistant so far.
CONVERSATION:
"""

#########################################################################

AGENT_CONV_PROMPT = (
    "%s\n\n"
    "%s last talked to %s at %s. %s\n\n"
    "Today is %s. Assume the role of %s and write the next thing you would say to %s in the conversation. "
    "Reference relevant parts of the summary when helpful, offer concise and supportive guidance, "
    "ask clarifying follow-up questions, avoid inventing lived experiences, and keep replies under 30 words. "
    "Write only the next turn in the chat.\n\n"
    "CONVERSATION:\n\n"
)

SESSION_SUMMARY_PROMPT = (
    "Previous conversations between %s and %s so far can be summarized as follows: %s. "
    "The current time and date are %s. %s and %s just had the following conversation:\n\n"
    "%s\n\n"
    "Summarize the previous and current conversations between %s and %s in 150 words or less. "
    "Include key facts about both speakers and time references.\n\n"
)

SESSION_SUMMARY_INIT_PROMPT = (
    "Write a concise summary containing key facts mentioned about %s and %s on %s "
    "in the following conversation:\n\n"
    "%s\n\n"
)


def get_msc_persona(args):
    # check if personas exist, else generate persona + summary
    if (os.path.exists(args.agent_a_file) and os.path.exists(args.agent_b_file)) and not args.overwrite_persona:
        return None, None
    else:
        all_personas = json.load(open('./data/msc_personas_all.json'))
        selected_idx = random.choice([idx for idx, d in enumerate(all_personas['train']) if not d["in_dataset"]])
        attributes = all_personas['train'][selected_idx]
        with open('./data/msc_personas_all.json', "w") as f:
            all_personas['train'][selected_idx]["in_dataset"] = 1
            json.dump(all_personas, f, indent=2)
        agent_a = get_persona(args, attributes['Speaker 1'])

        agent_a['persona_summary'] = agent_a['persona']
        agent_a['msc_prompt'] = attributes['Speaker 1']
        agent_b = get_persona(args, attributes['Speaker 2']) # setting the second agent to have age within +/- 5 years of first agent

        agent_b['persona_summary'] = agent_b['persona']
        agent_b['msc_prompt'] = attributes['Speaker 2']
        del agent_a['persona']
        del agent_b['persona']
        print("Agent A Persona: %s" % agent_a['persona_summary'])
        print("Agent B Persona: %s" % agent_b['persona_summary'])
    return agent_a, agent_b


def get_persona(args, attributes, target='human', ref_age=None):

    task = json.load(open(os.path.join(args.prompt_dir, 'persona_generation_examples.json')))
    persona_examples = [task["input_prefix"] + json.dumps(e["input"], indent=2) + '\n' + task["output_prefix"] + e["output"] for e in task['examples']]
    input_string = task["input_prefix"] + json.dumps(attributes, indent=2)

    query = PERSONA_FROM_MSC_PROMPT % (persona_examples, input_string)

    try:
        output = run_chatgpt(query, num_gen=1, num_tokens_request=1000, use_16k=True).strip()
        output = json.loads(output)
    except:
        output = run_chatgpt(query, num_gen=1, num_tokens_request=1000, use_16k=True).strip()
        output = json.loads(output)
    
    if type(output) == list:
        output = [clean_json_output(out) for out in output]
    elif type(output) == str:
        output = clean_json_output(output)
    elif type(output) == dict:
        output = {k.lower(): v for k,v in output.items()}
        pass
    else:
        raise TypeError
    
    # print(output)

    return output


def get_datetime_string(input_time='', input_date=''):

    assert input_time or input_date

    if input_date:
        year, month, day = input_date
    if input_time:
        hour, min = input_time
        time_mod = 'am' if hour <= 12 else 'pm'
        hour = hour if hour <= 12 else hour-12
        min = str(min).zfill(2)

    if input_time and not input_date:
        return str(hour) + ':' + min + ' ' + time_mod
    elif input_date and not input_time:
        return day + ' ' + month + ', ' + year
    else:
        return str(hour) + ':' + min + ' ' + time_mod + ' on ' + day + ' ' + month + ', ' + year 


def replace_captions(text, args):

    text = text.replace('[END]', '')
    text = re.sub(r"\[[^\]]*\]", "", text)
    return text.strip()

def clean_dialog(output, name):

    if output.startswith(name):
        output = output[len(name):]
        output = output.strip()
        if output[0] == ':':
            output = output[1:]
            output = output.strip()
    
    return output


def clean_json_output(output_string):

    print(output_string)

    output_string = output_string.strip()

    if output_string[0] == '[' and output_string[-1] != ']':
        start_index = output_string.index('[')
        end_index = output_string.rindex(']')
        output_string = output_string[start_index:end_index+1]

    if output_string[0] == '{' and output_string[-1] != '}':
        start_index = output_string.index('{')
        end_index = output_string.rindex('}')
        output_string = output_string[start_index:end_index+1]

    # balance brackets in json
    num_start_bracket = len(find_indices(output_string, '{'))
    num_end_bracket = len(find_indices(output_string, '}'))

    if num_start_bracket != num_end_bracket:
        if num_end_bracket < num_start_bracket:
            output_string = output_string + ' '.join(['}']*(num_start_bracket-num_end_bracket))
        if num_start_bracket < num_end_bracket:
            output_string = ' '.join(['{']*(num_end_bracket-num_start_bracket)) + ' ' + output_string

    # balance brackets in json
    num_start_bracket = len(find_indices(output_string, '['))
    num_end_bracket = len(find_indices(output_string, ']'))

    if num_start_bracket != num_end_bracket:
        if num_end_bracket < num_start_bracket:
            output_string = output_string + ' '.join(['[']*(num_start_bracket-num_end_bracket))
        if num_start_bracket < num_end_bracket:
            output_string = ' '.join([']']*(num_end_bracket-num_start_bracket)) + ' ' + output_string

    return json.loads(output_string)


def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices
