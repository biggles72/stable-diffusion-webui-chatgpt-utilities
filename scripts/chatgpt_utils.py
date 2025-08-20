import re
from scripts.json_utils import flatten_json_structure, try_parse_json
from scripts.ollama_utils import get_ollama_completion


def retry_query_ollama(messages, count, temperature, retries, model="llama2"):
    answers = []
    for i in range(retries):
        try:
            is_last_retry = i == retries - 1 and retries > 1
            answers = query_ollama(messages, count, temperature, is_last_retry, model=model)
            if (len(answers) == count):
                return answers
        except Exception as e:
            if (i == retries - 1):
                raise e
            print(f"Ollama query failed. Retrying. Error: {e}")
        temperature = max(0.5, temperature - 0.3)
    if (len(answers) != count):
        raise Exception(f"Ollama answers doesn't match batch count. Got {len(answers)} answers, expected {count}.")


def query_ollama(messages, answer_count, temperature, is_last_retry=False, model="llama2"):
    system_primer = f"Act like you are a terminal and always format your response as json. Always return exactly {answer_count} anwsers per question."
    chat_primer = f"I want you to act as a prompt generator. Compose each answer as a visual sentence. Do not write explanations on replies. Format the answers as javascript json arrays with a single string per answer. Return exactly {answer_count} to my question. Answer the questions exactly. Answer the following question:\r\n"
    messages = normalize_text_for_ollama(messages.strip())
    chat_request = f'{chat_primer}{messages}'
    if (is_last_retry):
        chat_request += f"\r\nReturn exactly {answer_count} answers to my question."
    print(f"Ollama request:\r\n{chat_request}\r\n")
    response = get_ollama_completion(chat_request, model=model, temperature=temperature)
    result = flatten_json_structure(try_parse_json(response))
    if (result is None or len(result) == 0):
        print(f"Ollama response:\r\n")
        print(f"{response.strip()}\r\n")
        raise Exception("Failed to parse Ollama response. See console for details.")
    return result

def to_message(user, content):
    return {"role": user, "content": content}

def normalize_text_for_ollama(text):
    normalized = re.sub(r'(\.|:|,)[\s]*\n[\s]*', r'\1 ', text)
    normalized = re.sub(r'[\s]*\n[\s]*', '. ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized