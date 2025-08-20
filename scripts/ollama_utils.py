import requests

# Utility for Ollama local API

def get_ollama_completion(prompt, model="llama2", temperature=1.0):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": temperature}
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    # Ollama streams responses line by line; get the last line with 'response' key
    lines = response.text.strip().split("\n")
    for line in reversed(lines):
        if '"response"' in line:
            import json
            data = json.loads(line)
            return data["response"]
    raise Exception("No valid response from Ollama.")
