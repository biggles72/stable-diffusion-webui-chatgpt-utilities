import requests

def list_ollama_models():
    """Return a list of installed Ollama model names."""
    url = "http://localhost:11434/api/tags"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns a dict with a 'models' key, each with a 'name'
        return [m['name'] for m in data.get('models', [])]
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []
