import requests

GROQ_API_KEY = "YOUR_GROQ_API_KEY"

def build_prompt(query, contexts):
    context_text = "\n\n".join(contexts)

    return f"""
You are a policy assistant.

Use ONLY the context below to answer.

CONTEXT:
{context_text}

QUESTION:
{query}

Rules:
- If answer not in context, say "I don't know based on the policies"
- Keep answer short
- Always mention source is company policy

Answer:
"""


def call_llm(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(url, json=data, headers=headers)
    return response.json()["choices"][0]["message"]["content"]


def rag_answer(query, retrieve_fn):
    contexts = retrieve_fn(query, k=3)
    prompt = build_prompt(query, contexts)
    return call_llm(prompt), contexts
