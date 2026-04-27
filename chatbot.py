import os
import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are an expert AI fitness trainer. "
    "Help users with workouts, nutrition, form correction, and motivation."
)

# These models are free on OpenRouter (no credits needed, just an API key).
# They are tried in order; the first successful response wins.
FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-3-4b-it:free",
]


def _try_anthropic(messages: list):
    """Call Anthropic API directly. Returns reply text or None on failure."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    anthropic_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] != "system"
    ]

    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": anthropic_messages,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]
    except Exception:
        return None


def _try_openrouter_free(messages: list):
    """Try each free OpenRouter model in turn. Returns (reply, error)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None, "OPENROUTER_API_KEY not set in environment."

    last_error = "All free models failed."

    for model in FREE_MODELS:
        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                },
                timeout=30.0,
            )

            if response.status_code in (429, 503):
                # Rate-limited or unavailable — try next model
                last_error = f"{model} unavailable ({response.status_code}), trying next..."
                continue

            if response.status_code == 402:
                # Paid model slipped in — skip
                last_error = f"{model} requires credits, trying next..."
                continue

            response.raise_for_status()
            result = response.json()

            if "choices" not in result or not result["choices"]:
                last_error = f"{model} returned no choices."
                continue

            return result["choices"][0]["message"]["content"], None

        except httpx.TimeoutException:
            last_error = f"{model} timed out."
            continue
        except Exception as e:
            last_error = str(e)
            continue

    return None, last_error


def chat_ui():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    for msg in st.session_state.messages[1:]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask your fitness question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        reply = None

        # 1. Anthropic Claude (if ANTHROPIC_API_KEY is set)
        reply = _try_anthropic(st.session_state.messages)

        # 2. Free OpenRouter models (no credits needed)
        if reply is None:
            reply, error = _try_openrouter_free(st.session_state.messages)
            if reply is None:
                st.error(
                    f"⚠️ Could not get a response: {error}\n\n"
                    "**Fix options:**\n"
                    "- Add `ANTHROPIC_API_KEY` to your Streamlit secrets, OR\n"
                    "- Add credits at https://openrouter.ai/credits"
                )
                st.session_state.messages.pop()
                return

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)
