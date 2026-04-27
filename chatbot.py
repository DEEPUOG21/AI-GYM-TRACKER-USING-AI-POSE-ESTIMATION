import os
import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are an expert AI fitness trainer. "
    "Help users with workouts, nutrition, form correction, and motivation."
)

FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-3-4b-it:free",
]


def _build_messages(session_messages: list) -> list:
    """Return messages with system role injected as a user-prefixed first turn,
    which all OpenRouter models accept regardless of provider quirks."""
    # Strip any existing system messages and rebuild as pure user/assistant turns
    turns = [m for m in session_messages if m["role"] != "system"]

    # Prepend the system prompt as a priming user→assistant exchange so even
    # models that reject the 'system' role work correctly.
    primed = [
        {"role": "user", "content": f"[Instructions] {SYSTEM_PROMPT}"},
        {"role": "assistant", "content": "Understood! I'm your AI fitness trainer. How can I help you today?"},
    ] + turns

    return primed


def _try_anthropic(session_messages: list):
    """Call Anthropic API directly. Returns reply text or None."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    turns = [m for m in session_messages if m["role"] != "system"]
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
                "messages": turns,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]
    except Exception:
        return None


def _try_openrouter_free(session_messages: list):
    """Try each free OpenRouter model. Returns (reply, error)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None, "OPENROUTER_API_KEY not set in environment."

    messages = _build_messages(session_messages)
    last_error = "All free models failed."

    for model in FREE_MODELS:
        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://apex-ai-gym-tracker.streamlit.app",
                    "X-Title": "APEX AI Gym Tracker",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1024,
                },
                timeout=30.0,
            )

            if response.status_code in (402, 429, 503):
                last_error = f"{model} returned {response.status_code}, trying next..."
                continue

            if not response.is_success:
                last_error = f"{model} — HTTP {response.status_code}: {response.text[:200]}"
                continue

            result = response.json()
            if "choices" not in result or not result["choices"]:
                last_error = f"{model} returned no choices: {result}"
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

        # 1. Try Anthropic Claude (free if ANTHROPIC_API_KEY is set)
        reply = _try_anthropic(st.session_state.messages)

        # 2. Fall back to free OpenRouter models
        if reply is None:
            reply, error = _try_openrouter_free(st.session_state.messages)
            if reply is None:
                st.error(
                    f"⚠️ Could not get a response: {error}\n\n"
                    "**Fix:** Add `ANTHROPIC_API_KEY` to your Streamlit secrets "
                    "or top up credits at https://openrouter.ai/credits"
                )
                st.session_state.messages.pop()
                return

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)
