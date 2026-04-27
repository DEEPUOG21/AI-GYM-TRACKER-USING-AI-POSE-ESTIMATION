import os
import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_client():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        st.error("OPENROUTER_API_KEY not set.")
        st.stop()

    return api_key


def chat_ui():
    st.title("💪 AI Fitness Trainer")

    api_key = get_client()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI fitness trainer. "
                    "Help users with workouts, nutrition, form correction, and motivation."
                )
            }
        ]

    for msg in st.session_state.messages[1:]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask your fitness question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": st.session_state.messages,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()

            # Guard against missing 'choices' (bad API key, quota exceeded, etc.)
            if "choices" not in result or not result["choices"]:
                error_detail = result.get("error", {})
                if isinstance(error_detail, dict):
                    msg = error_detail.get("message", str(result))
                else:
                    msg = str(result)
                st.error(f"⚠️ OpenRouter API error: {msg}\n\nCheck that your `OPENROUTER_API_KEY` is valid and has sufficient credits.")
                return

            reply = result["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            st.error(f"⚠️ HTTP {e.response.status_code} from OpenRouter. Check your API key and quota.")
            return
        except httpx.TimeoutException:
            st.error("⚠️ Request timed out. OpenRouter did not respond in 30 s. Try again.")
            return
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")
            return

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

        with st.chat_message("assistant"):
            st.write(reply)
