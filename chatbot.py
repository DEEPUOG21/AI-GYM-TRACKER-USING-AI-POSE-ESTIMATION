import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_client():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        st.error("OPENROUTER_API_KEY not set.")
        st.stop()

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )


def chat_ui():
    st.title("💪 AI Fitness Trainer")

    client = get_client()

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

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",   # OpenRouter model format
            messages=st.session_state.messages
        )

        reply = response.choices[0].message.content

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

        with st.chat_message("assistant"):
            st.write(reply)