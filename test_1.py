# import os
# import json
# import time
# from typing import List, Dict, Any, Generator, Optional

# import streamlit as st
# from dotenv import load_dotenv
# from openai import AzureOpenAI
# from openai import APIError, RateLimitError, APITimeoutError, InternalServerError

# # =============================
# # Boot & Config
# # =============================
# load_dotenv()

# st.set_page_config(page_title="Azure OpenAI Chat", page_icon="🤖", layout="wide")

# # --- Read ENV ---
# AZURE_ENDPOINT   = os.getenv("AZURE_ENDPOINT", "").strip()
# AZURE_API_KEY    = os.getenv("AZURE_API_KEY", "").strip()
# AZURE_API_VERSION= os.getenv("AZURE_API_VERSION", "2024-10-01-preview").strip()
# # You can keep a default here; user can override in sidebar
# DEFAULT_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "").strip()

# # =============================
# # Helpers
# # =============================
# def env_ready() -> bool:
#     missing = []
#     if not AZURE_ENDPOINT:    missing.append("AZURE_ENDPOINT")
#     if not AZURE_API_KEY:     missing.append("AZURE_API_KEY")
#     if not AZURE_API_VERSION: missing.append("AZURE_API_VERSION")
#     if missing:
#         with st.sidebar:
#             st.error("Missing required environment variables:\n\n" + "\n".join(f"- {m}" for m in missing))
#             st.info(
#                 "Create a `.env` file with:\n"
#                 "AZURE_ENDPOINT=https://<YOUR-RESOURCE>.openai.azure.com/\n"
#                 "AZURE_API_KEY=****\n"
#                 "AZURE_API_VERSION=2024-10-01-preview\n"
#                 "AZURE_DEPLOYMENT=<your-deployment-name>"
#             )
#         return False
#     return True

# @st.cache_resource(show_spinner=False)
# def get_client() -> AzureOpenAI:
#     # Reuse one client across runs
#     return AzureOpenAI(
#         api_key=AZURE_API_KEY,
#         api_version=AZURE_API_VERSION,
#         azure_endpoint=AZURE_ENDPOINT,  # type: ignore
#         timeout=120,
#     )

# def backoff_sleep(attempt: int):
#     # exponential-ish backoff
#     time.sleep(min(2 ** attempt, 10))

# def build_payload_messages(history: List[Dict[str, str]], system_prompt: str, k_pairs: int) -> List[Dict[str, str]]:
#     # Take only user/assistant from history, keep last 2*k messages
#     convo = [m for m in history if m["role"] in ("user", "assistant")]
#     window = convo[-(2 * k_pairs):] if k_pairs > 0 else convo
#     return [{"role": "system", "content": system_prompt}] + window

# def message_to_markdown(messages: List[Dict[str, str]]) -> str:
#     out = ["# Chat Export\n"]
#     for m in messages:
#         role = m["role"].capitalize()
#         out.append(f"## {role}\n\n{m['content']}\n")
#     return "\n".join(out)

# def messages_to_json(messages: List[Dict[str, str]]) -> str:
#     return json.dumps(messages, indent=2, ensure_ascii=False)

# def make_stream_generator(client: AzureOpenAI, model: str, messages: List[Dict[str, str]], **params) -> Generator[str, None, str]:
#     """
#     Yields tokens as they arrive and returns the final full string at the end.
#     """
#     full_text = []
#     stream = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         stream=True,
#         **params,
#     )
#     for chunk in stream:
#         if not chunk.choices:
#             continue
#         delta = chunk.choices[0].delta
#         if delta and delta.content:
#             full_text.append(delta.content)
#             yield delta.content  # stream each token
#     return "".join(full_text)

# def render_message(msg: Dict[str, str]):
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])

# def try_chat_completion_stream(
#     client: AzureOpenAI,
#     model: str,
#     messages: List[Dict[str, str]],
#     max_retries: int = 3,
#     **params
# ) -> str:
#     """
#     Streams text with retries. Returns the final text.
#     """
#     last_err: Optional[Exception] = None
#     for attempt in range(max_retries):
#         try:
#             # Stream to a placeholder using write_stream for no flicker
#             stream_gen = make_stream_generator(client, model, messages, **params)
#             out_text = st.write_stream(stream_gen)  # returns the generator's return value if provided
#             if out_text is None:  # write_stream returns None unless generator returns at end
#                 # If None, reconstruct from messages (we already added to session), but safer to refetch:
#                 # Here we simply can't, so fall back to empty string (rare case).
#                 out_text = ""
#             return out_text
#         except (RateLimitError, APITimeoutError, InternalServerError, APIError) as e:
#             last_err = e
#             with st.status("Transient error. Retrying...", expanded=False) as status:
#                 status.update(label=f"Attempt {attempt+1}/{max_retries} failed: {type(e).__name__}")
#                 backoff_sleep(attempt + 1)
#         except Exception as e:
#             # Non-retryable
#             last_err = e
#             break
#     # Out of retries
#     st.error(f"⚠️ Error: {last_err}")
#     return f"⚠️ Error: {last_err}"

# # =============================
# # Defaults & Session
# # =============================
# DEFAULT_SYSTEM_PROMPT = (
#     "You are a kind, friendly, and helpful AI assistant. "
#     "Keep responses concise but complete. Use a conversational tone with a hint of humor when appropriate."
# )

# SYSTEM_PRESETS = {
#     "Friendly Helper (default)": DEFAULT_SYSTEM_PROMPT,
#     "Strict & Formal": "You are precise, formal, and concise. Prefer bullet points. Avoid jokes.",
#     "Brainstorm Buddy": "You excel at creative brainstorming. Offer diverse ideas, ask clarifying questions, and be optimistic.",
#     "Teacher Mode": "Explain concepts step-by-step with simple examples. Check understanding and avoid jargon.",
# }

# if "messages" not in st.session_state:
#     st.session_state.messages: List[Dict[str, str]] = [
#         {"role": "assistant", "content": "Hello there, I'm your ChatGPT clone (Azure edition)! 😊"}
#     ]

# if "system_prompt" not in st.session_state:
#     st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

# if "last_raw_json" not in st.session_state:
#     st.session_state.last_raw_json = None

# # =============================
# # Sidebar (Controls)
# # =============================
# with st.sidebar:
#     st.header("⚙️ Settings")

#     # Deployment / Model controls
#     deployment = st.text_input("Azure Deployment (model)", value=DEFAULT_DEPLOYMENT, placeholder="e.g., gpt-4o-mini-chat")
#     colA, colB = st.columns(2)
#     with colA:
#         temperature = st.slider("Temperature", 0.0, 2.0, 0.6, 0.1)
#         top_p       = st.slider("Top-p", 0.0, 1.0, 1.0, 0.05)
#         max_tokens  = st.slider("Max tokens", 16, 4096, 1024, 16)
#     with colB:
#         freq_penalty = st.slider("Frequency penalty", -2.0, 2.0, 0.0, 0.1)
#         pres_penalty = st.slider("Presence penalty",  -2.0, 2.0, 0.0, 0.1)
#         window_k     = st.slider("Memory window (K pairs)", 0, 20, 4, 1)

#     st.divider()
#     st.subheader("System Prompt")
#     preset = st.selectbox("Preset", list(SYSTEM_PRESETS.keys()), index=0)
#     if st.button("Apply Preset"):
#         st.session_state.system_prompt = SYSTEM_PRESETS[preset]
#     st.session_state.system_prompt = st.text_area("Edit system prompt", value=st.session_state.system_prompt, height=140)
#     if st.button("Reset to Default"):
#         st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

#     st.divider()
#     show_raw = st.toggle("Show raw JSON response", value=False)
#     allow_images = st.toggle("Allow image input (vision)", value=False, help="Enable image upload to send with your prompt if your deployment supports it.")

#     st.divider()
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         if st.button("🧹 Clear Chat"):
#             st.session_state.messages = [{"role": "assistant", "content": "New chat started. How can I help? 😊"}]
#             st.rerun()
#     with c2:
#         if st.button("🔁 Retry Last"):
#             # re-send last user message
#             if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
#                 # find last user
#                 for m in reversed(st.session_state.messages):
#                     if m["role"] == "user":
#                         st.session_state.messages.append({"role": "assistant", "content": "Retrying that…"})
#                         st.rerun()
#                         break
#     with c3:
#         export_choice = st.selectbox("Export", ["—", "Markdown", "JSON"], index=0)
#         if export_choice == "Markdown":
#             st.download_button("⬇️ Download .md", data=message_to_markdown(st.session_state.messages), file_name="chat_export.md", mime="text/markdown")
#         elif export_choice == "JSON":
#             st.download_button("⬇️ Download .json", data=messages_to_json(st.session_state.messages), file_name="chat_export.json", mime="application/json")

# # =============================
# # Top UI
# # =============================
# st.title("ChatGPT-like clone (Azure OpenAI) 🤖")
# st.caption("Streaming • Memory window • Exports • Vision (optional)")

# if not env_ready():
#     st.stop()

# client = get_client()

# # =============================
# # Render History
# # =============================
# for msg in st.session_state.messages:
#     render_message(msg)

# # =============================
# # Chat Input (with optional image)
# # =============================
# image_bytes = None
# image_name = None
# if allow_images:
#     img_file = st.file_uploader("Add an image (optional)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=False)
#     if img_file is not None:
#         image_bytes = img_file.read()
#         image_name  = img_file.name
#         st.image(image_bytes, caption=image_name, use_container_width=True)

# user_prompt = st.chat_input("Type your message…")

# if user_prompt:
#     # Build possibly multimodal user content
#     user_content: Any = user_prompt
#     if allow_images and image_bytes:
#         # Azure Chat Completions supports content as a list of items (text + image_url/base64)
#         # We'll embed the image as base64 data URL.
#         import base64
#         b64 = base64.b64encode(image_bytes).decode("utf-8")
#         user_content = [
#             {"type": "text", "text": user_prompt},
#             {"type": "image_url", "image_url": {"url": f"data:image/*;base64,{b64}" }},
#         ]

#     st.session_state.messages.append({"role": "user", "content": user_prompt})
#     with st.chat_message("user"):
#         st.write(user_prompt)

#     # Assistant placeholder
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking…"):
#             # ---- Build payload with window + system ----
#             payload_messages = build_payload_messages(st.session_state.messages, st.session_state.system_prompt, window_k)

#             # Replace the last user text content in payload with multimodal structure if applicable
#             if isinstance(user_content, list):
#                 # Find the last user message index in payload and swap content to multimodal
#                 for i in range(len(payload_messages) - 1, -1, -1):
#                     if payload_messages[i]["role"] == "user":
#                         payload_messages[i] = {"role": "user", "content": user_content}
#                         break

#             # ---- Call Azure with streaming & retries ----
#             params = dict(
#                 temperature=temperature,
#                 top_p=top_p,
#                 max_tokens=max_tokens,
#                 frequency_penalty=freq_penalty,
#                 presence_penalty=pres_penalty,
#             )

#             # If user forgot to set a deployment, show a friendly error
#             if not deployment:
#                 ai_text = "⚠️ Please set your **Azure Deployment** name in the sidebar."
#                 st.error(ai_text)
#             else:
#                 # Stream response
#                 ai_text = try_chat_completion_stream(
#                     client=client,
#                     model=deployment,  # deployment name
#                     messages=payload_messages,
#                     **params,
#                 )

#     # Append assistant reply to history
#     st.session_state.messages.append({"role": "assistant", "content": ai_text or ""})

#     # Optional: Copy last reply
#     last = st.session_state.messages[-1]["content"]
#     st.toast("Reply ready. You can export from the sidebar.")
#     st.code(last, language="markdown")

# # =============================
# # Debug JSON (toggle)
# # =============================
# if show_raw:
#     st.divider()
#     st.subheader("🧪 Debug / Raw (last turn)")
#     st.write("Note: Streaming returns chunks; this section only shows the **final turn** content.")
#     st.json({
#         "endpoint": AZURE_ENDPOINT,
#         "api_version": AZURE_API_VERSION,
#         "deployment": deployment or "<unset>",
#         "messages_tail": st.session_state.messages[-6:],
#         "settings": {
#             "temperature": temperature,
#             "top_p": top_p,
#             "max_tokens": max_tokens,
#             "frequency_penalty": freq_penalty,
#             "presence_penalty": pres_penalty,
#             "window_k": window_k,
#         },
#     })

# app.py — Azure OpenAI Multi-Tool Assistant (Streaming + Browse + Files + Data + Images)
# --------------------------------------------------------------------------------------
# Features:
# - Streaming chat (token-by-token) with presets for writing/coding tasks
# - Web browsing with citations (ddgs + trafilatura)
# - Files/PDFs: summarize, extract tables, compare, export
# - Data Lab: CSV preview, quick EDA, optional "LLM -> Python code" sandbox (approve & run)
# - Images: text-to-image (Azure Images), basic edit/variation if supported
#
# Optional deps (install as needed):
#   pip install streamlit python-dotenv openai ddgs trafilatura pdfplumber pandas duckdb matplotlib python-docx
#
# Env:
#   AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION (e.g., 2024-10-01-preview), AZURE_DEPLOYMENT (chat)
#   AZURE_IMAGES_DEPLOYMENT (optional for Images)
# --------------------------------------------------------------------------------------

import os
import io
import re
import json
import time
import base64
import contextlib
from typing import List, Dict, Any, Optional, Generator

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai import APIError, RateLimitError, APITimeoutError, InternalServerError

# ---- Optional imports (soft) ----
with contextlib.suppress(Exception):
    import pandas as pd
with contextlib.suppress(Exception):
    import matplotlib.pyplot as plt  # NOTE: don't set styles/colors
with contextlib.suppress(Exception):
    import pdfplumber
with contextlib.suppress(Exception):
    import duckdb  # for quick SQL over CSVs
with contextlib.suppress(Exception):
    import trafilatura
with contextlib.suppress(Exception):
    from ddgs import DDGS  # renamed duckduckgo-search

with contextlib.suppress(Exception):
    from docx import Document

load_dotenv()
st.set_page_config(page_title="Azure OpenAI Assistant", page_icon="🧠", layout="wide")

# -----------------------------
# Environment
# -----------------------------
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "").strip()
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "").strip()
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-10-01-preview").strip()
DEFAULT_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "").strip()  # chat
IMAGES_DEPLOYMENT = os.getenv("AZURE_IMAGES_DEPLOYMENT", "").strip()  # images (optional)

# -----------------------------
# Helpers
# -----------------------------
def env_ready() -> bool:
    missing = []
    if not AZURE_ENDPOINT: missing.append("AZURE_ENDPOINT")
    if not AZURE_API_KEY: missing.append("AZURE_API_KEY")
    if not AZURE_API_VERSION: missing.append("AZURE_API_VERSION")
    if missing:
        st.error("Missing environment variables:\n" + "\n".join(f"- {m}" for m in missing))
        with st.expander("How to configure"):
            st.code(
                "AZURE_ENDPOINT=https://<YOUR-RESOURCE>.openai.azure.com/\n"
                "AZURE_API_KEY=...\n"
                "AZURE_API_VERSION=2024-10-01-preview\n"
                "AZURE_DEPLOYMENT=<your-chat-deployment>\n"
                "# Optional for images:\n"
                "AZURE_IMAGES_DEPLOYMENT=<your-images-deployment>",
                language="bash"
            )
        return False
    return True

@st.cache_resource(show_spinner=False)
def get_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,  # type: ignore
        # timeout=120,
    )

def backoff_sleep(attempt: int):
    time.sleep(min(2 ** attempt, 10))

def write_stream_from_chunks(stream) -> str:
    """Stream tokens to UI; return final text (best-effort)."""
    full = []
    for chunk in stream:
        if not getattr(chunk, "choices", None):  # safety
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            full.append(delta.content)
            yield delta.content
    return "".join(full)

def build_payload_messages(history: List[Dict[str, Any]], system_prompt: str, k_pairs: int) -> List[Dict[str, Any]]:
    convo = [m for m in history if m["role"] in ("user", "assistant")]
    window = convo[-(2 * k_pairs):] if k_pairs > 0 else convo
    return [{"role": "system", "content": system_prompt}] + window

def try_stream_chat(client: AzureOpenAI, model: str, messages: List[Dict[str, Any]], **params) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **params,
            )
            out = st.write_stream(write_stream_from_chunks(stream))
            return out or ""
        except (RateLimitError, APITimeoutError, InternalServerError, APIError) as e:
            last_err = e
            with st.status("Retrying after transient error…", expanded=False) as s:
                s.update(label=f"{type(e).__name__} on attempt {attempt+1}")
                backoff_sleep(attempt + 1)
        except Exception as e:
            last_err = e
            break
    st.error(f"⚠️ Chat error: {last_err}")
    return f"⚠️ Chat error: {last_err}"

def md_export(messages: List[Dict[str, Any]]) -> str:
    out = ["# Chat Export\n"]
    for m in messages:
        role = m["role"].capitalize()
        out.append(f"## {role}\n\n{m['content']}\n")
    return "\n".join(out)

def json_export(messages: List[Dict[str, Any]]) -> str:
    return json.dumps(messages, indent=2, ensure_ascii=False)

# -----------------------------
# Presets & Templates
# -----------------------------
DEFAULT_SYSTEM = (
    "You are a helpful, friendly assistant. Be concise yet complete. Use a warm tone and simple structure."
)

WRITING_TEMPLATES = {
    "Explain concept": "Explain {topic} to a {audience_level} with a clear analogy and a 3-step breakdown.",
    "Brainstorm ideas": "Brainstorm 10 creative ideas about {goal}. Group by theme, add 1 pro & 1 con each.",
    "Outline document": "Create a detailed outline for a {doc_type} about {subject}. Include sections, bullet points, and callouts.",
    "Rewrite in my tone": "Rewrite the following in a friendly, confident tone. Keep key facts. Text:\n\n{text}",
    "Translate": "Translate to {language}, preserving tone and formatting. Text:\n\n{text}",
    "Draft email": "Draft a crisp, polite email to {recipient} about {purpose}. Include subject line and 3 action bullets.",
    "Draft SOP": "Create a step-by-step SOP for {process}. Include prerequisites, steps, checks, and failure handling.",
    "Slide outline": "Make a 10-slide outline on {topic}. Slide titles + 3 bullets each. End with key risks & next steps.",
    "Resume bullet": "Turn this achievement into 3 strong resume bullets with metrics and impact:\n\n{text}",
}

CODE_TEMPLATES = {
    "Write code": "Write {language} code to {task}. Include comments and a brief docstring.",
    "Review code": "Review the following {language} code for correctness, clarity, and performance. Suggest improvements:\n\n```{language}\n{code}\n```",
    "Debug": "Find and fix bugs in this {language} code. Explain the root causes and provide corrected code:\n\n```{language}\n{code}\n```",
    "Refactor": "Refactor this {language} code for readability and testability. Keep behavior same:\n\n```{language}\n{code}\n```",
    "Generate tests": "Write unit tests for this {language} module using {framework}. Cover edge cases:\n\n```{language}\n{code}\n```",
    "Suggest architecture": "Propose an architecture for {app_type} using {stack}. Include modules, data flow, and trade-offs."
}

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey! I’m your Azure OpenAI assistant—what shall we build today? 🚀"}
    ]
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    deployment = st.text_input("Chat deployment", value=DEFAULT_DEPLOYMENT, placeholder="e.g. gpt-4o-mini-chat")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.6, 0.1)
    top_p = st.slider("Top-p", 0.0, 1.0, 1.0, 0.05)
    max_tokens = st.slider("Max tokens", 16, 8192, 1024, 16)
    freq_penalty = st.slider("Frequency penalty", -2.0, 2.0, 0.0, 0.1)
    pres_penalty = st.slider("Presence penalty", -2.0, 2.0, 0.0, 0.1)
    window_k = st.slider("Memory window (K pairs)", 0, 20, 4, 1)

    st.divider()
    st.subheader("System Prompt")
    st.session_state.system_prompt = st.text_area("Edit system", value=st.session_state.system_prompt, height=120)
    if st.button("Reset system to default"):
        st.session_state.system_prompt = DEFAULT_SYSTEM

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🧹 Clear chat"):
            st.session_state.messages = [{"role": "assistant", "content": "New chat started. How can I help? 😊"}]
            st.rerun()
    with c2:
        st.download_button("⬇️ Export .md", data=md_export(st.session_state.messages),
                           file_name="chat.md", mime="text/markdown")
    with c3:
        st.download_button("⬇️ Export .json", data=json_export(st.session_state.messages),
                           file_name="chat.json", mime="application/json")

# -----------------------------
# App Tabs
# -----------------------------
st.title("Azure OpenAI Assistant 🧠")
tabs = st.tabs(["💬 Chat & Writing", "🧑‍💻 Code Help", "🌐 Browse (with sources)", "📄 Files & PDFs", "📊 Data & Analysis", "🖼️ Images"])

if not env_ready():
    st.stop()
client = get_client()

# =====================================================================
# TAB 1: Chat & Writing
# =====================================================================
with tabs[0]:
    st.subheader("Core chat, writing, explaining, brainstorming, outlining, rewriting/translation")
    with st.expander("📌 Writing templates", expanded=False):
        colA, colB, colC = st.columns(3)
        for i, (name, tmpl) in enumerate(WRITING_TEMPLATES.items()):
            col = [colA, colB, colC][i % 3]
            with col:
                if st.button(name):
                    st.session_state.messages.append({"role": "user", "content": tmpl})

    # Render history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # Input (with optional image for vision deployments)
    allow_vision = st.toggle("Allow image in prompt (if your chat deployment supports vision)", value=False)
    image_bytes = None
    if allow_vision:
        up = st.file_uploader("Attach image (optional)", type=["png", "jpg", "jpeg", "webp"])
        if up:
            image_bytes = up.read()
            st.image(image_bytes, caption=up.name, use_container_width=True)

    user_text = st.chat_input("Type your message…")
    if user_text:
        # Show user bubble
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        # Prepare payload
        payload = build_payload_messages(st.session_state.messages, st.session_state.system_prompt, window_k)

        # If multimodal content
        if allow_vision and image_bytes:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            # Replace last user message content with a content-list
            for i in range(len(payload) - 1, -1, -1):
                if payload[i]["role"] == "user":
                    payload[i] = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/*;base64,{b64}"}}
                        ]
                    }
                    break

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                if not deployment:
                    ai = "⚠️ Set your chat deployment in the sidebar."
                    st.error(ai)
                else:
                    ai = try_stream_chat(
                        client,
                        model=deployment,
                        messages=payload,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        frequency_penalty=freq_penalty,
                        presence_penalty=pres_penalty,
                    )
        st.session_state.messages.append({"role": "assistant", "content": ai or ""})

# =====================================================================
# TAB 2: Code Help
# =====================================================================
with tabs[1]:
    st.subheader("Write • Review • Debug • Refactor • Tests • Architecture")
    mode = st.selectbox("Task", list(CODE_TEMPLATES.keys()), index=0)
    lang = st.selectbox("Language", ["Python", "SQL", "JavaScript", "TypeScript", "Java", "C#", "Go", "Rust", "Bash", "Other"], index=0)
    framework = st.text_input("Test framework (if generating tests)", value="pytest" if lang=="Python" else "")
    app_type = st.text_input("App type (for architecture)", value="Streamlit + LangChain + Snowflake")
    stack = st.text_input("Stack (for architecture)", value="Python, Streamlit, LangGraph, Snowflake")
    task = st.text_input("Short description / goal", value="build a CRUD API with auth")
    code_in = st.text_area("Paste code (for review/debug/refactor/tests)", height=200)

    # Prepare prompt
    tmpl = CODE_TEMPLATES[mode]
    prompt = tmpl.format(language=lang, framework=framework, app_type=app_type, stack=stack, task=task, code=code_in)
    st.code(prompt, language="markdown")
    if st.button("Ask assistant (streaming)"):
        with st.chat_message("user"): st.write(prompt)
        payload = build_payload_messages(st.session_state.messages + [{"role":"user","content": prompt}],
                                         st.session_state.system_prompt, window_k)
        with st.chat_message("assistant"):
            ai = try_stream_chat(client, model=deployment or DEFAULT_DEPLOYMENT, messages=payload,
                                 temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                 frequency_penalty=freq_penalty, presence_penalty=pres_penalty)
        st.session_state.messages += [{"role":"user","content": prompt},{"role":"assistant","content": ai or ""}]

# =====================================================================
# TAB 3: Browse (with sources)
# =====================================================================
with tabs[2]:
    st.subheader("Live web browsing with sources & citations")
    browse_ok = ("DDGS" in globals()) and ("trafilatura" in globals())
    if not browse_ok:
        st.warning("Install: `pip install ddgs trafilatura` to enable browsing.")
    query = st.text_input("Search query", placeholder="e.g., latest news about Mars Incorporated sustainability")
    n_results = st.slider("Results", 1, 10, 5)
    include_page_text = st.toggle("Include extracted page text in LLM context", value=True)

    if st.button("Search & Summarize") and browse_ok and query:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=n_results, safesearch="moderate"):
                # r keys typically: title, href, body
                results.append({"title": r.get("title",""), "url": r.get("href",""), "snippet": r.get("body","")})

        sources_panel = st.container()
        ctx_chunks = []
        for idx, r in enumerate(results, start=1):
            with sources_panel.expander(f"[{idx}] {r['title'] or r['url']}", expanded=False):
                st.write(r["url"])
                st.write(r["snippet"])
            if include_page_text and r["url"]:
                try:
                    html = trafilatura.fetch_url(r["url"])
                    text = trafilatura.extract(html, include_comments=False) or ""
                    if text:
                        ctx_chunks.append(f"[{idx}] {r['title']} — {r['url']}\n{text[:4000]}")  # cap per source
                except Exception:
                    pass

        # Build a prompt with inline citation style like [1], [2], ...
        browse_prompt = (
            "Using the following web context, write a concise, up-to-date answer. "
            "Cite sources inline like [1], [2].\n\n"
            "=== Web Context ===\n" + ("\n\n".join(ctx_chunks) if ctx_chunks else "No page text available.") +
            "\n\n=== User Question ===\n" + query
        )

        with st.chat_message("user"): st.write(query)
        payload = build_payload_messages(st.session_state.messages + [{"role":"user","content": browse_prompt}],
                                         st.session_state.system_prompt, window_k)
        with st.chat_message("assistant"):
            ai = try_stream_chat(client, model=deployment or DEFAULT_DEPLOYMENT, messages=payload,
                                 temperature=0.3, top_p=top_p, max_tokens=max_tokens)
        st.session_state.messages += [{"role":"user","content": browse_prompt},{"role":"assistant","content": ai or ""}]

# =====================================================================
# TAB 4: Files & PDFs
# =====================================================================
with tabs[3]:
    st.subheader("Summarize • Extract tables • Compare • Export structured data")
    uploaded = st.file_uploader("Upload one or two documents", type=["pdf","docx","txt"], accept_multiple_files=True)
    action = st.selectbox("Action", ["Summarize", "Extract tables (PDF only)", "Compare two files", "Extract to JSON"])

    def read_text(file) -> str:
        name = file.name.lower()
        if name.endswith(".txt"):
            return file.read().decode("utf-8", errors="ignore")
        if name.endswith(".docx") and "Document" in globals():
            try:
                file_bytes = io.BytesIO(file.read())
                doc = Document(file_bytes)
                return "\n".join(p.text for p in doc.paragraphs)
            except Exception:
                return ""
        if name.endswith(".pdf") and "pdfplumber" in globals():
            text = []
            with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                for page in pdf.pages:
                    text.append(page.extract_text() or "")
            return "\n".join(text)
        return ""

    def extract_pdf_tables(file) -> List[pd.DataFrame]:
        dfs = []
        if "pdfplumber" in globals() and "pd" in globals():
            with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                for page in pdf.pages:
                    try:
                        tables = page.extract_tables()
                        for t in tables or []:
                            df = pd.DataFrame(t[1:], columns=t[0])
                            dfs.append(df)
                    except Exception:
                        pass
        return dfs

    if uploaded:
        if action == "Summarize":
            full_text = ""
            for f in uploaded[:2]:  # cap to 2 to keep token usage sane
                with st.expander(f"Preview: {f.name}", expanded=False):
                    st.write("Showing first ~2000 chars.")
                text = read_text(f)
                full_text += f"\n\n--- {f.name} ---\n" + text[:10000]  # cap per file
            prompt = f"Summarize the following documents into a crisp brief with headings and bullet points also roast after anlyse it:\n{full_text}"
            with st.chat_message("user"): st.write(f"Summarize {', '.join([f.name for f in uploaded])}")
            payload = build_payload_messages(st.session_state.messages + [{"role":"user","content": prompt}],
                                             st.session_state.system_prompt, window_k)
            with st.chat_message("assistant"):
                ai = try_stream_chat(client, deployment or DEFAULT_DEPLOYMENT, payload, temperature=0.3, max_tokens=max_tokens)
            st.session_state.messages += [{"role":"user","content": prompt},{"role":"assistant","content": ai or ""}]

        elif action == "Extract tables (PDF only)":
            if "pd" not in globals():
                st.error("Install pandas for table extraction: pip install pandas pdfplumber")
            else:
                all_tables = []
                for f in uploaded:
                    if not f.name.lower().endswith(".pdf"):
                        st.warning(f"Skipping non-PDF: {f.name}")
                        continue
                    f_bytes = f.read()
                    tables = extract_pdf_tables(io.BytesIO(f_bytes))
                    for i, df in enumerate(tables, 1):
                        st.write(f"**{f.name} — Table {i}**")
                        st.dataframe(df)
                        all_tables.append((f.name, i, df))
                if all_tables:
                    # export as CSV zip
                    import zipfile, tempfile
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
                    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
                        for fname, idx, df in all_tables:
                            csv_bytes = df.to_csv(index=False).encode("utf-8")
                            zf.writestr(f"{os.path.splitext(fname)[0]}_table_{idx}.csv", csv_bytes)
                    st.download_button("⬇️ Download all CSVs (zip)", data=open(tmp.name,"rb").read(),
                                       file_name="extracted_tables.zip", mime="application/zip")

        elif action == "Compare two files":
            if len(uploaded) != 2:
                st.warning("Please upload exactly two files.")
            else:
                t1 = read_text(uploaded[0])
                t2 = read_text(uploaded[1])
                diff_prompt = (
                    "Compare the two documents. Summarize key differences, changed numbers, and tone shifts.\n\n"
                    f"--- File A ({uploaded[0].name}) ---\n{t1[:12000]}\n\n"
                    f"--- File B ({uploaded[1].name}) ---\n{t2[:12000]}"
                )
                with st.chat_message("user"): st.write(f"Compare {uploaded[0].name} vs {uploaded[1].name}")
                payload = build_payload_messages(st.session_state.messages + [{"role":"user","content": diff_prompt}],
                                                 st.session_state.system_prompt, window_k)
                with st.chat_message("assistant"):
                    ai = try_stream_chat(client, deployment or DEFAULT_DEPLOYMENT, payload, temperature=0.2, max_tokens=max_tokens)
                st.session_state.messages += [{"role":"user","content": diff_prompt},{"role":"assistant","content": ai or ""}]

        elif action == "Extract to JSON":
            full_text = ""
            for f in uploaded[:2]:
                text = read_text(f)
                full_text += f"\n\n--- {f.name} ---\n" + text[:16000]
            jprompt = (
                "Extract structured data as JSON arrays of objects (keys inferred from headings and tables). "
                "Return ONLY JSON, no prose.\n\n" + full_text
            )
            with st.chat_message("user"): st.write("Extract to JSON")
            payload = build_payload_messages(st.session_state.messages + [{"role":"user","content": jprompt}],
                                             st.session_state.system_prompt, window_k)
            with st.chat_message("assistant"):
                ai = try_stream_chat(client, deployment or DEFAULT_DEPLOYMENT, payload, temperature=0.0, max_tokens=max_tokens)
            st.session_state.messages += [{"role":"user","content": jprompt},{"role":"assistant","content": ai or ""}]

# =====================================================================
# TAB 5: Data & Analysis
# =====================================================================
with tabs[4]:
    st.subheader("CSV preview • Quick EDA • Python sandbox (approve & run) • Charts & downloads")
    if "pd" not in globals():
        st.warning("Install pandas/matplotlib/duckdb for this tab: pip install pandas matplotlib duckdb")
    else:
        data_files = st.file_uploader("Upload CSV (multiple allowed)", type=["csv","excel"], accept_multiple_files=True)
        dfs = {}
        if data_files:
            for f in data_files:
                df = pd.read_csv(f)
                dfs[f.name] = df
                st.write(f"**{f.name}** — {df.shape[0]} rows × {df.shape[1]} cols")
                st.dataframe(df.head(20))

            st.divider()
            if st.button("Quick EDA"):
                for name, df in dfs.items():
                    st.write(f"### {name} — describe()")
                    st.dataframe(df.describe(include="all", datetime_is_numeric=True))

            st.divider()
            sql_area = st.text_area("Run SQL over all CSVs (duckdb). Refer to tables as file names without dots.", value="-- Example:\n-- SELECT * FROM my_data WHERE amount > 100 LIMIT 20;")
            if st.button("Run SQL"):
                # register dataframes in duckdb
                con = duckdb.connect()
                for name, df in dfs.items():
                    safe = re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(name)[0])
                    con.register(safe, df)
                try:
                    if sql_area.strip() and not sql_area.strip().startswith("--"):
                        out = con.execute(sql_area).df()
                        st.dataframe(out)
                        st.download_button("⬇️ Download results (CSV)", data=out.to_csv(index=False).encode("utf-8"),
                                           file_name="query_results.csv", mime="text/csv")
                    else:
                        st.info("Enter SQL and click Run.")
                except Exception as e:
                    st.error(f"SQL error: {e}")

            st.divider()
            st.markdown("#### LLM → Python (sandbox)")
            st.caption("The assistant proposes Python code; you can **review & approve** before running.")
            analysis_goal = st.text_input("Describe what you want (e.g., 'plot sales by month and top 5 products')")
            if st.button("Generate analysis code"):
                gen_prompt = (
                    "Write Python (pandas + matplotlib) to analyze the loaded dataframes. "
                    "DO NOT import seaborn. Each chart must be a single matplotlib figure. "
                    "The available dataframes are named after the uploaded CSV files (sanitized underscores). "
                    "Print key tables and show plots. Finally, save any important tables to CSV in memory and "
                    "print the filenames. Goal:\n" + analysis_goal
                )
                payload = build_payload_messages(st.session_state.messages + [{"role":"user","content": gen_prompt}],
                                                 st.session_state.system_prompt, window_k)
                with st.chat_message("assistant"):
                    code_text = try_stream_chat(client, deployment or DEFAULT_DEPLOYMENT, payload, temperature=0.2, max_tokens=1400)
                st.session_state.generated_code = code_text

            if "generated_code" in st.session_state and st.session_state.generated_code:
                st.code(st.session_state.generated_code, language="python")
                if st.button("Run code (unsafe — runs locally)"):
                    # Build safe-ish exec env
                    safe_globals = {"__builtins__": __builtins__}
                    safe_locals = {}
                    # inject pandas, plt, dfs
                    if "pd" in globals(): safe_globals["pd"] = pd
                    if "plt" in globals(): safe_globals["plt"] = plt
                    # register dfs with safe names
                    for name, df in dfs.items():
                        safe = re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(name)[0])
                        safe_globals[safe] = df
                    # capture prints
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        try:
                            exec(st.session_state.generated_code, safe_globals, safe_locals)
                            st.success("Code executed.")
                            out_text = buf.getvalue()
                            if out_text.strip():
                                st.text_area("Console output", value=out_text, height=180)
                            # Show any active figures
                            try:
                                st.pyplot(plt.gcf())
                            except Exception:
                                pass
                        except Exception as e:
                            st.error(f"Execution error: {e}")

# =====================================================================
# TAB 6: Images
# =====================================================================
with tabs[5]:
    st.subheader("Text-to-image • (Optional) Edit/variation if supported")
    if not IMAGES_DEPLOYMENT:
        st.info("Set `AZURE_IMAGES_DEPLOYMENT` to enable image generation.")
    else:
        prompt = st.text_area("Image prompt", placeholder="A minimalist poster of an electric scooter in monsoon, cinematic lighting")
        img_to_edit = st.file_uploader("Upload image to edit/variation (optional)", type=["png","jpg","jpeg","webp"])
        size = st.selectbox("Size", ["1024x1024", "1024x576", "576x1024"], index=0)
        n_imgs = st.slider("Number of images", 1, 4, 1)

        def show_image_b64(b64: str, name: str):
            st.image(base64.b64decode(b64), caption=name, use_container_width=True)
            st.download_button("⬇️ Download PNG", data=base64.b64decode(b64), file_name=f"{name}.png", mime="image/png")

        if st.button("Generate"):
            if not prompt.strip():
                st.warning("Enter a prompt.")
            else:
                with st.spinner("Generating…"):
                    try:
                        # Generate (Azure Images via OpenAI SDK)
                        result = client.images.generate(
                            model=IMAGES_DEPLOYMENT,  # deployment name for images
                            prompt=prompt,
                            size=size,
                            n=n_imgs,
                            response_format="b64_json",
                        )
                        for i, d in enumerate(result.data, 1):
                            show_image_b64(d.b64_json, f"image_{i}")
                    except Exception as e:
                        st.error(f"Image error: {e}")

        if img_to_edit:
            st.caption("Basic edit/variation (only if your deployment supports it).")
            if st.button("Try variation"):
                try:
                    b64in = base64.b64encode(img_to_edit.read()).decode("utf-8")
                    res = client.images.variations(
                        model=IMAGES_DEPLOYMENT,
                        image=b64in,
                        n=n_imgs,
                        size=size,
                        response_format="b64_json",
                    )
                    for i, d in enumerate(res.data, 1):
                        show_image_b64(d.b64_json, f"variation_{i}")
                except Exception as e:
                    st.error(f"Variation not supported by this deployment or failed: {e}")
