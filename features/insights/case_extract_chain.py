# features/insights/case_extract_chain.py
from __future__ import annotations
import re
from typing import Any, Dict, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from .case_schema import CaseFacts
from .case_prompts import SYSTEM_PROMPT, RULES_BLOCK
from utils.rag_env import get_chat_llm

def _clean_and_clip_text(text: str, max_chars: int = 120_000) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = text.strip()
    if len(text) > max_chars:
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        text = head + "\n\n...[TRUNCATED]...\n\n" + tail
    return text

def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user",
         "Extract fields from the following case-study content.\n\n"
         "CONTENT START\n{full_text}\nCONTENT END\n\n"
         "JSON SCHEMA:\n{schema_json}\n\n"
         "{rules}")
    ])

def _structured_llm(llm: AzureChatOpenAI | None = None) -> AzureChatOpenAI:
    # Force function_calling method to be Azure/vision-safe (json_schema not required)
    base = llm or get_chat_llm(temperature=0.0)
    return base.with_structured_output(CaseFacts, method="function_calling")

def extract_case_facts(full_text: str, llm_version: str | None = None) -> Tuple[CaseFacts, Dict[str, Any]]:
    cleaned = _clean_and_clip_text(full_text)
    prompt = _build_prompt()
    llm = _structured_llm()
    inputs: Dict[str, Any] = {
        "full_text": cleaned,
        "schema_json": CaseFacts.json_schema_str(),
        "rules": RULES_BLOCK,
    }
    chain = prompt | llm
    try:
        facts: CaseFacts = chain.invoke(inputs)
    except Exception:
        repair_inputs = {
            **inputs,
            "rules": (
                "Your previous output did not strictly match the schema. "
                "Return ONLY a single valid JSON object that conforms exactly to the schema.\n\n"
                + inputs["rules"]
            ),
        }
        facts = (prompt | llm).invoke(repair_inputs)

    raw_json: Dict[str, Any] = facts.model_dump()
    if llm_version:
        raw_json["_llm_version"] = llm_version
    return facts, raw_json
