"""LLM 호출과 구조화 출력 헬퍼.

생성기와 채점기를 나눈다. 같은 모델이 쓰고 같은 모델이 채점하면 자기 글을
너그럽게 본다. 채점에는 한 등급 위 모델을 쓰고, 그마저도 수치 검증에는
쓰지 않는다. 앞선 실험에서 LLM 충실성 채점기가 틀린 숫자 23건 중 2건만,
근거 채점기는 21건 중 1건만 잡아냈다. 숫자는 코드로 대조한다.
"""
from __future__ import annotations

import json
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import config

_cache: dict[tuple, ChatOpenAI] = {}
USAGE = {"calls": 0, "in": 0, "out": 0}


def chat(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    model = model or config.GENERATOR
    key = (model, temperature)
    if key not in _cache:
        _cache[key] = ChatOpenAI(model=model, temperature=temperature, timeout=120, max_retries=3)
    return _cache[key]


def ask(system: str, user: str, model: str | None = None,
        temperature: float = 0.0) -> str:
    msgs = [SystemMessage(content=system), HumanMessage(content=user)]
    for attempt in range(3):
        try:
            r = chat(model, temperature).invoke(msgs)
            USAGE["calls"] += 1
            meta = getattr(r, "usage_metadata", None) or {}
            USAGE["in"] += meta.get("input_tokens", 0)
            USAGE["out"] += meta.get("output_tokens", 0)
            return r.content if isinstance(r.content, str) else str(r.content)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return ""


def ask_json(system: str, user: str, model: str | None = None,
             temperature: float = 0.0, retries: int = 2) -> dict | list:
    """JSON을 받아 낸다. 파싱에 실패하면 오류를 보여 주고 다시 시킨다."""
    sys_json = system + "\n\n반드시 JSON 하나만 출력한다. 설명, 머리말, 코드 울타리를 붙이지 않는다."
    payload = user
    last = ""
    for _ in range(retries + 1):
        last = ask(sys_json, payload, model, temperature)
        try:
            return json.loads(_strip_fence(last))
        except json.JSONDecodeError as e:
            payload = (f"{user}\n\n직전 출력이 JSON으로 파싱되지 않았다: {e}\n"
                       f"받은 출력의 앞부분: {last[:300]}\nJSON만 다시 출력한다.")
    raise ValueError(f"JSON 파싱 실패. 마지막 출력: {last[:500]}")


def _strip_fence(t: str) -> str:
    t = t.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.S)
    return m.group(1) if m else t


def yes_no(system: str, user: str, model: str | None = None) -> bool:
    out = ask(system + "\n\nyes 또는 no 한 단어로만 답한다.", user, model).strip().lower()
    return out.startswith("y")
