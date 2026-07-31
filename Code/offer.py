# p2p_offer.py
#
# OBSERVATION → OFFER 생성 (N-agent, 2~8)
#
#   - 기존 p2p_phase.py의 phase1_offer(2-agent 전용)를 active_agents 리스트 기반으로 일반화
#   - 각 agent는 자기 zone_images(2~4장)만 보고 독립적으로 관찰 → Offer 생성 (병렬)
#   - Offer 내용(can_do/cannot_do/can_provide/need_from_other)은 기존 스키마 그대로 유지

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Tuple

from agent import Agent
from config import MAX_CAN_DO, MAX_CANNOT_DO, NON_PASSABLE_KW
from models import CannotEntry, Offer
from utils import (
    _banner, _fuzzy_match, _log, _match_conf, _norm_reason, clamp01, extract_json,
)
from vlm import run_vlm


# ── 키워드 유틸 (can_provide 물리적 전달 가능 여부 판단용) ────────────────────

def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _kw(text: str) -> Set[str]:
    from config import FUZZY_STOPWORDS
    return {_stem(w) for w in set(re.findall(r"\w+", text.lower())) - FUZZY_STOPWORDS}


def _is_passable(item: str) -> bool:
    """물리적으로 들고 이동 가능한 아이템인지 판단."""
    return not bool(_kw(item) & NON_PASSABLE_KW)


# ── 프롬프트 ──────────────────────────────────────────────────────────────────

_OFFER_EXAMPLE = """
EXAMPLE — kitchen-zone agent, task "prepare movie night":
<JSON>
{
  "room_type": "kitchen",
  "observation": "Kitchen with fruits on island, bread basket, countertops, sink.",
  "obs_scope": "island, counter, shelf, sink, stove, fruits, bread basket",
  "can_do": [
    "place apple and orange from island onto serving tray",
    "arrange bread from basket onto plate",
    "fill water glass from tap",
    "wipe counter surface with cloth",
    "clean visible sink with sponge"
  ],
  "cannot_do": [
    {"action": "arrange living room seating", "reason": "NO_OBJECT"},
    {"action": "adjust TV lighting", "reason": "NO_OBJECT"}
  ],
  "conf": {
    "place apple and orange from island onto serving tray": 0.9,
    "arrange bread from basket onto plate": 0.85,
    "fill water glass from tap": 0.9,
    "wipe counter surface with cloth": 0.95,
    "clean visible sink with sponge": 0.9
  },
  "can_provide": ["snack tray with fruits and bread"],
  "need_from_other": ["living room table cleared for snacks"]
}
</JSON>
""".strip()


def _build_offer_prompt(task: str, persona: str, n_other_agents: int) -> str:
    return f"""{persona}

Global task (shared with all agents): "{task}"

You are one of {n_other_agents + 1} collaborating agents. You do not yet know
what the other agents' zones look like — observe only what YOU can see.

{_OFFER_EXAMPLE}

Generate your Offer for YOUR zone only. Be faithful to what is actually visible.

RULES:
1. can_do: max {MAX_CAN_DO} actions using ONLY visible objects.
   - Prioritize actions that DIRECTLY contribute to the global task.
   - Format: "verb + specific visible object + purpose"
2. cannot_do: max {MAX_CANNOT_DO}. reason: NO_OBJECT | NO_CAPABILITY | UNCERTAIN
3. conf: confidence [0.0-1.0] per can_do item.
4. can_provide: items you can PHYSICALLY CARRY to another agent's zone boundary.
   - ONLY tangible objects: food tray, drink, meal, document, tool
   - NOT: "cleaned sink", "confirmation", "status", "organized shelf"
   - Keep to 1-2 items maximum. Only what another agent might actually need.
5. need_from_other: 1-2 things you genuinely need from ANY other agent to complete
   the task. Focus on physical items or critical information, not generic
   confirmations. You don't know who can provide it yet - just state the need.
6. Think about COLLABORATION: what can you prepare that helps the team?
7. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "room_type": "...",
  "observation": "one concise sentence describing the zone",
  "obs_scope": "comma-separated list of visible objects and areas",
  "can_do": ["verb + specific object + purpose"],
  "cannot_do": [{{"action": "...", "reason": "NO_OBJECT"}}],
  "conf": {{"action text": 0.9}},
  "can_provide": ["max 2 tangible items for other agents"],
  "need_from_other": ["max 2 specific needs"]
}}
</JSON>"""


# ── 거절 감지 + 재시도 ────────────────────────────────────────────────────────

_REFUSAL_PHRASES = [
    "i'm sorry", "i cannot", "i can't", "i apologize",
    "as an ai", "not able to", "unable to",
]


def _is_vlm_refusal(raw: str) -> bool:
    lower = raw.strip().lower()
    has_json = "<json>" in lower or "{" in lower
    if has_json:
        return False
    return any(p in lower for p in _REFUSAL_PHRASES)


def _vlm_with_retry(
    image_paths: List[str], prompt: str, return_logprobs: bool = False,
    max_retries: int = 2,
) -> Tuple[str, List[float]]:
    raw, logp = "", []
    for attempt in range(max_retries + 1):
        raw, logp = run_vlm(image_paths, prompt, return_logprobs)
        if not _is_vlm_refusal(raw):
            return raw, logp
        print(f"  [RETRY] VLM refusal detected (attempt {attempt+1}/{max_retries+1}), retrying...")
    print(f"  [WARN] VLM still refusing after {max_retries} retries, using empty response.")
    return raw, logp


# ── 파싱 ──────────────────────────────────────────────────────────────────────

def _parse_offer(raw: str, agent_id: str) -> Offer:
    data = extract_json(raw)
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        data = {}

    cannot_do: List[CannotEntry] = []
    uncertain_count = 0
    for item in data.get("cannot_do", [])[:MAX_CANNOT_DO]:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        reason = _norm_reason(item.get("reason", "UNCERTAIN"))
        if action:
            if reason == "UNCERTAIN":
                uncertain_count += 1
            cannot_do.append(CannotEntry(action, reason))

    cannot_set = {c.action.lower() for c in cannot_do}
    seen: Set[str] = set()
    can_do: List[str] = []
    for x in data.get("can_do", []):
        a = str(x).strip()
        if not a or a.lower() in seen:
            continue
        if any(_fuzzy_match(a, c, min_overlap=2) for c in cannot_set):
            continue
        seen.add(a.lower())
        can_do.append(a)
        if len(can_do) >= MAX_CAN_DO:
            break

    raw_scope = data.get("obs_scope", "")
    obs_scope = (
        ", ".join(str(x).strip() for x in raw_scope)
        if isinstance(raw_scope, list)
        else str(raw_scope).strip()
    )

    conf_raw = {str(k): clamp01(v) for k, v in data.get("conf", {}).items()}

    raw_provides = [str(x).strip() for x in data.get("can_provide", []) if str(x).strip()]
    can_provide  = [p for p in raw_provides if _is_passable(p)]
    filtered     = [p for p in raw_provides if not _is_passable(p)]
    if filtered:
        print(f"  [OFFER][{agent_id}] non-passable items filtered from can_provide: {filtered}")

    return Offer(
        agent_id        = agent_id,
        room_type       = str(data.get("room_type", "")).strip(),
        observation     = str(data.get("observation", "")).strip(),
        obs_scope       = obs_scope,
        can_do          = can_do,
        cannot_do       = cannot_do,
        conf            = _match_conf(conf_raw, can_do),
        can_provide     = can_provide,
        need_from_other = [str(x).strip() for x in data.get("need_from_other", [])
                           if str(x).strip()],
        uncertain_count = uncertain_count,
    )


# ── 메인: N-agent Offer 생성 ──────────────────────────────────────────────────

def generate_offers(
    active_agents: List[Agent],
    task: str,
    verbose: str = "full",
) -> Dict[str, Offer]:
    """
    active_agents 각각에 대해 독립적으로 관찰 → Offer 생성 (병렬).
    반환: {agent_id: Offer}
    """
    _banner(f"OBSERVATION & OFFER GENERATION — {len(active_agents)} agents")

    def _one(agent: Agent) -> Tuple[str, Offer]:
        prompt = _build_offer_prompt(
            task, agent.effective_persona, n_other_agents=len(active_agents) - 1,
        )
        raw, _ = _vlm_with_retry(agent.zone_images, prompt, False)
        if verbose == "full":
            _log(f"{agent.agent_id} OFFER RAW", raw)
        offer = _parse_offer(raw, agent.agent_id)
        return agent.agent_id, offer

    offers: Dict[str, Offer] = {}
    with ThreadPoolExecutor(max_workers=len(active_agents)) as ex:
        futs = [ex.submit(_one, a) for a in active_agents]
        for f in futs:
            agent_id, offer = f.result()
            offers[agent_id] = offer

    # agent 원래 순서대로 정렬해서 요약 출력
    for agent in active_agents:
        o = offers[agent.agent_id]
        print(
            f"  [{agent.agent_id}] room={o.room_type!r} "
            f"can_do={len(o.can_do)} cannot_do={len(o.cannot_do)} "
            f"can_provide={o.can_provide} need_from_other={o.need_from_other}"
        )

    return offers
