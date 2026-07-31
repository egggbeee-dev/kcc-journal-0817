# p2p_plan.py
#
# LOCAL PLAN 생성 (N-agent, 2~8)
#
#   - 각 agent는 자기 Offer + "다른 모든 agent들의 Offer 요약(can_provide/need_from_other)"을
#     보고 자기 zone에서 실행할 local plan을 생성 (병렬)
#   - 기존 2-agent phase2(coordinate)의 프롬프트/파싱/PASS-정규화 로직을 N-agent로 일반화
#   - target_agent는 이제 "agent_B 하나로 고정"이 아니라, 실제 active_agents 중에서
#     텍스트 매칭으로 해석해야 함 (_resolve_target_agent)

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

from agent import Agent
from config import AGENT_STEP_STRIDE, UNCERTAINTY_THRESH
from models import Handoff, HQEntry, LocalPlan, Offer, PlanStep
from offer import _is_passable, _kw, _vlm_with_retry
from utils import (
    _banner, _fuzzy_match_soft, _log, _norm_depends, _norm_handoff,
    clamp01, compute_plan_uncertainty, compute_token_uncertainty,
    extract_json, jdump, safe_int,
)


# ── target_agent 해석 (N-agent) ───────────────────────────────────────────────

def _resolve_target_agent(
    raw_target: Optional[str],
    my_agent_id: str,
    other_offers: Dict[str, Offer],
) -> Optional[str]:
    """
    target_agent 텍스트를 실제 active agent_id로 해석.
    1) 텍스트가 다른 agent_id를 직접 담고 있으면 그걸 사용
    2) 텍스트가 다른 agent의 room_type 키워드를 담고 있으면 그 agent로 매핑
    3) 매칭 실패 시 None (handoff 자체를 버림 — 잘못된 상대에게 보내는 것보다 안전)
    """
    if not raw_target:
        return None
    s = str(raw_target).strip().lower().replace("-", "_").replace(" ", "_")
    if not s or s in {"none", "null", "unknown"}:
        return None

    # 1) agent_id 직접 매칭 (자기 자신은 제외)
    for other_id in other_offers:
        if other_id.lower() == s and other_id != my_agent_id:
            return other_id

    # 2) room_type 키워드 매칭
    raw_lower = str(raw_target).strip().lower()
    for other_id, offer in other_offers.items():
        room = (offer.room_type or "").strip().lower()
        if room and room in raw_lower:
            return other_id

    # 3) 유일하게 남은 다른 agent가 하나뿐이면 그걸로 추론 (2-agent 케이스 하위호환)
    candidates = [oid for oid in other_offers if oid != my_agent_id]
    if len(candidates) == 1:
        return candidates[0]

    return None


# ── 프롬프트 ──────────────────────────────────────────────────────────────────

_PLAN_EXAMPLE = """
EXAMPLE — kitchen agent, with a bedroom agent and a living-room agent as teammates:
<JSON>
{
  "plan_steps": [
    {"step_id":1,"time_min":0,"action":"place apple and orange from island onto serving tray",
     "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,
     "uncertainty":0.1,"notes":""},
    {"step_id":2,"time_min":5,"action":"arrange bread from basket onto plate",
     "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,
     "uncertainty":0.1,"notes":""},
    {"step_id":3,"time_min":10,"action":"carry snack tray to kitchen doorway for living_room pickup",
     "preconditions":["snacks on tray"],"depends_on":[1,2],
     "handoff_type":"PASS","target_agent":"agent_C",
     "uncertainty":0.15,"notes":"snack tray ready at doorway"},
    {"step_id":4,"time_min":15,"action":"wipe counter surface with cloth",
     "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,
     "uncertainty":0.1,"notes":""}
  ]
}
</JSON>
""".strip()

_HANDOFF_RULES = """
HANDOFF RULES:

PASS — physical delivery to room boundary:
  USE WHEN: you physically carry an item to the doorway for a specific teammate.
  ACTION must start with: "carry" or "bring"
  target_agent MUST be one of the teammate agent_ids listed above (never yourself).
  CORRECT: {"action":"carry snack tray to doorway","handoff_type":"PASS",
             "target_agent":"agent_C","depends_on":[1,2]}
  WRONG: PASS on preparation steps (place, arrange, set up, organize)
  WRONG: PASS on non-physical items (sink, counter, status, confirmation)
  MAXIMUM: 1-2 PASS steps total. Only for items in your can_provide list.

INFORM — status notification (no physical movement):
  USE WHEN: you want to notify a specific teammate of completion.
  CORRECT: {"action":"notify agent_C: snacks are ready at doorway",
             "handoff_type":"INFORM","target_agent":"agent_C"}

KEY: "carry X to doorway" -> PASS | "notify <agent>" -> INFORM | all others -> null
""".strip()


def _build_plan_prompt(
    my_agent: Agent, my_offer: Offer, other_offers: Dict[str, Offer], task: str,
) -> str:
    teammates_ctx = "\n".join(
        f"- {oid} (room: {o.room_type}): can_provide={json.dumps(o.can_provide, ensure_ascii=False)}, "
        f"need_from_other={json.dumps(o.need_from_other, ensure_ascii=False)}"
        for oid, o in other_offers.items()
    )
    passable = [p for p in my_offer.can_provide if _is_passable(p)]

    return f"""{my_agent.effective_persona}
Global task: "{task}"

YOUR OFFER:
- room: {my_offer.room_type} ({my_agent.agent_id})
- can_provide (items you can PASS): {json.dumps(passable, ensure_ascii=False)}
- need_from_other: {json.dumps(my_offer.need_from_other, ensure_ascii=False)}

YOUR TEAMMATES (do not plan for them, just be aware):
{teammates_ctx}

{_PLAN_EXAMPLE}

{_HANDOFF_RULES}

Generate YOUR local plan. Think step by step:
1. What does the global task require from YOUR zone specifically?
2. What can you prepare for a teammate who needs it (see can_provide above)?
3. What do you need from a teammate, and can you tell from their can_provide who has it?

PLANNING RULES:
1. Steps ONLY in your zone ({my_offer.room_type}), using ONLY visible objects.
2. Generate 4-6 steps over 0-25 minutes. NO repeated actions.
3. Prioritize actions that DIRECTLY contribute to the global task.
4. HANDOFF - if can_provide is NOT empty:
   - Prepare the item first (1-2 prep steps)
   - Then add ONE PASS step naming the SPECIFIC teammate agent_id who needs it
   - PASS step must have depends_on=[prep step ids]
5. INFORM - if you want to notify a specific teammate of completion.
6. depends_on must ONLY reference your OWN step_ids (never a teammate's).
7. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "plan_steps": [
    {{"step_id":1,"time_min":0,"action":"verb + specific object",
      "preconditions":[],"depends_on":[],"handoff_type":null,
      "target_agent":null,"uncertainty":0.1,"notes":""}}
  ]
}}
</JSON>"""


# ── 파싱 ──────────────────────────────────────────────────────────────────────

def _parse_local_plan(
    raw: str,
    log_probs: List[float],
    my_agent: Agent,
    my_offer: Offer,
    other_offers: Dict[str, Offer],
) -> LocalPlan:
    step_offset = my_agent.step_offset
    own_lo, own_hi = step_offset, step_offset + AGENT_STEP_STRIDE - 1

    data = extract_json(raw)
    if isinstance(data, list):
        data = {"plan_steps": data}
    if not isinstance(data, dict):
        data = {}
    raw_steps = data.get("plan_steps", [])
    if not isinstance(raw_steps, list):
        raw_steps = []

    token_unc = compute_token_uncertainty(log_probs)
    steps:    List[PlanStep] = []
    hq_list:  List[HQEntry]  = []
    seen_ids: Set[int]       = set()
    seen_act: Set[frozenset] = set()

    for i, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        if not action:
            continue

        akey = frozenset(_kw(action))
        if akey and akey in seen_act:
            continue
        seen_act.add(akey)

        raw_sid  = safe_int(item.get("step_id", i), i)
        raw_time = safe_int(item.get("time_min", 0), 0)
        if raw_time > 25 and raw_time == raw_sid:
            raw_time = 0

        sid = raw_sid + step_offset
        while sid in seen_ids:
            sid += 1
        seen_ids.add(sid)

        json_unc    = clamp01(item.get("uncertainty", 0.2))
        action_conf = max(
            (v for k, v in my_offer.conf.items() if _fuzzy_match_soft(action, k)),
            default=0.7,
        )
        step_unc = clamp01(json_unc * 0.5 + token_unc * 0.2 + (1 - action_conf) * 0.3)

        raw_deps = _norm_depends(item.get("depends_on"))
        # own range 안에 들어오는 dep만 허용 — cross-agent 참조는 드롭
        deps = []
        for d in raw_deps:
            cand = d + step_offset
            if own_lo <= cand <= own_hi:
                deps.append(cand)

        handoff = _norm_handoff(item.get("handoff_type")) if item.get("handoff_type") else None
        target  = _resolve_target_agent(item.get("target_agent"), my_agent.agent_id, other_offers)
        if handoff and not target:
            # target을 해석 못 하면 handoff 자체를 버림 (잘못된 상대에게 보내는 것보다 안전)
            handoff = None

        first_word = action.lower().split()[0] if action.strip() else ""
        if handoff == "INFORM" and first_word in {"carry", "bring", "deliver", "transport"}:
            handoff = "PASS"

        step = PlanStep(
            step_id       = sid,
            time_min      = max(0, min(30, raw_time)),
            room          = my_offer.room_type,
            agent_id      = my_agent.agent_id,
            action        = action,
            preconditions = [str(x).strip() for x in item.get("preconditions", [])
                             if str(x).strip()],
            depends_on    = deps,
            handoff_type  = handoff,
            target_agent  = target,
            uncertainty   = step_unc,
            notes         = str(item.get("notes", "")).strip(),
        )
        steps.append(step)

        if step_unc >= UNCERTAINTY_THRESH:
            hq_list.append(HQEntry(sid, f"Is '{action}' feasible?", step_unc))

    steps.sort(key=lambda s: (s.time_min, s.step_id))
    steps = _normalize_pass(steps, valid_target_ids=set(other_offers.keys()))

    handoffs = [
        Handoff(s.step_id, s.action, s.handoff_type, s.target_agent,
                s.notes if s.handoff_type == "INFORM" else "",
                my_agent.agent_id)
        for s in steps if s.handoff_type
    ]

    all_unc = [s.uncertainty for s in steps] if steps else [token_unc]
    return LocalPlan(my_agent.agent_id, steps, compute_plan_uncertainty(all_unc), hq_list, handoffs)


def _normalize_pass(steps: List[PlanStep], valid_target_ids: Set[str]) -> List[PlanStep]:
    """비정상 PASS 제거 (N-agent: target은 valid_target_ids 중 하나여야 함)."""
    _CARRY = {"carry", "bring", "deliver", "transport", "move", "transfer"}
    _RECV  = {"place", "set", "organize", "receive", "pick", "get", "put", "sort"}

    for s in steps:
        if s.handoff_type != "PASS":
            continue

        first = s.action.lower().split()[0] if s.action.strip() else ""

        if not s.target_agent or s.target_agent not in valid_target_ids:
            print(f"  [NORM] step{s.step_id} PASS removed: no valid target_agent")
            s.handoff_type = None; s.target_agent = None; continue

        if first in _RECV:
            print(f"  [NORM] step{s.step_id} PASS removed: receiver verb '{first}'")
            s.handoff_type = None; s.target_agent = None; continue

        if not s.depends_on:
            prev_steps = [p for p in steps if p.step_id < s.step_id and not p.handoff_type]
            if prev_steps:
                s.depends_on = [max(prev_steps, key=lambda p: p.step_id).step_id]
                print(f"  [NORM] step{s.step_id} PASS: auto-linked depends_on={s.depends_on}")
            else:
                print(f"  [NORM] step{s.step_id} PASS removed: no depends_on")
                s.handoff_type = None; s.target_agent = None

    return steps


# ── 메인: N-agent Local Plan 생성 ─────────────────────────────────────────────

def generate_local_plans(
    active_agents: List[Agent],
    offers: Dict[str, Offer],
    task: str,
    verbose: str = "full",
) -> Dict[str, LocalPlan]:
    """
    각 agent가 (자기 Offer + 다른 모든 agent의 Offer 요약)을 보고
    local plan을 생성 (병렬). 반환: {agent_id: LocalPlan}
    """
    _banner(f"LOCAL PLANNING — {len(active_agents)} agents")

    def _one(agent: Agent) -> Tuple[str, LocalPlan]:
        my_offer = offers[agent.agent_id]
        other_offers = {aid: o for aid, o in offers.items() if aid != agent.agent_id}
        prompt = _build_plan_prompt(agent, my_offer, other_offers, task)
        raw, logp = _vlm_with_retry(agent.zone_images, prompt, True)
        if verbose == "full":
            _log(f"{agent.agent_id} PLAN RAW", raw)
        plan = _parse_local_plan(raw, logp, agent, my_offer, other_offers)
        return agent.agent_id, plan

    plans: Dict[str, LocalPlan] = {}
    with ThreadPoolExecutor(max_workers=len(active_agents)) as ex:
        futs = [ex.submit(_one, a) for a in active_agents]
        for f in futs:
            agent_id, plan = f.result()
            plans[agent_id] = plan

    for agent in active_agents:
        p = plans[agent.agent_id]
        n_pass = sum(1 for s in p.steps if s.handoff_type == "PASS")
        n_inform = sum(1 for s in p.steps if s.handoff_type == "INFORM")
        print(
            f"  [{agent.agent_id}] steps={len(p.steps)} U={p.U_plan:.3f} "
            f"PASS={n_pass} INFORM={n_inform}"
        )

    return plans
