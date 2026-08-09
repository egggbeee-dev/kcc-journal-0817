# commander.py
#
# CENTRALIZED(agent0 통솔) 베이스라인 — P2P(offer.py + p2p_plan.py) 방법론과
# 비교하기 위한 대조군.
#
#   agent0(commander)이 전체를 통솔하는 중앙집권 구조:
#     - agent0은 물리적 zone이 없는 순수 조율자. 각 subordinate의 관찰 보고서
#       (텍스트)만 받아서 판단하며, 방을 직접 보지 않는다.
#     - "이번 태스크에 참여할 agent를 선별"하는 것도 agent0 혼자 결정한다
#       (P2P처럼 각 agent가 스스로 참여 여부를 판단하지 않음).
#     - agent0은 참여가 확정된 agent마다 자연어 지시(directive)를 내리고,
#       subordinate는 그 지시를 그대로 따라 자기 zone 이미지를 보고
#       구체적인 로컬 플랜(step-by-step)을 만든다.
#     - subordinate끼리는 서로 통신할 수 없고, Offer/can_provide/negotiation
#       같은 것도 전혀 없다. handoff(물건 전달)가 필요하면 agent0이 directive에서
#       "carry X to <agent_id>" 식으로 직접 지정하고, subordinate는 그 지시에
#       있는 경우에만 PASS 스텝을 만들 수 있다 (스스로 판단해서 만드는 것 금지).
#
# 실행 순서: observe_subordinates -> commander_decide -> filter_active_subordinates
#          -> generate_subordinate_plans -> ensure_receive_steps (Centralized 전용)
#          -> universal_graph.merge_joint_plan (Kahn's algorithm 위상정렬 +
#             cycle 검증/해소, P2P/Independent와 동일한 로직 공유)
#
# 반환되는 joint_plan은 P2P/Independent 파이프라인의 결과물과 동일한 포맷
# (utils.format_joint_plan 사용) — 세 방법론을 나란히 비교할 때 그대로 쓸 수 있다.

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple

from agent import Agent
from config import AGENT_STEP_STRIDE, MIN_ACTIVE_AGENTS, UNCERTAINTY_THRESH
try:
    from config import PASS_SEND_VERBS  # 최신(GitHub) config.py에만 있음
except ImportError:
    # Drive에 아직 옛날 config.py가 있는 경우를 위한 폴백 (실제 레포 값과 동일)
    PASS_SEND_VERBS = {"carry", "bring", "deliver", "transport", "pass"}
from models import Handoff, HQEntry, LocalPlan, PlanStep
from universal_graph import merge_joint_plan
from utils import (
    _banner, _log, _norm_depends, _norm_handoff, clamp01,
    compute_plan_uncertainty, compute_token_uncertainty, extract_json,
    format_joint_plan, safe_int,
)
from vlm import run_vlm

DEFAULT_COMMANDER_PERSONA = (
    "You are agent0, the sole commander of a household robot team. You do not "
    "have a physical zone of your own — you coordinate everything through the "
    "subordinate agents' reports and your own directives."
)


# ══════════════════════════════════════════════════════════════════════════
# STAGE A — 각 subordinate zone 관찰 (자연어, 병렬)
# ══════════════════════════════════════════════════════════════════════════

def _build_observation_prompt(task: str) -> str:
    return f"""Look at this image carefully.

Task: "{task}"

Describe the following in natural language:
1. What room/zone is this?
2. What objects and areas do you see?
3. What actions could be done here to help with the task?

Be specific about visible objects. Keep it concise (3-5 sentences)."""


def observe_subordinates(subordinates: List[Agent], task: str) -> Dict[str, str]:
    """
    각 subordinate zone을 병렬로 관찰해 자연어 요약을 얻는다.
    이 요약이 agent0에게 올라가는 '보고서' 역할을 한다 — agent0은 방을
    직접 보지 않고 이 텍스트만으로 판단한다.
    """
    _banner(f"STAGE A — SUBORDINATE OBSERVATION ({len(subordinates)} agents)")
    prompt = _build_observation_prompt(task)

    def _one(agent: Agent) -> Tuple[str, str]:
        raw, _ = run_vlm(agent.zone_images, prompt, False)
        return agent.agent_id, raw

    obs: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(subordinates)) as ex:
        futs = [ex.submit(_one, a) for a in subordinates]
        for f in futs:
            agent_id, text = f.result()
            obs[agent_id] = text

    for agent in subordinates:
        _log(f"{agent.agent_id} REPORT TO AGENT0", obs[agent.agent_id])

    return obs


# ══════════════════════════════════════════════════════════════════════════
# STAGE B — COMMANDER: 참여 agent 선별 + 전체 directive 생성 (단일 호출)
# ══════════════════════════════════════════════════════════════════════════

_COMMANDER_EXAMPLE = """
EXAMPLE — commander overseeing kitchen, living_room, bathroom, bedroom:
<JSON>
{
  "kitchen":     {"participates": true,  "reason": "has food/drink prep items", "directive": "Prepare a snack tray and coffee, then carry both to the living_room doorway for handoff."},
  "living_room": {"participates": true,  "reason": "main space for guests",     "directive": "Clear the coffee table and arrange seating for 5 guests. Receive the snack tray and coffee pot from kitchen and place them on the coffee table."},
  "bathroom":    {"participates": true,  "reason": "guests will use it",        "directive": "Set out fresh towels and make sure the sink area is presentable."},
  "bedroom":     {"participates": false, "reason": "not relevant to guest reception", "directive": ""}
}
</JSON>
""".strip()


def _build_commander_prompt(
    task: str, commander_persona: str, observations: Dict[str, str],
) -> str:
    reports = "\n\n".join(f"[{aid} report]\n{text}" for aid, text in observations.items())
    return f"""{commander_persona}

You are the sole decision-maker. Subordinates cannot talk to each other and
have no visibility into any zone but their own — they only see what YOU tell
them via your directive.

Global task: "{task}"

SUBORDINATE REPORTS:
{reports}

{_COMMANDER_EXAMPLE}

For EACH subordinate listed above, decide:
1. participates: true/false — should this zone be used for this task at all?
   YOU make this call alone, not the subordinate.
2. reason: one short sentence.
3. directive: if participates=true, a concise natural-language instruction
   telling that subordinate EXACTLY what to accomplish in its own zone,
   including whether it must hand an item to another zone (carry X to
   <agent_id>) or is expecting to receive one (receive X from <agent_id>).
   If participates=false, use an empty string "".

RULES:
- Only assign a physical handoff (carry/receive) between two zones that BOTH participate.
- Keep each directive to 1-3 sentences.
- Return ONLY valid JSON inside <JSON> tags, one key per subordinate agent_id
  (use the EXACT agent_ids shown in the reports above, nothing else).

<JSON>
{{
  "<agent_id>": {{"participates": true, "reason": "...", "directive": "..."}}
}}
</JSON>"""


@dataclass
class CommanderDecision:
    agent_id:     str
    participates: bool
    reason:       str
    directive:    str


def commander_decide(
    subordinates: List[Agent],
    observations: Dict[str, str],
    task: str,
    commander_persona: str = DEFAULT_COMMANDER_PERSONA,
) -> Dict[str, CommanderDecision]:
    """agent0이 모든 report를 한 번에 보고, 참여 여부 + directive를 단일 호출로 결정."""
    _banner("STAGE B — COMMANDER SELECTION & DIRECTIVE (single call)")
    prompt = _build_commander_prompt(task, commander_persona, observations)

    # run_vlm은 이미지 인자가 필수라 대표로 첫 subordinate의 이미지를 동반한다.
    # agent0의 실제 판단 근거는 이미지가 아니라 위 텍스트 report이다.
    representative_images = subordinates[0].zone_images
    raw, _ = run_vlm(representative_images, prompt, False)
    _log("AGENT0 RAW DECISION", raw)

    data = extract_json(raw)
    if not isinstance(data, dict):
        data = {}

    decisions: Dict[str, CommanderDecision] = {}
    for agent in subordinates:
        item = data.get(agent.agent_id, {})
        if not isinstance(item, dict):
            item = {}
        participates = bool(item.get("participates", True))  # 파싱 실패 시 안전하게 IN
        reason       = str(item.get("reason", "")).strip()
        directive    = str(item.get("directive", "")).strip() if participates else ""
        decisions[agent.agent_id] = CommanderDecision(agent.agent_id, participates, reason, directive)

    for d in decisions.values():
        vote_str = "IN " if d.participates else "OUT"
        print(f"  [agent0 -> {d.agent_id}] {vote_str}  {d.reason}")
        if d.directive:
            print(f"      directive: {d.directive}")

    return decisions


def filter_active_subordinates(
    subordinates: List[Agent], decisions: Dict[str, CommanderDecision],
) -> List[Agent]:
    """
    agent0의 최종 판단을 그대로 신뢰한다. P2P(filter_active_agents)와 달리
    여기는 판단 주체가 agent0 하나뿐이므로, confidence 기반으로 뒤집을 근거가 없다.
    """
    active = [a for a in subordinates if decisions[a.agent_id].participates]
    print(f"\n  최종 active subordinates ({len(active)}/{len(subordinates)}): "
          f"{[a.agent_id for a in active]}")

    if len(active) < MIN_ACTIVE_AGENTS:
        raise RuntimeError(
            f"협업이 성립하려면 최소 {MIN_ACTIVE_AGENTS}개 agent가 참여해야 합니다. "
            f"현재 active: {len(active)}개 ({[a.agent_id for a in active]})."
        )
    return active


# ══════════════════════════════════════════════════════════════════════════
# STAGE C — SUBORDINATE: agent0의 directive를 그대로 따라 로컬 플랜 생성 (병렬)
# ══════════════════════════════════════════════════════════════════════════

_SUBORDINATE_EXAMPLE = """
EXAMPLE — kitchen subordinate, directive: "Prepare a snack tray and coffee,
then carry both to the living_room doorway for handoff.":
<JSON>
{
  "plan_steps": [
    {"step_id":1,"time_min":0,"action":"arrange fruits and bread on serving tray",
     "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,
     "uncertainty":0.1,"notes":""},
    {"step_id":2,"time_min":5,"action":"brew coffee in coffee maker",
     "preconditions":[],"depends_on":[],"handoff_type":null,"target_agent":null,
     "uncertainty":0.1,"notes":""},
    {"step_id":3,"time_min":10,"action":"carry snack tray and coffee to living_room doorway",
     "preconditions":["tray ready","coffee brewed"],"depends_on":[1,2],
     "handoff_type":"PASS","target_agent":"living_room",
     "uncertainty":0.15,"notes":"handoff as directed by agent0"}
  ]
}
</JSON>
""".strip()


def _build_subordinate_prompt(
    agent: Agent, directive: str, task: str, valid_targets: List[str],
) -> str:
    return f"""{agent.effective_persona}

You are {agent.agent_id}, a subordinate agent. You CANNOT communicate with any
other subordinate and cannot see any zone but your own. You must follow
agent0's directive exactly — do not negotiate, and do not decide on your own
what to hand off or request from anyone.

Global task: "{task}"

AGENT0'S DIRECTIVE FOR YOU:
"{directive}"

{_SUBORDINATE_EXAMPLE}

RULES:
1. Steps ONLY in your own zone, using ONLY objects visible in your images.
2. Generate 3-6 steps over 0-25 minutes that carry out the directive above.
3. handoff_type/target_agent: ONLY set these if the directive explicitly tells
   you to carry something to, or receive something from, another agent.
   - "carry X to <agent_id>" -> handoff_type="PASS", target_agent="<agent_id>"
     (target_agent MUST be exactly one of: {valid_targets})
   - otherwise leave handoff_type null and target_agent null.
4. depends_on must ONLY reference your OWN step_ids.
5. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "plan_steps": [
    {{"step_id":1,"time_min":0,"action":"verb + specific object",
      "preconditions":[],"depends_on":[],"handoff_type":null,
      "target_agent":null,"uncertainty":0.1,"notes":""}}
  ]
}}
</JSON>"""


def _parse_subordinate_plan(
    raw: str, log_probs: List[float], agent: Agent, valid_targets: List[str],
) -> LocalPlan:
    step_offset = agent.step_offset
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
    seen_ids  = set()

    for i, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        if not action:
            continue

        raw_sid  = safe_int(item.get("step_id", i), i)
        raw_time = safe_int(item.get("time_min", 0), 0)
        sid = raw_sid + step_offset
        while sid in seen_ids:
            sid += 1
        seen_ids.add(sid)

        json_unc = clamp01(item.get("uncertainty", 0.2))
        step_unc = clamp01(json_unc * 0.6 + token_unc * 0.4)

        raw_deps = _norm_depends(item.get("depends_on"))
        deps = [d + step_offset for d in raw_deps if own_lo <= d + step_offset <= own_hi]

        handoff = _norm_handoff(item.get("handoff_type")) if item.get("handoff_type") else None
        target  = str(item.get("target_agent", "") or "").strip() or None
        if handoff and target not in valid_targets:
            # agent0가 지시하지 않은 대상으로의 handoff는 subordinate의 월권 —무효 처리
            handoff, target = None, None
        if not handoff:
            target = None

        step = PlanStep(
            step_id       = sid,
            time_min      = max(0, min(30, raw_time)),
            room          = agent.agent_id,
            agent_id      = agent.agent_id,
            action        = action,
            preconditions = [str(x).strip() for x in item.get("preconditions", []) if str(x).strip()],
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

    handoffs = [
        Handoff(s.step_id, s.action, s.handoff_type, s.target_agent, "", agent.agent_id)
        for s in steps if s.handoff_type
    ]
    all_unc = [s.uncertainty for s in steps] if steps else [token_unc]
    return LocalPlan(agent.agent_id, steps, compute_plan_uncertainty(all_unc), hq_list, handoffs)


def generate_subordinate_plans(
    active_subordinates: List[Agent],
    decisions: Dict[str, CommanderDecision],
    task: str,
    verbose: str = "full",
) -> Dict[str, LocalPlan]:
    """
    agent0의 directive를 받은 각 subordinate가 자기 zone 이미지를 보고
    그 지시를 그대로 실행하는 로컬 플랜을 생성 (병렬, 서로 소통 없음, negotiation 없음).
    """
    _banner(f"STAGE C — SUBORDINATE LOCAL PLANNING ({len(active_subordinates)} agents)")
    valid_targets = [a.agent_id for a in active_subordinates]

    def _one(agent: Agent) -> Tuple[str, LocalPlan]:
        directive = decisions[agent.agent_id].directive
        prompt = _build_subordinate_prompt(agent, directive, task, valid_targets)
        raw, logp = run_vlm(agent.zone_images, prompt, True)
        if verbose == "full":
            _log(f"{agent.agent_id} SUBORDINATE PLAN RAW", raw)
        plan = _parse_subordinate_plan(raw, logp, agent, valid_targets)
        return agent.agent_id, plan

    plans: Dict[str, LocalPlan] = {}
    with ThreadPoolExecutor(max_workers=len(active_subordinates)) as ex:
        futs = [ex.submit(_one, a) for a in active_subordinates]
        for f in futs:
            agent_id, plan = f.result()
            plans[agent_id] = plan

    for agent in active_subordinates:
        p = plans[agent.agent_id]
        n_pass = sum(1 for s in p.steps if s.handoff_type == "PASS")
        print(f"  [{agent.agent_id}] steps={len(p.steps)} U={p.U_plan:.3f} PASS={n_pass}")

    return plans


# ══════════════════════════════════════════════════════════════════════════
# STAGE D — MISSING RECEIVE 처리 (Centralized 전용, Offer/auction 없이 동작)
# ══════════════════════════════════════════════════════════════════════════
#
# universal_graph.py의 resolve_missing_receive는 auction의 MatchAssignment를
# 입력으로 받는 구조라 Offer가 없는 Centralized에서는 그대로 재사용할 수 없다.
# agent0이 directive에서 이미 발신자/수신자를 직접 지정했으므로, 여기서는
# "선언된 PASS인데 대상 agent 쪽에 그걸 받는 스텝이 없으면 삽입"이라는
# 단순한 규칙만 있으면 충분하다.

_ITEM_RE = re.compile(r"^(?:" + "|".join(PASS_SEND_VERBS) + r")\s+(.*?)\s+to\b", re.IGNORECASE)


def _extract_item(action: str) -> str:
    m = _ITEM_RE.match(action.strip())
    return m.group(1).strip() if m else "item"


def ensure_receive_steps(plans: Dict[str, LocalPlan]) -> List[int]:
    """
    agent0이 지시한 PASS 스텝인데도 대상 agent의 로컬 플랜에 대응하는 receive
    스텝(depends_on으로 그 PASS를 참조하는 스텝)이 없으면 자동으로 삽입한다.
    반환: 새로 삽입된 step_id 리스트.
    """
    inserted: List[int] = []
    for plan in plans.values():
        for step in list(plan.steps):
            if step.handoff_type != "PASS" or not step.target_agent:
                continue
            target_plan = plans.get(step.target_agent)
            if target_plan is None:
                continue
            already_received = any(step.step_id in s.depends_on for s in target_plan.steps)
            if already_received:
                continue

            item = _extract_item(step.action)
            new_id = max([s.step_id for s in target_plan.steps], default=step.step_id) + 1
            recv_step = PlanStep(
                step_id       = new_id,
                time_min      = step.time_min + 2,
                room          = target_plan.steps[0].room if target_plan.steps else step.target_agent,
                agent_id      = step.target_agent,
                action        = f"receive {item}",
                depends_on    = [step.step_id],
                handoff_type  = None,
                target_agent  = None,
                uncertainty   = 0.2,
                notes         = "auto-inserted (Centralized: declared PASS had no matching receive step)",
            )
            target_plan.steps.append(recv_step)
            target_plan.steps.sort(key=lambda s: (s.time_min, s.step_id))
            inserted.append(new_id)
            print(f"    [RESOLVE] step{step.step_id}(PASS→{step.target_agent})에 대응하는 "
                  f"receive 스텝 자동 삽입: step{new_id} 'receive {item}'")
    return inserted


# ══════════════════════════════════════════════════════════════════════════
# 메인: CENTRALIZED(agent0 통솔) 파이프라인 전체 실행
# ══════════════════════════════════════════════════════════════════════════

def run_centralized_commander(
    subordinates: List[Agent],
    task: str,
    commander_persona: str = DEFAULT_COMMANDER_PERSONA,
    verbose: str = "full",
):
    """
    observe -> commander_decide -> filter_active_subordinates
    -> generate_subordinate_plans -> ensure_receive_steps (Centralized 전용
    missing-receive 처리) -> universal_graph.merge_joint_plan (Kahn's algorithm
    위상정렬 + cycle 검증/해소, P2P/Independent와 동일한 로직 공유) 까지
    한 번에 실행한다.

    반환값: (joint_plan, decisions, plans)
    """
    observations = observe_subordinates(subordinates, task)
    decisions = commander_decide(subordinates, observations, task, commander_persona)
    active_subordinates = filter_active_subordinates(subordinates, decisions)
    plans = generate_subordinate_plans(active_subordinates, decisions, task, verbose)

    ensure_receive_steps(plans)

    joint_plan, broken_cycle_edges = merge_joint_plan(plans)
    if broken_cycle_edges:
        print(f"  [MERGE] 사이클 감지 → {len(broken_cycle_edges)}개 엣지 자동 해제: {broken_cycle_edges}")

    print(format_joint_plan(joint_plan))
    return joint_plan, decisions, plans
