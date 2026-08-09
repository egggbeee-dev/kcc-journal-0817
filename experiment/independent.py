# independent.py
#
# INDEPENDENT 베이스라인 — P2P(offer.py+p2p_plan.py), Centralized(commander.py)와
# 비교하기 위한 세 번째 대조군.
#
#   Centralized : 참여 여부도 agent0 혼자 결정, 플랜도 agent0 지시를 그대로 따름
#   P2P         : 참여는 각자 스스로 판단, 플랜은 Offer를 교환하며 negotiation
#   Independent : 참여는 각자 스스로 판단(P2P와 동일하게 agent.py의 self-assessment
#                 재사용) 하지만, 로컬 플랜을 세울 때는 다른 agent가 존재하는지조차
#                 모르는 것처럼 순수하게 자기 zone 이미지 + 전역 태스크만 보고 계획한다.
#                 소통이 안 될 뿐이지 행동 자체는 자유롭다 — "요리해서 방 밖으로
#                 내놓는다" 같은 행동은 agent 자기 판단으로 얼마든지 계획할 수 있다.
#                 다만 그걸 누가 받는지, 언제 받는지, 실제로 닿는지를 확정해줄
#                 메커니즘(Offer 매칭이든 agent0 지시든)이 전혀 없기 때문에,
#                 파싱 단계에서 handoff_type/target_agent는 항상 None으로 남고
#                 결과적으로 최종 플랜에서는 아무에게도 연결되지 않은 채 남는다 —
#                 이게 바로 "소통 없이 일할 때의 낭비/단절"을 보여주는 지점이다.
#
# 실행 순서: decide_participation_all -> filter_active_agents (agent.py 재사용)
#          -> generate_independent_plans -> universal_graph.merge_joint_plan
#          (Kahn's algorithm 위상정렬 + cycle 검증/해소, P2P/Centralized와 공유)
#
# handoff가 파싱 단계에서 항상 연결 안 된 채로 남으므로(위 설명 참고) 이 파일
# 자체에서 별도의 missing-receive 처리는 하지 않는다 — 애초에 확정된 handoff가
#없으니 처리할 대상 자체가 없음.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from agent import Agent, decide_participation_all, filter_active_agents
from config import AGENT_STEP_STRIDE, UNCERTAINTY_THRESH
from models import HQEntry, LocalPlan, PlanStep
from universal_graph import merge_joint_plan
from utils import (
    _banner, _log, _norm_depends, clamp01, compute_plan_uncertainty,
    compute_token_uncertainty, extract_json, format_joint_plan, safe_int,
)
from vlm import run_vlm


# ── 프롬프트 ──────────────────────────────────────────────────────────────────

_PLAN_EXAMPLE = """
EXAMPLE — kitchen agent, working alone:
<JSON>
{
  "plan_steps": [
    {"step_id":1,"time_min":0,"action":"place apple and orange from island onto serving tray",
     "preconditions":[],"depends_on":[],"uncertainty":0.1,"notes":""},
    {"step_id":2,"time_min":5,"action":"arrange bread from basket onto plate",
     "preconditions":[],"depends_on":[],"uncertainty":0.1,"notes":""},
    {"step_id":3,"time_min":10,"action":"carry snack tray out of the kitchen toward where guests will likely gather",
     "preconditions":["snack tray ready"],"depends_on":[1,2],"uncertainty":0.2,
     "notes":"acting on own judgment — no way to confirm anyone will receive it"},
    {"step_id":4,"time_min":15,"action":"wipe counter surface with cloth",
     "preconditions":[],"depends_on":[],"uncertainty":0.1,"notes":""}
  ]
}
</JSON>
""".strip()


def _build_independent_plan_prompt(agent: Agent, task: str) -> str:
    return f"""{agent.effective_persona}

Global task: "{task}"

You are working COMPLETELY ALONE in your zone. You do not know whether other
agents exist, and even if they do, you cannot communicate with them in any
way: you cannot ask them for anything, cannot confirm they will receive
anything from you, and cannot wait for them to be ready.

{_PLAN_EXAMPLE}

PLANNING RULES:
1. Steps ONLY in your zone, using ONLY visible objects.
2. Generate 4-6 steps over 0-25 minutes. NO repeated actions.
3. Prioritize actions that DIRECTLY contribute to the global task, using only
   what your zone already has — do NOT assume any item will arrive from
   elsewhere, and do NOT wait for anyone else to be ready.
4. depends_on must ONLY reference your OWN step_ids.
5. If moving something out of your zone genuinely helps the task (e.g.
   carrying prepared food toward where it's likely needed), you MAY plan
   that — but you cannot name who receives it, cannot confirm it arrives,
   and cannot coordinate timing with anyone. It's purely your own best
   guess, acted on independently.
6. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "plan_steps": [
    {{"step_id":1,"time_min":0,"action":"verb + specific object",
      "preconditions":[],"depends_on":[],"uncertainty":0.1,"notes":""}}
  ]
}}
</JSON>"""


# ── 파싱 ──────────────────────────────────────────────────────────────────────

def _parse_independent_plan(raw: str, log_probs: List[float], agent: Agent) -> LocalPlan:
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
    steps:   List[PlanStep] = []
    hq_list: List[HQEntry]  = []
    seen_ids = set()

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

        step = PlanStep(
            step_id       = sid,
            time_min      = max(0, min(30, raw_time)),
            room          = agent.agent_id,
            agent_id      = agent.agent_id,
            action        = action,
            preconditions = [str(x).strip() for x in item.get("preconditions", []) if str(x).strip()],
            depends_on    = deps,
            handoff_type  = None,   # Independent는 매칭 메커니즘이 없어 항상 연결 안 됨
                                     # (agent가 "무언가를 옮긴다"는 행동 자체는 할 수 있지만,
                                     #  그게 누구에게 어떻게 닿는지 확정해줄 주체가 없음 —
                                     #  그래서 최종 플랜에서는 연결 없는 일반 스텝으로 남는다)
            target_agent  = None,
            uncertainty   = step_unc,
            notes         = str(item.get("notes", "")).strip(),
        )
        steps.append(step)

        if step_unc >= UNCERTAINTY_THRESH:
            hq_list.append(HQEntry(sid, f"Is '{action}' feasible?", step_unc))

    steps.sort(key=lambda s: (s.time_min, s.step_id))
    all_unc = [s.uncertainty for s in steps] if steps else [token_unc]
    return LocalPlan(agent.agent_id, steps, compute_plan_uncertainty(all_unc), hq_list, handoffs=[])


# ── 메인: N-agent Independent Local Plan 생성 ─────────────────────────────────

def generate_independent_plans(
    active_agents: List[Agent], task: str, verbose: str = "full",
) -> Dict[str, LocalPlan]:
    """
    각 active agent가 다른 agent의 존재를 모르는 것처럼, 자기 zone 이미지와
    전역 태스크만 보고 완전히 독립적으로 로컬 플랜을 생성한다 (병렬).
    Offer도, handoff도, 다른 agent에 대한 어떤 정보도 주어지지 않는다.
    """
    _banner(f"INDEPENDENT LOCAL PLANNING — {len(active_agents)} agents")

    def _one(agent: Agent) -> Tuple[str, LocalPlan]:
        prompt = _build_independent_plan_prompt(agent, task)
        raw, logp = run_vlm(agent.zone_images, prompt, True)
        if verbose == "full":
            _log(f"{agent.agent_id} INDEPENDENT PLAN RAW", raw)
        plan = _parse_independent_plan(raw, logp, agent)
        return agent.agent_id, plan

    plans: Dict[str, LocalPlan] = {}
    with ThreadPoolExecutor(max_workers=len(active_agents)) as ex:
        futs = [ex.submit(_one, a) for a in active_agents]
        for f in futs:
            agent_id, plan = f.result()
            plans[agent_id] = plan

    for agent in active_agents:
        p = plans[agent.agent_id]
        print(f"  [{agent.agent_id}] steps={len(p.steps)} U={p.U_plan:.3f}")

    return plans


def run_independent(agents: List[Agent], task: str, verbose: str = "full"):
    """
    decide_participation_all -> filter_active_agents (agent.py 재사용, P2P와
    동일한 self-assessment 기반 참여 판단) -> generate_independent_plans
    -> universal_graph.merge_joint_plan (Kahn's algorithm 위상정렬 + cycle
    검증/해소를 P2P/Centralized와 동일한 로직으로 공유) 까지 한 번에 실행한다.

    Independent는 handoff가 구조적으로 연결되는 일이 없으므로(파싱 단계에서
    handoff_type이 항상 None) cycle이 걸릴 일은 사실상 없지만, 세 방법론의
    최종 정렬 단계를 동일한 코드로 맞추기 위해 그대로 재사용한다.

    반환값: (joint_plan, active_agents, plans)
    """
    decisions = decide_participation_all(agents, task)
    active_agents = filter_active_agents(agents, decisions)

    plans = generate_independent_plans(active_agents, task, verbose)

    joint_plan, broken_cycle_edges = merge_joint_plan(plans)
    if broken_cycle_edges:
        print(f"  [MERGE] 사이클 감지 → {len(broken_cycle_edges)}개 엣지 자동 해제: {broken_cycle_edges}")

    print(format_joint_plan(joint_plan))
    return joint_plan, active_agents, plans
