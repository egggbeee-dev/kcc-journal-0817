# p2p_agent.py
#
# N-agent 정의 + PARTICIPATION(IN/OUT) 단계
#
#   1) build_agents()          : agent_id, zone_images(2~4장), persona로 Agent 리스트 생성
#   2) decide_participation()  : 전역 태스크를 브로드캐스트 → 각 agent가 독립적으로
#                                자기 zone 이미지만 보고 참여 여부 + verbalized confidence 산출
#   3) filter_active_agents()  : confidence threshold 적용 → 최종 active_agents 확정
#
# 설계 근거 (MetaCogAgent, arXiv:2605.17292):
#   "실행 전에 verbalized uncertainty를 산출하고, 확신도가 threshold 밑이면
#    해당 작업에서 제외한다" — 이 논문의 self-assessment 구조를 참여 판단에 적용.
#   단, LLM이 자기 능력을 과대평가하는 경향이 있다는 지적(Hitchhiker's Guide, 2606.24937)을
#   고려해 confidence threshold는 보수적으로 잡고, threshold 근처 애매한 경우는
#   기본적으로 IN 쪽에 편향되게 설계함 (거짓 OUT으로 협업 자체가 깨지는 게
#   거짓 IN으로 쓸모없는 로컬 플랜이 하나 섞이는 것보다 비용이 크다고 판단).

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from p2p_config import (
    MAX_AGENTS,
    MIN_ACTIVE_AGENTS,
    MIN_ZONE_IMAGES,
    MAX_ZONE_IMAGES,
    PARTICIPATION_CONFIDENCE_THRESHOLD,
    AGENT_STEP_STRIDE,
)
from p2p_utils import extract_json, _banner, _log
from p2p_vlm import run_vlm


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 정의
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Agent:
    agent_id:       str
    zone_images:    List[str]            # 이 agent가 볼 수 있는 사진 (2~4장)
    persona:        str = ""             # 선택: agent별 개별 역할 설명. 비어있으면 자동 생성
    common_persona: str = ""             # 전체 agent가 공유하는 공통 지침(협업 태도 등)
    index:          int = 0              # 생성 순서 (step_offset 계산용)

    @property
    def step_offset(self) -> int:
        return self.index * AGENT_STEP_STRIDE

    def default_persona(self) -> str:
        return (
            f"You are {self.agent_id}, an embodied agent responsible for the zone "
            f"shown in your assigned images."
        )

    @property
    def effective_persona(self) -> str:
        """common_persona(협업 지침) + 개별 persona(또는 자동 생성 기본값)를 합쳐서 반환."""
        individual = self.persona.strip() if self.persona.strip() else self.default_persona()
        common = self.common_persona.strip()
        return f"{individual}\n{common}" if common else individual


# 기본 공통 페르소나 — 명시적으로 다른 걸 안 주면 이걸 모든 agent에 적용
DEFAULT_COMMON_PERSONA = (
    "You are part of a team of collaborating agents working toward one shared "
    "global task. Prioritize the team's overall success over your own local plan: "
    "offer help proactively when you can, ask for what you need clearly, and avoid "
    "redundant or conflicting actions with the other agents. Assume the other "
    "agents are also acting in good faith to cooperate with you."
)


def build_agents(
    zone_image_map: Dict[str, List[str]],
    personas: Optional[Dict[str, str]] = None,
    common_persona: Optional[str] = None,
) -> List[Agent]:
    """
    zone_image_map : {"agent_A": [img1, img2, ...], "agent_B": [...], ...}
                      각 리스트는 2~4장 사이여야 함.
    personas       : {"agent_A": "너는 부엌 담당 로봇이다", ...} (선택, agent별 개별 지침)
    common_persona : 모든 agent에게 공통으로 적용할 협업 지침.
                      None이면 DEFAULT_COMMON_PERSONA 사용. 빈 문자열("")을 명시적으로
                      주면 공통 지침 없이 individual persona만 사용.

    2~8 에이전트를 지원. agent_id 순서는 dict 순서를 그대로 index로 사용
    (Python 3.7+ dict는 삽입 순서를 보존하므로 재현 가능).
    """
    personas = personas or {}
    resolved_common = DEFAULT_COMMON_PERSONA if common_persona is None else common_persona
    n = len(zone_image_map)
    if not (2 <= n <= MAX_AGENTS):
        raise ValueError(f"에이전트 수는 2~{MAX_AGENTS}개여야 합니다. 받은 값: {n}")

    agents: List[Agent] = []
    for i, (agent_id, imgs) in enumerate(zone_image_map.items()):
        if not (MIN_ZONE_IMAGES <= len(imgs) <= MAX_ZONE_IMAGES):
            raise ValueError(
                f"{agent_id}: zone_images는 {MIN_ZONE_IMAGES}~{MAX_ZONE_IMAGES}장이어야 합니다. "
                f"받은 값: {len(imgs)}"
            )
        agents.append(Agent(
            agent_id=agent_id,
            zone_images=list(imgs),
            persona=personas.get(agent_id, ""),
            common_persona=resolved_common,
            index=i,
        ))
    return agents


# ══════════════════════════════════════════════════════════════════════════════
# PARTICIPATION (IN/OUT) — 전역 태스크 브로드캐스트
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParticipationDecision:
    agent_id:     str
    participates: bool     # threshold 적용 후 최종 판단
    raw_vote:     bool     # 모델이 직접 답한 participates (threshold 적용 전)
    confidence:   int      # 0-100, verbalized confidence
    reason:       str = ""


_PARTICIPATION_EXAMPLE = """
EXAMPLE OUTPUT:
<JSON>
{
  "participates": true,
  "confidence": 85,
  "reason": "The kitchen zone contains the fridge and counter needed to prepare snacks for the task."
}
</JSON>
""".strip()


def _build_participation_prompt(task: str, persona: str) -> str:
    return f"""{persona}

Global task (shared with all agents): "{task}"

Look at your assigned zone images carefully.

Decide whether YOUR zone is relevant to completing this global task.

{_PARTICIPATION_EXAMPLE}

RULES:
1. participates: true if your zone contains objects/areas that could contribute
   to the task, even indirectly (e.g. providing an item, clearing a path).
   false only if your zone is clearly irrelevant.
2. confidence: 0-100, how certain you are about this decision. Base it on:
   - how clearly your zone relates to the task
   - how visible/identifiable the relevant objects are
   - whether you have enough information to be sure
3. reason: one concise sentence.
4. Return ONLY valid JSON inside <JSON> tags.

<JSON>
{{
  "participates": true,
  "confidence": 0,
  "reason": "..."
}}
</JSON>"""


def _parse_participation(raw: str, agent_id: str) -> ParticipationDecision:
    data = extract_json(raw)
    if not isinstance(data, dict):
        data = {}

    raw_vote = bool(data.get("participates", True))  # 파싱 실패 시 안전한 기본값: IN

    try:
        confidence = int(data.get("confidence", 50))
    except (TypeError, ValueError):
        confidence = 50
    confidence = max(0, min(100, confidence))

    reason = str(data.get("reason", "")).strip()

    return ParticipationDecision(
        agent_id=agent_id,
        participates=raw_vote,   # threshold는 filter_active_agents에서 별도 적용
        raw_vote=raw_vote,
        confidence=confidence,
        reason=reason,
    )


def decide_participation(agent: Agent, task: str) -> ParticipationDecision:
    """단일 agent의 참여 판단. zone_images 전체(2~4장)를 멀티이미지로 넘긴다."""
    prompt = _build_participation_prompt(task, agent.effective_persona)
    raw, _ = run_vlm(agent.zone_images, prompt)
    return _parse_participation(raw, agent.agent_id)


def decide_participation_all(agents: List[Agent], task: str) -> List[ParticipationDecision]:
    """모든 agent에게 전역 태스크를 '브로드캐스트'하고 병렬로 참여 판단을 받는다."""
    _banner("PARTICIPATION — GLOBAL TASK BROADCAST")
    print(f'  Task: "{task}"')
    print(f"  Broadcasting to {len(agents)} agents: {[a.agent_id for a in agents]}")

    with ThreadPoolExecutor(max_workers=len(agents)) as ex:
        futs = {ex.submit(decide_participation, a, task): a for a in agents}
        results = {futs[f].agent_id: f.result() for f in futs}

    # agent 원래 순서대로 정렬해서 반환 (ThreadPoolExecutor 완료 순서는 비결정적이므로)
    decisions = [results[a.agent_id] for a in agents]

    for d in decisions:
        vote_str = "IN " if d.raw_vote else "OUT"
        print(f"  [{d.agent_id}] vote={vote_str} confidence={d.confidence:>3d}  {d.reason}")

    return decisions


def filter_active_agents(
    agents: List[Agent],
    decisions: List[ParticipationDecision],
    confidence_threshold: int = PARTICIPATION_CONFIDENCE_THRESHOLD,
) -> List[Agent]:
    """
    최종 IN/OUT 확정 로직:
      - raw_vote=False 이고 confidence >= threshold  → 확실한 OUT
      - raw_vote=False 이고 confidence <  threshold  → 판단이 불확실하므로 안전하게 IN으로 편입
      - raw_vote=True                                 → confidence와 무관하게 IN
        (참여하겠다는 판단 자체를 뒤집을 근거는 없음. threshold는 "빠지겠다"는
         판단의 신뢰도를 검증하는 데만 사용— 거짓 OUT 방지가 목적)
    """
    active: List[Agent] = []
    dec_by_id = {d.agent_id: d for d in decisions}

    for agent in agents:
        d = dec_by_id[agent.agent_id]
        if d.raw_vote:
            final_in = True
        else:
            # OUT 판단인데 확신도가 낮으면 신뢰하지 말고 IN으로 되돌림
            final_in = d.confidence < confidence_threshold
            if final_in:
                print(
                    f"  [{agent.agent_id}] OUT 판단이었으나 confidence={d.confidence} "
                    f"< threshold={confidence_threshold} → IN으로 재편입"
                )
        d.participates = final_in
        if final_in:
            active.append(agent)

    print(f"\n  최종 active agents ({len(active)}/{len(agents)}): "
          f"{[a.agent_id for a in active]}")

    if len(active) < MIN_ACTIVE_AGENTS:
        raise RuntimeError(
            f"협업이 성립하려면 최소 {MIN_ACTIVE_AGENTS}개 agent가 참여해야 합니다. "
            f"현재 active: {len(active)}개 ({[a.agent_id for a in active]}). "
            f"태스크 또는 zone 이미지 구성을 확인하세요."
        )

    return active
