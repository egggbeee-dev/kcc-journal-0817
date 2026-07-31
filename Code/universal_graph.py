# universal_graph.py
#
# 2안: Universal Graph — 전체 노드를 다 올리고, 매칭 + 컨플릭트를 하나의
# 워크리스트(큐)로 통합 처리한다. 필터링(협업 무관 agent 제외) 없음.
#
#   노드: Agent, Item(NEED/PROVIDE), Step  — 전부 필터링 없이 다 올림
#   엣지: MATCH 후보, HANDOFF 확정, DEPENDS_ON, CONFLICT(5종)
#   알고리즘: 매칭 대기 + 컨플릭트 대기를 큐 하나에서 순서대로 처리
#   마무리: Kahn's algorithm으로 사이클 체크 + 위상 정렬 → 최종 joint_plan
#
# 설계 원칙: LLM 호출 없음. 전부 텍스트 비교 / 정렬 / rule-based 로직.

from __future__ import annotations

import copy
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from config import FUZZY_STOPWORDS
from models import ConflictEntry, ConflictType, LocalPlan, Offer, PlanStep
from utils import extract_json, format_joint_plan
from vlm import run_vlm


# ══════════════════════════════════════════════════════════════════════════════
# 노드 정의
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ItemNode:
    item_id:   str
    agent_id:  str
    kind:      str   # "NEED" | "PROVIDE"
    text:      str
    urgency:   int = 3     # 1~5, Offer에 urgency 필드가 없으면 기본값 3(중간)
    consumed:  bool = False


@dataclass
class HandoffResult:
    need_agent:     str
    provide_agent:  str
    item_text:      str
    need_item_id:   str
    provide_item_id: str


def build_item_nodes(offers: Dict[str, Offer]) -> List[ItemNode]:
    """필터링 없이 모든 agent의 need/provide 텍스트를 전부 Item 노드로 만든다."""
    items: List[ItemNode] = []
    for agent_id, offer in offers.items():
        for i, text in enumerate(offer.need_from_other):
            items.append(ItemNode(f"{agent_id}_need_{i}", agent_id, "NEED", text))
        for i, text in enumerate(offer.can_provide):
            items.append(ItemNode(f"{agent_id}_provide_{i}", agent_id, "PROVIDE", text))
    return items


# ══════════════════════════════════════════════════════════════════════════════
# MATCH 후보 생성 — 노드 센트릭 (각 NEED 노드가 "자기 몫만" 스스로 계산)
#
#   중앙 조정자가 전체 need+provide를 한 번에 모아서 계산하는 게 아니라,
#   NEED 노드 하나하나가 독립적으로 "자기 need 텍스트 + 전체 provide 목록"을
#   놓고 스스로 후보를 판단한다 (GraphAgent-Reasoner의 노드 센트릭 분해와 동일한
#   원리). 계산 주체가 항상 "그 need를 가진 agent 자신"이므로 중앙집중이 아님.
#   need 노드 수만큼 개별 호출(병렬) — 배치 1회로 전체를 계산하지 않음.
#
#   텍스트만 오가므로 저비용. run_vlm은 이미지 인자를 요구하는 인터페이스라서,
#   그 need를 가진 agent 자신의 zone 이미지 1장을 "앵커"로 재사용한다(추론 자체는
#   순수 텍스트 기반이며, 이미지 내용은 이 판단에 실제로 사용되지 않는다).
# ══════════════════════════════════════════════════════════════════════════════

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


_MATCH_PROMPT_TEMPLATE = """You are {agent_id}. You stated this need: "{need_text}"

Below is the full list of items other agents have offered to provide
(you already know this — offers were broadcast to everyone):
{candidates_str}

Which of these items could satisfy your need? Consider substitutes and
related items, not just literal word matches — for example, a "vegetable
tray" could satisfy a need for "snacks", or a "flashlight" could satisfy
a need described as "light source".

Return ONLY valid JSON:
<JSON>
{{"candidates": [{{"item_id": "...", "reason": "one short phrase"}}]}}
</JSON>
If nothing plausibly fits, return {{"candidates": []}}."""


def _build_match_prompt(agent_id: str, need_text: str, provide_items: List[ItemNode]) -> str:
    candidates_str = "\n".join(
        f'- item_id="{p.item_id}" (from {p.agent_id}): "{p.text}"' for p in provide_items
    )
    return _MATCH_PROMPT_TEMPLATE.format(
        agent_id=agent_id, need_text=need_text, candidates_str=candidates_str,
    )


def _anchor_image(agent_id: str, agent_zone_images: Dict[str, List[str]]) -> List[str]:
    """이 need-agent 자신의 zone 이미지 1장(없으면 다른 agent 것이라도)을 앵커로."""
    imgs = agent_zone_images.get(agent_id) or []
    if imgs:
        return [imgs[0]]
    for other_imgs in agent_zone_images.values():
        if other_imgs:
            return [other_imgs[0]]
    return []


def compute_match_candidates(
    items: List[ItemNode],
    agent_zone_images: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    need_item_id -> [provide_item_id, ...]
    각 NEED 노드가 독립적으로(병렬) 자기 몫만 계산. 노드 수만큼 개별 호출.
    """
    needs = [it for it in items if it.kind == "NEED"]
    provides = [it for it in items if it.kind == "PROVIDE"]

    def _one(need: ItemNode) -> Tuple[str, List[str]]:
        others = [p for p in provides if p.agent_id != need.agent_id]
        if not others:
            return need.item_id, []
        prompt = _build_match_prompt(need.agent_id, need.text, others)
        anchor = _anchor_image(need.agent_id, agent_zone_images)
        raw, _ = run_vlm(anchor, prompt)
        data = extract_json(raw)
        valid_ids = {p.item_id for p in others}
        cand_ids: List[str] = []
        if isinstance(data, dict):
            for c in data.get("candidates", []):
                if isinstance(c, dict) and c.get("item_id") in valid_ids:
                    cand_ids.append(c["item_id"])
        return need.item_id, cand_ids

    candidates: Dict[str, List[str]] = {}
    if not needs:
        return candidates
    with ThreadPoolExecutor(max_workers=len(needs)) as ex:
        futs = [ex.submit(_one, n) for n in needs]
        for f in futs:
            nid, cids = f.result()
            candidates[nid] = cids
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# room 거리 (기본: 같은 room=0, 다르면 1. adjacency map 있으면 그걸로 대체)
# ══════════════════════════════════════════════════════════════════════════════

def room_distance(
    room_a: str, room_b: str,
    adjacency: Optional[Dict[Tuple[str, str], int]] = None,
) -> int:
    ra, rb = (room_a or "").strip().lower(), (room_b or "").strip().lower()
    if ra == rb:
        return 0
    if adjacency:
        return adjacency.get((ra, rb), adjacency.get((rb, ra), 1))
    return 1


# ══════════════════════════════════════════════════════════════════════════════
# 매칭 확정 (우선순위: urgency desc → 후보 수 asc, 확정 시 거리 오름차순으로 pick)
# ══════════════════════════════════════════════════════════════════════════════

def resolve_matches(
    items: List[ItemNode],
    candidates: Dict[str, List[str]],
    offers: Dict[str, Offer],
    room_adjacency: Optional[Dict[Tuple[str, str], int]] = None,
) -> Tuple[List[HandoffResult], List[str]]:
    items_by_id = {it.item_id: it for it in items}
    needs = [it for it in items if it.kind == "NEED"]

    # urgency 내림차순, 후보 수 오름차순 정렬
    ordered = sorted(
        needs,
        key=lambda n: (-n.urgency, len(candidates.get(n.item_id, []))),
    )

    handoffs: List[HandoffResult] = []
    unresolved: List[str] = []

    for need in ordered:
        if need.consumed:
            continue
        cand_ids = [cid for cid in candidates.get(need.item_id, [])
                    if not items_by_id[cid].consumed]
        if not cand_ids:
            unresolved.append(need.item_id)
            continue

        need_room = offers[need.agent_id].room_type
        cand_ids.sort(key=lambda cid: room_distance(
            need_room, offers[items_by_id[cid].agent_id].room_type, room_adjacency,
        ))
        chosen = items_by_id[cand_ids[0]]
        chosen.consumed = True
        need.consumed = True
        handoffs.append(HandoffResult(
            need_agent=need.agent_id,
            provide_agent=chosen.agent_id,
            item_text=chosen.text,
            need_item_id=need.item_id,
            provide_item_id=chosen.item_id,
        ))

    return handoffs, unresolved


# ══════════════════════════════════════════════════════════════════════════════
# 키워드 유틸 (rule-based 체크용 — fuzzy 유사도 계산 아님, 단순 집합 겹침만 봄)
# ══════════════════════════════════════════════════════════════════════════════

def _kw(text: str) -> Set[str]:
    return set(re.findall(r"\w+", (text or "").lower())) - FUZZY_STOPWORDS


_RECEIVE_VERBS = {"receive", "get", "take", "pick", "accept"}
_SEND_VERBS    = {"carry", "bring", "deliver", "transport", "pass"}


def _find_receive_step(plan: LocalPlan, item_text: str) -> Optional[PlanStep]:
    item_kw = _kw(item_text)
    for s in plan.steps:
        first = s.action.lower().split()[0] if s.action.strip() else ""
        if first in _RECEIVE_VERBS and (item_kw & _kw(s.action)):
            return s
    return None


def _find_send_step(plan: LocalPlan, item_text: str) -> Optional[PlanStep]:
    item_kw = _kw(item_text)
    for s in plan.steps:
        first = s.action.lower().split()[0] if s.action.strip() else ""
        if first in _SEND_VERBS and (item_kw & _kw(s.action)):
            return s
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CONFLICT 탐지 (5종, 전부 rule-based, 전체 agent 쌍 대상 — 필터링 없음)
# ══════════════════════════════════════════════════════════════════════════════

_TEMPORAL_WINDOW_MIN = 3


def detect_conflicts(
    plans: Dict[str, LocalPlan],
    offers: Dict[str, Offer],
    only_step_ids: Optional[Set[int]] = None,
) -> List[ConflictEntry]:
    conflicts: List[ConflictEntry] = []
    all_steps: List[PlanStep] = [s for p in plans.values() for s in p.steps]

    def _touches(*step_ids: int) -> bool:
        return only_step_ids is None or any(sid in only_step_ids for sid in step_ids)

    # (a) TEMPORAL — 다른 agent, 같은 room, 시간 겹침
    for i, s1 in enumerate(all_steps):
        for s2 in all_steps[i + 1:]:
            if s1.agent_id == s2.agent_id:
                continue
            if not _touches(s1.step_id, s2.step_id):
                continue
            if s1.room and s1.room.strip().lower() == (s2.room or "").strip().lower():
                if abs(s1.time_min - s2.time_min) < _TEMPORAL_WINDOW_MIN:
                    conflicts.append(ConflictEntry(
                        ConflictType.TEMPORAL, [s1.step_id, s2.step_id],
                        [s1.agent_id, s2.agent_id],
                        f"same room '{s1.room}', time overlap ({s1.time_min}m vs {s2.time_min}m)",
                        "shift the later step's time_min later",
                    ))

    # (c) REDUNDANCY — 다른 agent끼리 같은 행동 완전 일치
    seen: Dict[str, PlanStep] = {}
    for s in all_steps:
        key = _norm_text(s.action)
        if key in seen and seen[key].agent_id != s.agent_id:
            if _touches(s.step_id, seen[key].step_id):
                later = s if s.step_id > seen[key].step_id else seen[key]
                conflicts.append(ConflictEntry(
                    ConflictType.REDUNDANCY, [seen[key].step_id, s.step_id],
                    [seen[key].agent_id, s.agent_id],
                    f"duplicate action '{s.action}' across agents",
                    f"remove step {later.step_id}",
                ))
        else:
            seen[key] = s

    # (d) CANNOT_DO — 자기 혼자 체크 (자동 삽입된 receive 스텝은 제외 — 관찰 못 한
    #     아이템을 핸드오프로 받은 것이므로 cannot_do/관찰범위 체크 대상이 아님)
    for s in all_steps:
        if not _touches(s.step_id):
            continue
        if "auto-inserted" in (s.notes or ""):
            continue
        offer = offers.get(s.agent_id)
        if not offer:
            continue
        for cd in offer.cannot_do:
            if _norm_text(cd.action) == _norm_text(s.action) or (
                _kw(cd.action) and _kw(cd.action) <= _kw(s.action)
            ):
                conflicts.append(ConflictEntry(
                    ConflictType.CANNOT_DO, [s.step_id], [s.agent_id],
                    f"agent marked cannot_do: '{cd.action}' (reason={cd.reason})",
                    f"remove step {s.step_id}",
                ))
                break

    # (e) OBSERVABILITY — 자기 혼자 체크
    for s in all_steps:
        if not _touches(s.step_id):
            continue
        if s.handoff_type:  # PASS/INFORM 스텝은 관찰 범위 체크 제외
            continue
        if "auto-inserted" in (s.notes or ""):  # 핸드오프로 받은 아이템 사용 스텝도 제외
            continue
        offer = offers.get(s.agent_id)
        if not offer:
            continue
        pool = _kw(offer.obs_scope) | set().union(*(_kw(cd_) for cd_ in offer.can_do)) if offer.can_do else _kw(offer.obs_scope)
        kw = _kw(s.action)
        if kw and pool and not (kw & pool):
            conflicts.append(ConflictEntry(
                ConflictType.OBSERV, [s.step_id], [s.agent_id],
                f"action '{s.action}' references objects outside observed scope",
                f"remove step {s.step_id}",
            ))

    return conflicts


# ══════════════════════════════════════════════════════════════════════════════
# CONFLICT 해결 (전부 rule-based)
# ══════════════════════════════════════════════════════════════════════════════

def resolve_conflict(conflict: ConflictEntry, plans: Dict[str, LocalPlan]) -> Set[int]:
    """conflict를 plans에 반영해서 수정. 영향받은 step_id 집합을 반환(재검사용)."""
    affected: Set[int] = set()

    if conflict.conflict_type == ConflictType.TEMPORAL:
        s1_id, s2_id = conflict.step_ids
        later_id = max(s1_id, s2_id)
        for plan in plans.values():
            for s in plan.steps:
                if s.step_id == later_id:
                    other_time = next(
                        (o.time_min for p2 in plans.values() for o in p2.steps
                         if o.step_id == min(s1_id, s2_id)), s.time_min
                    )
                    s.time_min = max(s.time_min, other_time + _TEMPORAL_WINDOW_MIN)
                    affected.add(s.step_id)

    elif conflict.conflict_type in (ConflictType.REDUNDANCY, ConflictType.CANNOT_DO, ConflictType.OBSERV):
        remove_id = conflict.step_ids[-1] if conflict.conflict_type == ConflictType.REDUNDANCY else conflict.step_ids[0]
        for plan in plans.values():
            before = len(plan.steps)
            plan.steps = [s for s in plan.steps if s.step_id != remove_id]
            if len(plan.steps) != before:
                affected.add(remove_id)

    return affected


def check_and_insert_receive_step(
    handoff: HandoffResult, plans: Dict[str, LocalPlan],
) -> Tuple[Optional[ConflictEntry], Set[int]]:
    """
    HANDOFF 확정 후: 받는 쪽 plan에 대응하는 receive 스텝이 있는지 확인.
    없으면 자동으로 추가(DEPENDENCY 해결). 보내는 쪽 스텝이 없으면
    (자동 생성이 위험하므로) 해결하지 않고 conflict만 기록.
    """
    need_plan = plans[handoff.need_agent]
    provide_plan = plans[handoff.provide_agent]
    affected: Set[int] = set()

    send_step = _find_send_step(provide_plan, handoff.item_text)
    if send_step is None:
        return ConflictEntry(
            ConflictType.DEPENDENCY, [], [handoff.provide_agent],
            f"{handoff.provide_agent} confirmed to provide '{handoff.item_text}' "
            f"but has no send step — needs human review",
            "manual review required (not auto-resolved)",
        ), affected

    recv_step = _find_receive_step(need_plan, handoff.item_text)
    if recv_step is None:
        new_id = max([s.step_id for s in need_plan.steps], default=need_plan.steps[0].step_id if need_plan.steps else 0) + 1
        new_step = PlanStep(
            step_id=new_id,
            time_min=send_step.time_min + 2,
            room=next((o.room for p in plans.values() for o in p.steps if o.agent_id == handoff.need_agent), ""),
            agent_id=handoff.need_agent,
            action=f"receive {handoff.item_text}",
            depends_on=[],
            handoff_type=None,
            target_agent=None,
            uncertainty=0.2,
            notes="auto-inserted by conflict check (DEPENDENCY)",
        )
        # room은 need_plan의 기존 스텝에서 가져옴 (없으면 빈 문자열)
        if need_plan.steps:
            new_step.room = need_plan.steps[0].room
        need_plan.steps.append(new_step)
        need_plan.steps.sort(key=lambda s: (s.time_min, s.step_id))
        affected.add(new_id)
        recv_step = new_step

    # 순서 충돌 체크: 이 아이템을 실제로 "쓰는" 다른 스텝이 받기 전(time_min <=)에
    # 와 있으면, 그 스텝을 받은 이후로 미루고 depends_on을 걸어 순서를 강제함
    item_kw = _kw(handoff.item_text)
    for s in need_plan.steps:
        if s.step_id == recv_step.step_id:
            continue
        first = s.action.lower().split()[0] if s.action.strip() else ""
        if first in _RECEIVE_VERBS:
            continue  # 다른 receive 스텝은 대상 아님
        if item_kw & _kw(s.action) and s.time_min <= recv_step.time_min:
            s.time_min = recv_step.time_min + 2
            if recv_step.step_id not in s.depends_on:
                s.depends_on = list(s.depends_on) + [recv_step.step_id]
            affected.add(s.step_id)
    need_plan.steps.sort(key=lambda s: (s.time_min, s.step_id))

    return None, affected


# ══════════════════════════════════════════════════════════════════════════════
# Kahn's Algorithm — 사이클 체크 + 위상 정렬
# ══════════════════════════════════════════════════════════════════════════════

def kahn_topological_order(plans: Dict[str, LocalPlan]) -> Tuple[List[int], bool]:
    """
    모든 step의 DEPENDS_ON 엣지(같은 agent 안)를 모아 위상 정렬.
    반환: (정렬된 step_id 리스트, 사이클 없음 여부)
    """
    all_steps = {s.step_id: s for p in plans.values() for s in p.steps}
    in_degree = {sid: 0 for sid in all_steps}
    adj: Dict[int, List[int]] = {sid: [] for sid in all_steps}

    for sid, s in all_steps.items():
        for dep in s.depends_on:
            if dep in all_steps:
                adj[dep].append(sid)
                in_degree[sid] += 1

    queue = deque(sorted(sid for sid, d in in_degree.items() if d == 0))
    order: List[int] = []
    while queue:
        # 같은 in-degree=0 그룹 안에서는 time_min 순으로 처리해 결정론적으로
        queue = deque(sorted(queue, key=lambda sid: all_steps[sid].time_min))
        sid = queue.popleft()
        order.append(sid)
        for nxt in adj[sid]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    no_cycle = len(order) == len(all_steps)
    if not no_cycle:
        # 사이클이 남은 step들은 그냥 time_min 순으로 뒤에 이어붙임 (안전 fallback)
        remaining = [sid for sid in all_steps if sid not in order]
        remaining.sort(key=lambda sid: all_steps[sid].time_min)
        order.extend(remaining)

    return order, no_cycle


def merge_joint_plan(plans: Dict[str, LocalPlan]) -> List[dict]:
    order, _ = kahn_topological_order(plans)
    all_steps = {s.step_id: s for p in plans.values() for s in p.steps}
    ordered_steps = [all_steps[sid] for sid in order]
    # 위상순 유지하되, 동일 위상 레벨 안에서는 이미 time_min 정렬돼 있음
    return [
        {
            "step_id": s.step_id, "time_min": s.time_min, "room": s.room,
            "agent_id": s.agent_id, "action": s.action, "depends_on": s.depends_on,
            "handoff_type": s.handoff_type, "target_agent": s.target_agent,
            "notes": s.notes,
        }
        for s in ordered_steps
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 메인: 하나의 워크리스트로 매칭 + 컨플릭트 통합 처리
# ══════════════════════════════════════════════════════════════════════════════

def run(
    active_agents,
    offers: Dict[str, Offer],
    plans: Dict[str, LocalPlan],
    task: str = "",
    room_adjacency: Optional[Dict[Tuple[str, str], int]] = None,
    max_rounds: int = 10,
) -> dict:
    plans = copy.deepcopy(plans)  # 원본 보존

    # active_agents가 Agent 객체 리스트면 zone_images를 뽑아서 앵커 이미지로 사용.
    # (하위호환: agent_id 문자열 리스트가 들어오면 zone_images 없이 진행 — 이 경우
    #  텍스트-only 호출을 지원 안 하는 백엔드(Qwen)에서는 매칭이 실패할 수 있음)
    agent_zone_images: Dict[str, List[str]] = {}
    for a in active_agents:
        aid = getattr(a, "agent_id", a)
        imgs = getattr(a, "zone_images", None)
        if imgs:
            agent_zone_images[aid] = imgs

    # ── 1) 매칭 (노드 센트릭: 각 NEED 노드가 자기 몫만 독립적으로 계산) ────────
    items = build_item_nodes(offers)
    candidates = compute_match_candidates(items, agent_zone_images)
    handoffs, unresolved_needs = resolve_matches(items, candidates, offers, room_adjacency)

    print(f"  [MATCH] {len(handoffs)}개 확정, {len(unresolved_needs)}개 미해결")
    for h in handoffs:
        print(f"    {h.provide_agent} → {h.need_agent} : {h.item_text}")

    # ── 2) HANDOFF 확정 건마다 받는 스텝 유무 확인 + 자동 보완 ─────────────────
    dependency_conflicts: List[ConflictEntry] = []
    for h in handoffs:
        conflict, _affected = check_and_insert_receive_step(h, plans)
        if conflict:
            dependency_conflicts.append(conflict)

    # ── 3) 전체 컨플릭트 스캔 + 워크리스트 처리 ─────────────────────────────
    conflict_queue = deque(detect_conflicts(plans, offers))
    resolved: List[ConflictEntry] = []
    rounds = 0
    while conflict_queue and rounds < max_rounds:
        rounds += 1
        c = conflict_queue.popleft()
        affected = resolve_conflict(c, plans)
        resolved.append(c)
        if affected:
            new_conflicts = detect_conflicts(plans, offers, only_step_ids=affected)
            for nc in new_conflicts:
                if nc not in resolved:
                    conflict_queue.append(nc)

    print(f"  [CONFLICT] {len(resolved)}개 자동 해결, {len(dependency_conflicts)}개 미해결(DEPENDENCY)")

    # ── 4) Merge (Kahn's algorithm) ───────────────────────────────────────────
    joint_plan = merge_joint_plan(plans)

    return {
        "updated_plans": plans,
        "handoffs": handoffs,
        "conflicts_resolved": resolved,
        "conflicts_unresolved": dependency_conflicts + [
            ConflictEntry(ConflictType.DEPENDENCY, [], [], f"unmatched need item: {nid}", "no provide found")
            for nid in unresolved_needs
        ],
        "joint_plan": joint_plan,
        "joint_plan_text": format_joint_plan(joint_plan),
    }
