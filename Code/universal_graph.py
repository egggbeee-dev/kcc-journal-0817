# universal_graph.py  (v2 — embedding + Hungarian + edge-walk conflict check)
#
# 2안: Universal Graph — 전체 노드를 다 올리고, 매칭 + 컨플릭트를 그래프
# 자료구조 위에서 처리한다. 필터링(협업 무관 agent 제외) 없음. LLM 호출 없음.
#
#   노드: Item(NEED/PROVIDE), Step  — 전부 필터링 없이 다 올림
#   엣지:
#     - MATCH 후보  : NEED ↔ (PROVIDE 아이템 ∪ 다른 agent의 Step)
#                     가중치 = 텍스트 임베딩 코사인 유사도 (LLM 아님, 로컬 계산)
#     - HANDOFF     : MATCH가 PROVIDE 아이템으로 확정된 경우 (물건 전달)
#     - STATE_DEPENDENCY : MATCH가 다른 agent의 Step으로 확정된 경우
#                          (물건이 아니라 "상태"를 필요로 하는 need — 예:
#                          "dining area cleared"는 물건이 아니라 다른 agent의
#                          행동으로 이미 충족되는 상태임)
#     - SAME_ROOM   : 다른 agent의 Step끼리 같은 room이면 미리 그어둠
#                     (TEMPORAL/REDUNDANCY 체크가 이 엣지를 "순회"하도록)
#     - DEPENDS_ON  : 같은 agent 안(원래) + STATE_DEPENDENCY로 추가되는
#                     cross-agent 순서 제약
#
#   알고리즘:
#     - 매칭: NEED × (PROVIDE∪STEP) 가중치 행렬 → Hungarian Algorithm으로
#             전역 최적 1:1 배정 (scipy.optimize.linear_sum_assignment)
#     - Conflict: SAME_ROOM 엣지를 미리 만들어두고, 그 엣지를 순회하며 체크.
#                 문제 수정 후에는 전체 재스캔이 아니라 영향받은 노드의
#                 "이웃 엣지만" 다시 확인 (1-hop 전파)
#     - Merge: Kahn's algorithm (기존과 동일, DEPENDS_ON 전체를 대상으로
#              사이클 체크 + 위상 정렬 — cross-agent depends_on도 그대로 지원)
#
# 설계 원칙: 텍스트 임베딩(사전학습 모델, 로컬 계산)만 사용. LLM 호출 없음.
#           sentence-transformers가 없으면 TF-IDF 코사인 유사도로 자동 대체
#           (약한 대체재 — 실제 의미 유사도 포착력은 sentence-transformers보다
#           떨어짐. Colab 등 인터넷 되는 환경에서는 sentence-transformers 설치
#           권장: `pip install sentence-transformers`).

from __future__ import annotations

import copy
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import FUZZY_STOPWORDS
from models import ConflictEntry, ConflictType, LocalPlan, Offer, PlanStep
from utils import format_joint_plan


# ══════════════════════════════════════════════════════════════════════════════
# 임베딩 (LLM 아님 — 사전학습 로컬 모델. 없으면 TF-IDF로 자동 대체)
# ══════════════════════════════════════════════════════════════════════════════

_embed_backend = None  # ("st", model) | ("tfidf", None)


def _get_embed_backend():
    global _embed_backend
    if _embed_backend is not None:
        return _embed_backend
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        _embed_backend = ("st", model)
        print("  [EMBED] sentence-transformers(all-MiniLM-L6-v2) 사용")
    except Exception:
        _embed_backend = ("tfidf", None)
        print("  [EMBED] sentence-transformers 없음 → TF-IDF 코사인 유사도로 대체 "
              "(의미 유사도 포착력이 약함. `pip install sentence-transformers` 권장)")
    return _embed_backend


def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if w not in FUZZY_STOPWORDS]


def _tfidf_vectors(texts: List[str]) -> np.ndarray:
    """아주 가벼운 TF-IDF (외부 의존성 없음, 이 호출 안의 텍스트들 안에서만 정의)."""
    docs = [_tokenize(t) for t in texts]
    vocab = sorted({w for d in docs for w in d})
    if not vocab:
        return np.zeros((len(texts), 1))
    idx = {w: i for i, w in enumerate(vocab)}
    df = np.zeros(len(vocab))
    for d in docs:
        for w in set(d):
            df[idx[w]] += 1
    idf = np.log((len(docs) + 1) / (df + 1)) + 1.0

    mat = np.zeros((len(docs), len(vocab)))
    for row, d in enumerate(docs):
        if not d:
            continue
        counts = defaultdict(int)
        for w in d:
            counts[w] += 1
        for w, c in counts.items():
            mat[row, idx[w]] = (c / len(d)) * idf[idx[w]]
    return mat


def embed_texts(texts: List[str]) -> np.ndarray:
    """텍스트 리스트 -> 벡터 행렬 (LLM 호출 없음, 로컬 계산)."""
    backend, model = _get_embed_backend()
    if backend == "st":
        return np.asarray(model.encode(texts, normalize_embeddings=True))
    return _tfidf_vectors(texts)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


# ══════════════════════════════════════════════════════════════════════════════
# 노드 정의
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ItemNode:
    item_id:  str
    agent_id: str
    kind:     str   # "NEED" | "PROVIDE"
    text:     str
    urgency:  int = 3   # 1~5, Offer에 필드 없으면 기본값(중간)


@dataclass
class MatchAssignment:
    need_item_id:  str
    need_agent:    str
    need_text:     str
    target_kind:   str   # "PROVIDE" | "STEP"
    target_id:     str   # provide item_id 또는 step_id(문자열화)
    target_agent:  str
    target_text:   str
    weight:        float


def build_item_nodes(offers: Dict[str, Offer]) -> List[ItemNode]:
    items: List[ItemNode] = []
    for agent_id, offer in offers.items():
        for i, text in enumerate(offer.need_from_other):
            items.append(ItemNode(f"{agent_id}_need_{i}", agent_id, "NEED", text))
        for i, text in enumerate(offer.can_provide):
            items.append(ItemNode(f"{agent_id}_provide_{i}", agent_id, "PROVIDE", text))
    return items


# ══════════════════════════════════════════════════════════════════════════════
# MATCH 후보 그래프 + Hungarian Algorithm으로 전역 최적 매칭
# ══════════════════════════════════════════════════════════════════════════════

MIN_MATCH_SIM_ST    = 0.35   # sentence-transformers 사용 시 (코사인 유사도 스케일이 큼)
MIN_MATCH_SIM_TFIDF = 0.05   # TF-IDF 폴백 시 (실측 결과 관련 있는 쌍도 0.08~0.11 수준이라
                              # 낮게 잡음. 무관한 쌍은 0.000으로 명확히 구분됨)
ROOM_DIST_PENALTY = 0.02


def _min_match_sim() -> float:
    backend, _ = _get_embed_backend()
    return MIN_MATCH_SIM_ST if backend == "st" else MIN_MATCH_SIM_TFIDF


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


def compute_match_assignments(
    items: List[ItemNode],
    plans: Dict[str, LocalPlan],
    offers: Dict[str, Offer],
    room_adjacency: Optional[Dict[Tuple[str, str], int]] = None,
) -> Tuple[List[MatchAssignment], List[str]]:
    """
    NEED × (PROVIDE 아이템 ∪ 다른 agent의 Step) 가중치 그래프를 만들고,
    Hungarian Algorithm으로 전역 최적 1:1 배정을 계산한다.
    반환: (확정된 MatchAssignment 리스트, 매칭 안 된 need_item_id 리스트)
    """
    needs = [it for it in items if it.kind == "NEED"]
    provides = [it for it in items if it.kind == "PROVIDE"]
    all_steps = [s for p in plans.values() for s in p.steps]

    if not needs:
        return [], []

    # 타겟 후보: PROVIDE 아이템 + 모든 Step (핸드오프성 스텝까지 포함 — 상태
    # 요구가 다른 agent의 어떤 행동으로도 충족될 수 있으므로 전체 다 후보)
    targets: List[Tuple[str, str, str, str]] = []  # (kind, id, agent_id, text)
    for p in provides:
        targets.append(("PROVIDE", p.item_id, p.agent_id, p.text))
    for s in all_steps:
        targets.append(("STEP", str(s.step_id), s.agent_id, s.action))

    if not targets:
        return [], [n.item_id for n in needs]

    need_texts = [n.text for n in needs]
    target_texts = [t[3] for t in targets]
    combined_vecs = embed_texts(need_texts + target_texts)  # 한 번에 묶어서 벡터화
    need_vecs = combined_vecs[: len(need_texts)]
    target_vecs = combined_vecs[len(need_texts):]
    sim = cosine_sim_matrix(need_vecs, target_vecs)  # (n_needs, n_targets)

    # 비용 행렬: urgency 가중 + room 거리 페널티를 반영한 "점수"를 비용으로 변환
    cost = np.full(sim.shape, 1e6)
    for i, n in enumerate(needs):
        need_room = offers[n.agent_id].room_type
        for j, (tkind, tid, tagent, ttext) in enumerate(targets):
            if tagent == n.agent_id:
                continue  # 자기 자신은 매칭 대상 아님
            urgency_w = n.urgency / 5.0
            dist_pen = room_distance(need_room, offers[tagent].room_type if tagent in offers else "",
                                      room_adjacency) * ROOM_DIST_PENALTY
            score = sim[i, j] * urgency_w - dist_pen
            cost[i, j] = -score  # Hungarian은 최소화이므로 부호 반전

    row_ind, col_ind = linear_sum_assignment(cost)

    assignments: List[MatchAssignment] = []
    matched_need_ids: Set[str] = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= 1e5:  # 자기자신 등으로 비활성화된 셀
            continue
        weight = sim[r, c]
        if weight < _min_match_sim():
            continue
        n = needs[r]
        tkind, tid, tagent, ttext = targets[c]
        assignments.append(MatchAssignment(
            need_item_id=n.item_id, need_agent=n.agent_id, need_text=n.text,
            target_kind=tkind, target_id=tid, target_agent=tagent,
            target_text=ttext, weight=float(weight),
        ))
        matched_need_ids.add(n.item_id)

    unresolved = [n.item_id for n in needs if n.item_id not in matched_need_ids]
    return assignments, unresolved


# ══════════════════════════════════════════════════════════════════════════════
# 확정된 매칭을 실제 plans에 반영
#   - PROVIDE 타겟 → HANDOFF (기존과 동일: 받는 스텝 자동 삽입 + 순서 조정)
#   - STEP 타겟     → STATE_DEPENDENCY (cross-agent depends_on 엣지만 추가,
#                     새 스텝은 만들지 않음 — 이미 존재하는 상태 변화이므로)
# ══════════════════════════════════════════════════════════════════════════════

_RECEIVE_VERBS = {"receive", "get", "take", "pick", "accept"}
_SEND_VERBS    = {"carry", "bring", "deliver", "transport", "pass"}


def _kw(text: str) -> Set[str]:
    return set(re.findall(r"\w+", (text or "").lower())) - FUZZY_STOPWORDS


def _find_step_by_verb(plan: LocalPlan, item_text: str, verbs: Set[str]) -> Optional[PlanStep]:
    item_kw = _kw(item_text)
    for s in plan.steps:
        first = s.action.lower().split()[0] if s.action.strip() else ""
        if first in verbs and (item_kw & _kw(s.action)):
            return s
    return None


def _find_consuming_step(plan: LocalPlan, need_text: str, exclude_ids: Set[int]) -> Optional[PlanStep]:
    """need 텍스트 키워드와 겹치는, 아직 처리 안 된 스텝을 찾음 (상태 의존성용)."""
    need_kw = _kw(need_text)
    for s in plan.steps:
        if s.step_id in exclude_ids:
            continue
        if need_kw & _kw(s.action):
            return s
    return None


def apply_handoff(
    a: MatchAssignment, plans: Dict[str, LocalPlan],
) -> Tuple[Optional[ConflictEntry], Set[int]]:
    """PROVIDE 타겟 매칭 → 기존 HANDOFF 로직 (받는 스텝 자동 삽입 + 순서 조정)."""
    need_plan = plans[a.need_agent]
    provide_plan = plans[a.target_agent]
    affected: Set[int] = set()

    send_step = _find_step_by_verb(provide_plan, a.target_text, _SEND_VERBS)
    if send_step is None:
        return ConflictEntry(
            ConflictType.DEPENDENCY, [], [a.target_agent],
            f"{a.target_agent} matched to provide '{a.target_text}' but has no send step",
            "manual review required (not auto-resolved)",
        ), affected

    recv_step = _find_step_by_verb(need_plan, a.target_text, _RECEIVE_VERBS)
    if recv_step is None:
        new_id = max([s.step_id for s in need_plan.steps], default=0) + 1
        recv_step = PlanStep(
            step_id=new_id,
            time_min=send_step.time_min + 2,
            room=need_plan.steps[0].room if need_plan.steps else "",
            agent_id=a.need_agent,
            action=f"receive {a.target_text}",
            depends_on=[],
            handoff_type=None,
            target_agent=None,
            uncertainty=0.2,
            notes="auto-inserted by conflict check (DEPENDENCY)",
        )
        need_plan.steps.append(recv_step)
        need_plan.steps.sort(key=lambda s: (s.time_min, s.step_id))
        affected.add(new_id)

    # 이 아이템을 실제로 쓰는 다른 스텝이 받기 전에 와 있으면 순서 재조정
    item_kw = _kw(a.target_text)
    for s in need_plan.steps:
        if s.step_id == recv_step.step_id:
            continue
        first = s.action.lower().split()[0] if s.action.strip() else ""
        if first in _RECEIVE_VERBS:
            continue
        if item_kw & _kw(s.action) and s.time_min <= recv_step.time_min:
            s.time_min = recv_step.time_min + 2
            if recv_step.step_id not in s.depends_on:
                s.depends_on = list(s.depends_on) + [recv_step.step_id]
            affected.add(s.step_id)
    need_plan.steps.sort(key=lambda s: (s.time_min, s.step_id))

    return None, affected


def apply_state_dependency(
    a: MatchAssignment, plans: Dict[str, LocalPlan],
) -> Set[int]:
    """
    STEP 타겟 매칭 → 새 스텝을 만들지 않고, need-agent의 소비 스텝을
    provide-step 이후로 오도록 cross-agent DEPENDS_ON 엣지만 추가.
    (Kahn's algorithm은 agent 구분 없이 depends_on을 그대로 다루므로 안전)
    """
    need_plan = plans[a.need_agent]
    provide_step_id = int(a.target_id)
    provide_step = next(
        (s for p in plans.values() for s in p.steps if s.step_id == provide_step_id), None,
    )
    if provide_step is None:
        return set()

    consumer = _find_consuming_step(need_plan, a.need_text, exclude_ids=set())
    affected: Set[int] = set()
    if consumer is None:
        return affected  # 소비 스텝을 못 찾으면 순서 강제할 대상이 없음 — 정보만 기록

    if provide_step.step_id not in consumer.depends_on:
        consumer.depends_on = list(consumer.depends_on) + [provide_step.step_id]
        affected.add(consumer.step_id)
    if consumer.time_min <= provide_step.time_min:
        consumer.time_min = provide_step.time_min + 1
        affected.add(consumer.step_id)

    return affected


# ══════════════════════════════════════════════════════════════════════════════
# SAME_ROOM 엣지 (미리 구축 — TEMPORAL/REDUNDANCY 체크가 이 엣지를 순회)
# ══════════════════════════════════════════════════════════════════════════════

def build_same_room_edges(plans: Dict[str, LocalPlan]) -> Dict[int, List[int]]:
    all_steps = [s for p in plans.values() for s in p.steps]
    by_room: Dict[str, List[PlanStep]] = defaultdict(list)
    for s in all_steps:
        room = (s.room or "").strip().lower()
        if room:
            by_room[room].append(s)

    edges: Dict[int, List[int]] = defaultdict(list)
    for room, steps in by_room.items():
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                a, b = steps[i], steps[j]
                if a.agent_id == b.agent_id:
                    continue
                edges[a.step_id].append(b.step_id)
                edges[b.step_id].append(a.step_id)
    return edges


# ══════════════════════════════════════════════════════════════════════════════
# CONFLICT 탐지 — SAME_ROOM 엣지를 순회 (전체 쌍 재계산 아님)
# ══════════════════════════════════════════════════════════════════════════════

_TEMPORAL_WINDOW_MIN = 3
REDUNDANCY_SIM_THRESH = 0.92  # 임베딩 유사도 기준 (LLM 아님)


def detect_conflicts_via_edges(
    plans: Dict[str, LocalPlan],
    offers: Dict[str, Offer],
    same_room_edges: Dict[int, List[int]],
    only_step_ids: Optional[Set[int]] = None,
) -> List[ConflictEntry]:
    all_steps = {s.step_id: s for p in plans.values() for s in p.steps}
    conflicts: List[ConflictEntry] = []

    # ── (a) TEMPORAL + (c) REDUNDANCY : SAME_ROOM 엣지 순회 ──────────────────
    node_ids = only_step_ids if only_step_ids is not None else set(same_room_edges.keys())
    seen_pairs: Set[Tuple[int, int]] = set()

    pair_list: List[Tuple[int, int]] = []
    for sid in node_ids:
        for nbr in same_room_edges.get(sid, []):
            pair = (min(sid, nbr), max(sid, nbr))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            pair_list.append(pair)

    if pair_list:
        actions = [all_steps[a].action for a, b in pair_list] + [all_steps[b].action for a, b in pair_list]
        vecs = embed_texts(actions) if actions else np.zeros((0, 1))
        n = len(pair_list)
        for k, (sid1, sid2) in enumerate(pair_list):
            s1, s2 = all_steps[sid1], all_steps[sid2]
            if abs(s1.time_min - s2.time_min) < _TEMPORAL_WINDOW_MIN:
                conflicts.append(ConflictEntry(
                    ConflictType.TEMPORAL, [s1.step_id, s2.step_id],
                    [s1.agent_id, s2.agent_id],
                    f"same room '{s1.room}', time overlap ({s1.time_min}m vs {s2.time_min}m)",
                    "shift the later step's time_min later",
                ))
            v1, v2 = vecs[k], vecs[n + k]
            sim = float(cosine_sim_matrix(v1[None, :], v2[None, :])[0, 0])
            if sim >= REDUNDANCY_SIM_THRESH:
                later = s1 if s1.step_id > s2.step_id else s2
                conflicts.append(ConflictEntry(
                    ConflictType.REDUNDANCY, [s1.step_id, s2.step_id],
                    [s1.agent_id, s2.agent_id],
                    f"near-duplicate action across agents (sim={sim:.2f}): "
                    f"'{s1.action}' / '{s2.action}'",
                    f"remove step {later.step_id}",
                ))

    # ── (d) CANNOT_DO — 자기 혼자 체크 (엣지 불필요) ──────────────────────────
    for sid in (only_step_ids if only_step_ids is not None else all_steps.keys()):
        s = all_steps.get(sid)
        if s is None or "auto-inserted" in (s.notes or ""):
            continue
        offer = offers.get(s.agent_id)
        if not offer:
            continue
        for cd in offer.cannot_do:
            if _kw(cd.action) and _kw(cd.action) <= _kw(s.action):
                conflicts.append(ConflictEntry(
                    ConflictType.CANNOT_DO, [s.step_id], [s.agent_id],
                    f"agent marked cannot_do: '{cd.action}' (reason={cd.reason})",
                    f"remove step {s.step_id}",
                ))
                break

    # ── (e) OBSERVABILITY — 자기 혼자 체크 (엣지 불필요) ──────────────────────
    for sid in (only_step_ids if only_step_ids is not None else all_steps.keys()):
        s = all_steps.get(sid)
        if s is None or s.handoff_type or "auto-inserted" in (s.notes or ""):
            continue
        offer = offers.get(s.agent_id)
        if not offer:
            continue
        pool = _kw(offer.obs_scope)
        for cd in offer.can_do:
            pool |= _kw(cd)
        kw = _kw(s.action)
        if kw and pool and not (kw & pool):
            conflicts.append(ConflictEntry(
                ConflictType.OBSERV, [s.step_id], [s.agent_id],
                f"action '{s.action}' references objects outside observed scope",
                f"remove step {s.step_id}",
            ))

    return conflicts


def resolve_conflict(conflict: ConflictEntry, plans: Dict[str, LocalPlan]) -> Set[int]:
    affected: Set[int] = set()

    if conflict.conflict_type == ConflictType.TEMPORAL:
        s1_id, s2_id = conflict.step_ids
        later_id = max(s1_id, s2_id)
        earlier_id = min(s1_id, s2_id)
        earlier_time = next(
            (o.time_min for p in plans.values() for o in p.steps if o.step_id == earlier_id), None,
        )
        for plan in plans.values():
            for s in plan.steps:
                if s.step_id == later_id and earlier_time is not None:
                    s.time_min = max(s.time_min, earlier_time + _TEMPORAL_WINDOW_MIN)
                    affected.add(s.step_id)

    elif conflict.conflict_type in (ConflictType.REDUNDANCY, ConflictType.CANNOT_DO, ConflictType.OBSERV):
        remove_id = conflict.step_ids[-1] if conflict.conflict_type == ConflictType.REDUNDANCY else conflict.step_ids[0]
        for plan in plans.values():
            before = len(plan.steps)
            plan.steps = [s for s in plan.steps if s.step_id != remove_id]
            if len(plan.steps) != before:
                affected.add(remove_id)

    return affected


# ══════════════════════════════════════════════════════════════════════════════
# Kahn's Algorithm — 사이클 체크 + 위상 정렬 (cross-agent depends_on도 그대로 지원)
# ══════════════════════════════════════════════════════════════════════════════

def kahn_topological_order(plans: Dict[str, LocalPlan]) -> Tuple[List[int], bool]:
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
        queue = deque(sorted(queue, key=lambda sid: all_steps[sid].time_min))
        sid = queue.popleft()
        order.append(sid)
        for nxt in adj[sid]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    no_cycle = len(order) == len(all_steps)
    if not no_cycle:
        remaining = [sid for sid in all_steps if sid not in order]
        remaining.sort(key=lambda sid: all_steps[sid].time_min)
        order.extend(remaining)

    return order, no_cycle


def merge_joint_plan(plans: Dict[str, LocalPlan]) -> List[dict]:
    order, _ = kahn_topological_order(plans)
    all_steps = {s.step_id: s for p in plans.values() for s in p.steps}
    ordered_steps = [all_steps[sid] for sid in order]
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
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def run(
    active_agents,
    offers: Dict[str, Offer],
    plans: Dict[str, LocalPlan],
    task: str = "",
    room_adjacency: Optional[Dict[Tuple[str, str], int]] = None,
    max_rounds: int = 10,
) -> dict:
    plans = copy.deepcopy(plans)

    # ── 1) 매칭 (그래프 + Hungarian Algorithm) ────────────────────────────────
    items = build_item_nodes(offers)
    assignments, unresolved_needs = compute_match_assignments(items, plans, offers, room_adjacency)

    print(f"  [MATCH] {len(assignments)}개 확정 (Hungarian), {len(unresolved_needs)}개 미해결")
    handoff_conflicts: List[ConflictEntry] = []
    for a in assignments:
        if a.target_kind == "PROVIDE":
            print(f"    [HANDOFF] {a.target_agent} → {a.need_agent} : {a.target_text} (sim={a.weight:.2f})")
            conflict, _ = apply_handoff(a, plans)
            if conflict:
                handoff_conflicts.append(conflict)
        else:
            print(f"    [STATE_DEP] {a.need_agent} needs state from {a.target_agent}'s "
                  f"step '{a.target_text}' (sim={a.weight:.2f})")
            apply_state_dependency(a, plans)

    # ── 2) SAME_ROOM 엣지 구축 + Conflict 워크리스트 (엣지 순회) ───────────────
    same_room_edges = build_same_room_edges(plans)
    conflict_queue = deque(detect_conflicts_via_edges(plans, offers, same_room_edges))
    resolved: List[ConflictEntry] = []
    rounds = 0
    while conflict_queue and rounds < max_rounds:
        rounds += 1
        c = conflict_queue.popleft()
        affected = resolve_conflict(c, plans)
        resolved.append(c)
        if affected:
            same_room_edges = build_same_room_edges(plans)  # 스텝 변경 반영해 재구축
            new_conflicts = detect_conflicts_via_edges(plans, offers, same_room_edges, only_step_ids=affected)
            for nc in new_conflicts:
                if nc not in resolved:
                    conflict_queue.append(nc)

    print(f"  [CONFLICT] {len(resolved)}개 자동 해결, {len(handoff_conflicts)}개 미해결(DEPENDENCY)")

    # ── 3) Merge (Kahn's algorithm) ───────────────────────────────────────────
    joint_plan = merge_joint_plan(plans)

    return {
        "updated_plans": plans,
        "assignments": assignments,
        "conflicts_resolved": resolved,
        "conflicts_unresolved": handoff_conflicts + [
            ConflictEntry(ConflictType.DEPENDENCY, [], [], f"unmatched need item: {nid}", "no target found")
            for nid in unresolved_needs
        ],
        "joint_plan": joint_plan,
        "joint_plan_text": format_joint_plan(joint_plan),
    }
