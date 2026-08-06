# universal_graph.py  (v5 — CBAA auction + need-grounded declared PASS + orphan cleanup)
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
#   설계 원칙 — "전역성"과 "판단(reasoning)"을 분리한다:
#     본 시스템의 각 단계는 (a) 전역 정보가 필요한가, (b) 그 자리에서 뭔가를
#     "판단"하는가라는 두 축으로 분류된다. 이 둘은 서로 독립적이다.
#
#       매칭(Auction)         : 로컬(전역 정보 불필요) + 판단 있음(bid 비교)
#                                → 완전 decentralized
#       Conflict Detection    : 전역(SAME_ROOM 엣지 순회 필요) + 판단 없음(verify)
#       Conflict Resolution   : 로컬(영향받은 노드만) + 판단 있음(rule-based resolve)
#       Kahn's Cycle Check    : 전역(그래프 전체를 봐야 사이클 유무를 앎) + 판단 없음(verify)
#       Kahn's Cycle Break    : 로컬(끊을 엣지 하나만 선택) + 판단 있음(rule-based resolve)
#
#     즉 이 시스템에서 "전역 정보가 필요한 지점"(verify)에는 판단이 없고,
#     "판단이 일어나는 지점"(resolve)은 전역 정보가 필요 없다.
#
#   알고리즘:
#     - 매칭(CBAA 스타일 auction): 승자는 처리 순서가 아니라 bid 값(임베딩
#       유사도)으로 결정. 동률이면 item_id 사전순 고정 tie-break. 모든
#       need-agent가 broadcast된 공통 정보 위에서 로컬로 판단하는 걸
#       시뮬레이션 (CBAA, Choi/Brunet/How 2009, IEEE T-RO).
#     - declared PASS (v5, 신규): Local Plan에서 LLM이 스스로 만든 PASS
#       선언은 "이미 확정된 의도"로 auction보다 우선 신뢰해왔는데, 실측
#       결과 이 의도가 실제 need_from_other와 아무 근거 없이 만들어지는
#       경우가 있었음 (예: 아무도 요청 안 한 laptop을 스스로 handoff).
#       persona가 "능동적으로 도와라"를 지시하므로 이런 제안 자체를 막지는
#       않되, target agent의 need_from_other와 키워드가 전혀 안 겹치면
#       최우선 신뢰(=auction 생략) 자격을 박탈하고 그냥 auction 후보로
#       강등한다. 근거가 있으면 그대로 최우선 통과.
#     - orphan PASS 정리 (v5, 신규): declared든 auction매칭이든, 결국
#       아무도 받지 않은(어떤 스텝의 depends_on에도 안 걸린) PASS 스텝은
#       화면에 "→ 어딘가로 전달"만 뜨고 실제 수신자가 없어 혼란을 주므로,
#       매칭 단계가 끝난 뒤 handoff_type을 제거해 평범한 스텝으로 되돌림.
#     - Conflict: SAME_ROOM 엣지를 미리 만들어두고, 그 엣지를 순회하며
#       "확인"만 한다(verify). 문제가 확인되면 사전 정의된 고정 규칙으로
#       해소한다(resolve, 로컬 — 영향받은 노드의 이웃 엣지만 재확인).
#     - Merge: Kahn's algorithm. 사이클 유무 확인은 전역(verify)이지만,
#       사이클 발견 시 어떤 엣지를 끊을지는 cross-agent 엣지 우선이라는
#       고정 규칙(resolve, 로컬)으로 결정 — LLM 개입 없음.
#
# 설계 원칙: 텍스트 임베딩(사전학습 모델, 로컬 계산)만 사용. LLM 호출 없음.
#           sentence-transformers가 없으면 TF-IDF 코사인 유사도로 자동 대체.
#
# P2P 정리: 관찰/Offer/Local Plan + 매칭(auction)은 완전 분산(P2P) — 계산
#           주체가 항상 "그 need를 가진 agent 자신"이고, 승자 결정도 순서가
#           아니라 bid consensus로 이루어짐(중앙 중재자 없음). Conflict/
#           Kahn's의 "verify" 단계만 전역 정보가 필요하지만 이건 판단이
#           아니라 확인이고, "resolve" 단계는 판단이지만 전역 정보 없이
#           로컬 + 고정 규칙으로 이루어짐.

from __future__ import annotations

import copy
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import FUZZY_STOPWORDS, PASS_SEND_VERBS
from models import ConflictEntry, ConflictType, LocalPlan, Offer, PlanStep
from offer import _kw  # offer.py와 동일한 스테밍 규칙 재사용 (towel/towels 등)
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
    source_step_id: Optional[int] = None  # 선언된 handoff면 정확한 send step_id


def build_item_nodes(offers: Dict[str, Offer]) -> List[ItemNode]:
    items: List[ItemNode] = []
    for agent_id, offer in offers.items():
        for i, text in enumerate(offer.need_from_other):
            items.append(ItemNode(f"{agent_id}_need_{i}", agent_id, "NEED", text))
        for i, text in enumerate(offer.can_provide):
            items.append(ItemNode(f"{agent_id}_provide_{i}", agent_id, "PROVIDE", text))
    return items


# ══════════════════════════════════════════════════════════════════════════════
# MATCH 후보 그래프 + CBAA 스타일 auction 매칭 (P2P)
# ══════════════════════════════════════════════════════════════════════════════

MIN_MATCH_SIM_ST    = 0.35   # sentence-transformers 사용 시 (코사인 유사도 스케일이 큼)
MIN_MATCH_SIM_TFIDF = 0.05   # TF-IDF 폴백 시
ROOM_DIST_PENALTY = 0.02
DEFAULT_AUCTION_ROUNDS = 5   # CBAA 수렴 상한 — 이론적으로 수렴 보장되지만
                              # threshold 근처 값들의 진동에 대한 안전장치


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


# _kw는 이제 offer.py에서 import — 이 파일 안에서 따로 정의하지 않음
# (v5까지는 이 파일에 스테밍 없는 _kw가 따로 있었고, offer.py의 _kw는
# 스테밍이 있어서 'towel' vs 'towels'처럼 단복수만 다른 텍스트가
# 두 파일에서 서로 다르게 취급되는 잠재 버그가 있었음. 하나로 통일함.


DECLARED_BID_BONUS = 0.5           # 근거 있는 declared PASS에 주는 강한 prior —
                                    # 절대적 특권은 아니라 threshold/경쟁에 여전히 걸릴 수 있음


def _gather_declared_candidates(
    plans: Dict[str, LocalPlan],
) -> List[Tuple[int, str, str, str]]:
    """
    Local Plan에서 LLM이 스스로 만든 PASS 선언들을 모아 auction 후보로 변환.

    v6 — 별도의 "무조건 신뢰" 우선순위 계층(구 apply_declared_handoffs)을
    없애고, declared PASS를 auction 내부의 강한 prior(bid 보너스)로
    흡수했다. 예전 계층 구조가 "식탁을 나른다", "요청 안 한 laptop을
    보낸다" 같은 근거 없는 선언까지 강제로 통과시키는 버그의 근원이었음
    — need 근거 검증(강등 로직)을 별도로 계속 추가해가는 대신, auction
    이라는 단일 메커니즘 안에 통합하는 게 더 단순하고 일관됨.

    반환: (step_id, provide_agent, target_agent, item_text) 리스트.
    실제 grounding 판단(보너스를 줄지)과 승자 결정은 compute_match_assignments
    안에서 나머지 후보들과 동일한 auction 로직으로 처리됨.
    """
    out: List[Tuple[int, str, str, str]] = []
    for agent_id, plan in plans.items():
        for s in plan.steps:
            if s.handoff_type == "PASS" and s.target_agent and s.target_agent in plans:
                out.append((s.step_id, agent_id, s.target_agent, s.action))
    return out


def compute_match_assignments(
    items: List[ItemNode],
    plans: Dict[str, LocalPlan],
    offers: Dict[str, Offer],
    room_adjacency: Optional[Dict[Tuple[str, str], int]] = None,
    max_rounds: int = DEFAULT_AUCTION_ROUNDS,
) -> Tuple[List[MatchAssignment], List[str]]:
    """
    NEED × (PROVIDE 아이템 ∪ 다른 agent의 Step ∪ declared PASS 후보) 매칭.

    CBAA(Consensus-Based Auction Algorithm) 스타일 auction: 승자는 처리
    순서가 아니라 bid 값(임베딩 유사도)으로 결정된다. 모든 need-agent는
    이미 broadcast를 통해 동일한 정보를 갖고 있고, 이 공통 정보 위에서
    각 need가 스스로 bid를 계산해 매 라운드 "자기가 이길 수 있는 가장
    좋은 target"에 도전한다. 기존 낙찰자보다 bid가 높으면 교체되고
    (밀려난 need는 다음 라운드에 재도전), 동률이면 item_id 사전순으로
    고정 tie-break — 어느 need-agent가 계산해도 같은 결론에 도달한다
    (consensus, 중앙 중재자 없음).

    v6 — declared PASS를 별도 우선순위 계층이 아니라 auction 내부의 bid
    보너스로 통합. Local Plan에서 LLM이 스스로 만든 PASS 선언은:
      1) target agent의 need 중 하나와 임베딩 유사도가 grounding
         threshold(=_min_match_sim(), 나머지 매칭과 동일한 기준)를 넘으면
         "근거 있음"으로 보고 DECLARED_BID_BONUS를 받는다 — 사실상 거의
         항상 이기지만, 그 need를 다른 진짜 후보가 훨씬 잘 채울 수 있으면
         질 수도 있음(완전한 특권이 아님).
      2) 근거가 없으면 보너스 없이 다른 PROVIDE/STEP 후보와 완전히 동등한
         자격으로 경쟁 — 별도의 "강등" 처리 없이 auction 메커니즘 자체가
         자연스럽게 걸러냄. (이전 버전은 grounding 판단이 키워드 겹침이라
         나머지 매칭 로직 전부가 임베딩 기반인 것과 일관성이 없었음 —
         이제 grounding도 동일한 임베딩 유사도로 판단한다.)

    반환: (확정된 MatchAssignment 리스트, 매칭 안 된 need_item_id 리스트)
    """
    needs = [it for it in items if it.kind == "NEED"]
    provides = [it for it in items if it.kind == "PROVIDE"]
    all_steps = [s for p in plans.values() for s in p.steps]
    declared_candidates = _gather_declared_candidates(plans)

    if not needs:
        return [], []

    # targets: (kind, id, agent_id, text, declared_source_step_id)
    targets: List[Tuple[str, str, str, str, Optional[int]]] = []
    for p in provides:
        targets.append(("PROVIDE", p.item_id, p.agent_id, p.text, None))
    for s in all_steps:
        targets.append(("STEP", str(s.step_id), s.agent_id, s.action, None))

    declared_target_idx: Set[int] = set()
    declared_target_agent: Dict[int, str] = {}
    for step_id, provide_agent, target_agent, text in declared_candidates:
        # PROVIDE로 취급 — apply_handoff가 source_step_id로 바로 찾아가므로
        # verb 검색 없이 정확한 스텝을 사용함 (기존 declared 경로와 동일)
        targets.append(("PROVIDE", f"__declared__{step_id}", provide_agent, text, step_id))
        j = len(targets) - 1
        declared_target_idx.add(j)
        declared_target_agent[j] = target_agent

    if not targets:
        return [], [n.item_id for n in needs]

    need_texts = [n.text for n in needs]
    target_texts = [t[3] for t in targets]
    combined_vecs = embed_texts(need_texts + target_texts)
    need_vecs = combined_vecs[: len(need_texts)]
    target_vecs = combined_vecs[len(need_texts):]
    sim = cosine_sim_matrix(need_vecs, target_vecs)  # "broadcast된 공통 정보"

    threshold = _min_match_sim()

    # declared 후보 grounding 판단 + 보너스 적용 (임베딩 기반, v6)
    for j in sorted(declared_target_idx):
        tagent = declared_target_agent[j]
        agent_need_rows = [i for i, n in enumerate(needs) if n.agent_id == tagent]
        grounded = any(float(sim[i, j]) >= threshold for i in agent_need_rows)
        step_id = targets[j][4]
        provide_agent = targets[j][2]
        if grounded:
            for i in agent_need_rows:
                sim[i, j] = min(1.0, float(sim[i, j]) + DECLARED_BID_BONUS)
            print(f"  [DECLARE] step{step_id} ({provide_agent}\u2192{tagent}): "
                  f"'{targets[j][3]}' \uadfc\uac70 \uc788\uc74c \u2192 bid +{DECLARED_BID_BONUS} \ubcf4\ub108\uc2a4")
        else:
            print(f"  [DECLARE] step{step_id} ({provide_agent}\u2192{tagent}): "
                  f"'{targets[j][3]}' \uadfc\uac70 \uc5c6\uc74c \u2192 \ubcf4\ub108\uc2a4 \uc5c6\uc774 \ub2e4\ub978 \ud6c4\ubcf4\uc640 \ub3d9\ub4f1 \uacbd\uc7c1")

    valid_bidders: Dict[int, List[int]] = defaultdict(list)
    for j, (tkind, tid, tagent, ttext, decl_sid) in enumerate(targets):
        for i, n in enumerate(needs):
            if tagent == n.agent_id:
                continue
            if float(sim[i, j]) < threshold:
                continue
            valid_bidders[j].append(i)

    def _beats(challenger_score: float, challenger_idx: int,
               holder_score: float, holder_idx: int) -> bool:
        if challenger_score != holder_score:
            return challenger_score > holder_score
        return needs[challenger_idx].item_id < needs[holder_idx].item_id

    current_winner: Dict[int, Tuple[int, float]] = {}
    matched: Set[int] = set()

    for _round in range(max_rounds):
        changed = False
        order = sorted(range(len(needs)), key=lambda i: needs[i].item_id)
        for i in order:
            if i in matched:
                continue
            best_j: Optional[int] = None
            best_score = -1.0
            for j, bidders in valid_bidders.items():
                if i not in bidders:
                    continue
                score = float(sim[i, j])
                holder = current_winner.get(j)
                if holder is not None and not _beats(score, i, holder[1], holder[0]):
                    continue
                if score > best_score:
                    best_score, best_j = score, j
            if best_j is not None:
                prev = current_winner.get(best_j)
                if prev is not None:
                    matched.discard(prev[0])
                current_winner[best_j] = (i, best_score)
                matched.add(i)
                changed = True
        if not changed:
            break

    assignments: List[MatchAssignment] = []
    matched_need_ids: Set[str] = set()
    for j, (i, score) in current_winner.items():
        n = needs[i]
        tkind, tid, tagent, ttext, decl_sid = targets[j]
        assignments.append(MatchAssignment(
            need_item_id=n.item_id, need_agent=n.agent_id, need_text=n.text,
            target_kind=tkind, target_id=tid, target_agent=tagent,
            target_text=ttext, weight=score, source_step_id=decl_sid,
        ))
        matched_need_ids.add(n.item_id)

    unresolved = [n.item_id for n in needs if n.item_id not in matched_need_ids]
    return assignments, unresolved


# ══════════════════════════════════════════════════════════════════════════════
# 확정된 매칭을 실제 plans에 반영
# ══════════════════════════════════════════════════════════════════════════════

_RECEIVE_VERBS = {"receive", "get", "take", "pick", "accept"}
_SEND_VERBS    = PASS_SEND_VERBS  # config.py의 단일 소스 — localplan.py의
                                  # _normalize_pass 검증과 반드시 같은 목록이어야 함


def _find_step_by_verb(plan: LocalPlan, item_text: str, verbs: Set[str]) -> Optional[PlanStep]:
    item_kw = _kw(_clean_item_phrase(item_text))
    for s in plan.steps:
        first = s.action.lower().split()[0] if s.action.strip() else ""
        if first in verbs and (item_kw & _kw(s.action)):
            return s
    return None


def _find_consuming_step(plan: LocalPlan, need_text: str, exclude_ids: Set[int]) -> Optional[PlanStep]:
    need_kw = _kw(need_text)
    for s in plan.steps:
        if s.step_id in exclude_ids:
            continue
        if need_kw & _kw(s.action):
            return s
    return None


_SEND_VERB_PREFIX_RE = re.compile(r"^(carry|bring|deliver|transport|pass)\s+", re.IGNORECASE)
_DOORWAY_SUFFIX_RE = re.compile(r"\s+(to|for)\s+.*(doorway|pickup|delivery).*$", re.IGNORECASE)


def _clean_item_phrase(text: str) -> str:
    t = _SEND_VERB_PREFIX_RE.sub("", text)
    t = _DOORWAY_SUFFIX_RE.sub("", t)
    return t.strip() or text


def _propagate_time_forward(
    plans: Dict[str, LocalPlan],
    step_id: int,
    min_time: int,
    affected: Set[int],
    _visited: Optional[Set[int]] = None,
) -> None:
    if _visited is None:
        _visited = set()
    if step_id in _visited:
        return
    _visited.add(step_id)

    all_steps = [s for p in plans.values() for s in p.steps]
    by_id = {s.step_id: s for s in all_steps}
    target = by_id.get(step_id)
    if target is None:
        return

    if target.time_min < min_time:
        target.time_min = min_time
        affected.add(step_id)

    for s in all_steps:
        if step_id in s.depends_on and s.time_min <= target.time_min:
            _propagate_time_forward(plans, s.step_id, target.time_min + 2, affected, _visited)


def _outgoing_chain_step_ids(plan: LocalPlan) -> Set[int]:
    by_id = {s.step_id: s for s in plan.steps}
    result: Set[int] = set()

    def walk(sid: int) -> None:
        if sid in result or sid not in by_id:
            return
        result.add(sid)
        for dep in by_id[sid].depends_on:
            walk(dep)

    for s in plan.steps:
        if s.handoff_type == "PASS":
            walk(s.step_id)
    return result


def apply_handoff(
    a: MatchAssignment, plans: Dict[str, LocalPlan],
) -> Tuple[Optional[ConflictEntry], Set[int]]:
    """PROVIDE 타겟 매칭 → 기존 HANDOFF 로직 (받는 스텝 자동 삽입 + 순서 조정)."""
    need_plan = plans[a.need_agent]
    provide_plan = plans[a.target_agent]
    affected: Set[int] = set()

    if a.source_step_id is not None:
        send_step = next((s for s in provide_plan.steps if s.step_id == a.source_step_id), None)
    else:
        send_step = _find_step_by_verb(provide_plan, a.target_text, _SEND_VERBS)

    if send_step is None:
        return ConflictEntry(
            ConflictType.DEPENDENCY, [], [a.target_agent],
            f"{a.target_agent} matched to provide '{a.target_text}' but has no send step",
            "manual review required (not auto-resolved)",
        ), affected

    recv_step = _find_step_by_verb(need_plan, a.target_text, _RECEIVE_VERBS)
    if recv_step is None:
        clean_item = _clean_item_phrase(a.target_text)
        new_id = max([s.step_id for s in need_plan.steps], default=0) + 1
        recv_step = PlanStep(
            step_id=new_id,
            time_min=send_step.time_min + 2,
            room=need_plan.steps[0].room if need_plan.steps else "",
            agent_id=a.need_agent,
            action=f"receive {clean_item}",
            depends_on=[send_step.step_id],
            handoff_type=None,
            target_agent=None,
            uncertainty=0.2,
            notes="auto-inserted by conflict check (DEPENDENCY)",
        )
        need_plan.steps.append(recv_step)
        need_plan.steps.sort(key=lambda s: (s.time_min, s.step_id))
        affected.add(new_id)

    if send_step.step_id not in recv_step.depends_on:
        recv_step.depends_on = list(recv_step.depends_on) + [send_step.step_id]
        affected.add(recv_step.step_id)
    if recv_step.time_min <= send_step.time_min:
        _propagate_time_forward(plans, recv_step.step_id, send_step.time_min + 2, affected)

    item_kw = _kw(_clean_item_phrase(a.target_text))
    outgoing_ids = _outgoing_chain_step_ids(need_plan)
    for s in need_plan.steps:
        if s.step_id == recv_step.step_id:
            continue
        first = s.action.lower().split()[0] if s.action.strip() else ""
        if first in _RECEIVE_VERBS:
            continue
        if s.step_id in outgoing_ids:
            continue
        if item_kw & _kw(s.action) and s.time_min <= recv_step.time_min:
            if recv_step.step_id not in s.depends_on:
                s.depends_on = list(s.depends_on) + [recv_step.step_id]
            _propagate_time_forward(plans, s.step_id, recv_step.time_min + 2, affected)
    need_plan.steps.sort(key=lambda s: (s.time_min, s.step_id))

    return None, affected


def apply_state_dependency(
    a: MatchAssignment, plans: Dict[str, LocalPlan],
) -> Set[int]:
    """
    STEP 타겟 매칭 → need-agent의 "소비 스텝"을 provide-step 이후로 오도록
    cross-agent DEPENDS_ON 엣지만 추가한다.

    v2 — outgoing chain 제외 추가. _find_consuming_step은 키워드 겹침만
    보고 소비 스텝을 고르는데, need 텍스트에 우연히 겹치는 단어가 있으면
    (예: need="cleared space for serving TRAY", 그리고 같은 agent가 마침
    "arrange ... on serving TRAY"라는 *내보내는* 스텝을 갖고 있는 경우)
    "무언가를 준비해서 내보내는 파이프라인 스텝"을 엉뚱하게 "이 need를
    소비하는 스텝"으로 오인할 수 있음 (실측으로 확인된 버그 — apply_handoff
    에는 이미 _outgoing_chain_step_ids로 이 방어가 있었는데
    apply_state_dependency에는 빠져 있었음). PASS로 뭔가를 내보내는 체인에
    속한 스텝은 애초에 "이 need를 위해 대기하는" 소비자가 될 수 없으므로
    후보에서 제외한다.
    """
    need_plan = plans[a.need_agent]
    provide_step_id = int(a.target_id)
    provide_step = next(
        (s for p in plans.values() for s in p.steps if s.step_id == provide_step_id), None,
    )
    if provide_step is None:
        return set()

    outgoing_ids = _outgoing_chain_step_ids(need_plan)
    consumer = _find_consuming_step(need_plan, a.need_text, exclude_ids=outgoing_ids)
    affected: Set[int] = set()
    if consumer is None:
        return affected

    if provide_step.step_id not in consumer.depends_on:
        consumer.depends_on = list(consumer.depends_on) + [provide_step.step_id]
        affected.add(consumer.step_id)
    if consumer.time_min <= provide_step.time_min:
        _propagate_time_forward(plans, consumer.step_id, provide_step.time_min + 1, affected)

    return affected


# ══════════════════════════════════════════════════════════════════════════════
# orphan PASS 정리 (v5, 신규) — 아무도 안 받은 PASS는 평범한 스텝으로 되돌림
# ══════════════════════════════════════════════════════════════════════════════

def cleanup_orphan_pass(plans: Dict[str, LocalPlan]) -> List[int]:
    """
    declared든 auction매칭이든, 결국 어떤 스텝의 depends_on에도 참조되지
    않은 PASS 스텝은 실제로는 아무도 받지 않은 것 — 화면에 "→ 어딘가로
    전달"만 뜨고 대응하는 RECEIVE가 없어 혼란을 준다. 매칭 단계가 모두
    끝난 뒤, 그런 PASS는 handoff_type/target_agent를 지워 평범한 준비
    스텝으로 되돌린다.
    반환: 정리된 step_id 리스트.
    """
    all_steps = [s for p in plans.values() for s in p.steps]
    referenced: Set[int] = set()
    for s in all_steps:
        referenced.update(s.depends_on)

    cleaned: List[int] = []
    for s in all_steps:
        if s.handoff_type == "PASS" and s.step_id not in referenced:
            s.handoff_type = None
            s.target_agent = None
            cleaned.append(s.step_id)
    return cleaned


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
# CONFLICT 탐지 (global verify)
# ══════════════════════════════════════════════════════════════════════════════

_TEMPORAL_WINDOW_MIN = 3
REDUNDANCY_SIM_THRESH = 0.92


def detect_conflicts_via_edges(
    plans: Dict[str, LocalPlan],
    offers: Dict[str, Offer],
    same_room_edges: Dict[int, List[int]],
    only_step_ids: Optional[Set[int]] = None,
) -> List[ConflictEntry]:
    all_steps = {s.step_id: s for p in plans.values() for s in p.steps}
    conflicts: List[ConflictEntry] = []

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
        if earlier_time is not None:
            _propagate_time_forward(plans, later_id, earlier_time + _TEMPORAL_WINDOW_MIN, affected)

    elif conflict.conflict_type in (ConflictType.REDUNDANCY, ConflictType.CANNOT_DO, ConflictType.OBSERV):
        remove_id = conflict.step_ids[-1] if conflict.conflict_type == ConflictType.REDUNDANCY else conflict.step_ids[0]
        for plan in plans.values():
            before = len(plan.steps)
            plan.steps = [s for s in plan.steps if s.step_id != remove_id]
            if len(plan.steps) != before:
                affected.add(remove_id)

    return affected


# ══════════════════════════════════════════════════════════════════════════════
# Kahn's Algorithm
# ══════════════════════════════════════════════════════════════════════════════

def kahn_topological_order(plans: Dict[str, LocalPlan]) -> Tuple[List[int], bool, List[Tuple[int, int]]]:
    all_steps = {s.step_id: s for p in plans.values() for s in p.steps}
    in_degree = {sid: 0 for sid in all_steps}
    adj: Dict[int, List[int]] = {sid: [] for sid in all_steps}

    for sid, s in all_steps.items():
        for dep in s.depends_on:
            if dep in all_steps:
                adj[dep].append(sid)
                in_degree[sid] += 1

    order: List[int] = []
    remaining: Set[int] = set(all_steps.keys())
    broken_edges: List[Tuple[int, int]] = []
    queue = deque(sorted(sid for sid, d in in_degree.items() if d == 0))

    while remaining:
        while queue:
            queue = deque(sorted(queue, key=lambda sid: all_steps[sid].time_min))
            sid = queue.popleft()
            if sid not in remaining:
                continue
            order.append(sid)
            remaining.discard(sid)
            for nxt in adj[sid]:
                if nxt in remaining:
                    in_degree[nxt] -= 1
                    if in_degree[nxt] == 0:
                        queue.append(nxt)

        if not remaining:
            break

        stuck_id = min(remaining, key=lambda sid: all_steps[sid].time_min)
        stuck_step = all_steps[stuck_id]
        deps_in_remaining = [d for d in stuck_step.depends_on if d in remaining]

        if not deps_in_remaining:
            in_degree[stuck_id] = 0
            queue.append(stuck_id)
            continue

        cross = [d for d in deps_in_remaining if all_steps[d].agent_id != stuck_step.agent_id]
        to_break = cross[0] if cross else deps_in_remaining[0]
        stuck_step.depends_on = [d for d in stuck_step.depends_on if d != to_break]
        broken_edges.append((to_break, stuck_id))
        in_degree[stuck_id] -= 1
        if in_degree[stuck_id] == 0:
            queue.append(stuck_id)

    no_cycle = len(broken_edges) == 0
    return order, no_cycle, broken_edges


def merge_joint_plan(plans: Dict[str, LocalPlan]) -> Tuple[List[dict], List[Tuple[int, int]]]:
    order, no_cycle, broken_edges = kahn_topological_order(plans)
    all_steps = {s.step_id: s for p in plans.values() for s in p.steps}

    if broken_edges:
        print(f"  [MERGE] 사이클 감지 → {len(broken_edges)}개 cross-agent 엣지 자동 해제: {broken_edges}")

    for sid in order:
        s = all_steps[sid]
        for dep in s.depends_on:
            dep_step = all_steps.get(dep)
            if dep_step is not None and s.time_min <= dep_step.time_min:
                s.time_min = dep_step.time_min + 1

    ordered_steps = [all_steps[sid] for sid in order]
    joint_plan = [
        {
            "step_id": s.step_id, "time_min": s.time_min, "room": s.room,
            "agent_id": s.agent_id, "action": s.action, "depends_on": s.depends_on,
            "handoff_type": s.handoff_type, "target_agent": s.target_agent,
            "notes": s.notes,
        }
        for s in ordered_steps
    ]
    return joint_plan, broken_edges


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
    auction_rounds: int = DEFAULT_AUCTION_ROUNDS,
) -> dict:
    plans = copy.deepcopy(plans)
    events: List[dict] = []

    agent_ids = [getattr(a, "agent_id", a) for a in active_agents]
    for aid in agent_ids:
        events.append({"type": "node", "kind": "agent", "id": aid})

    # ── 1) 매칭 (CBAA 스타일 auction, P2P) ────────────────────────────────────
    items = build_item_nodes(offers)
    for it in items:
        events.append({
            "type": "node", "kind": "item", "id": it.item_id, "agent": it.agent_id,
            "item_kind": it.kind, "text": it.text,
        })
    for p in plans.values():
        for s in p.steps:
            events.append({
                "type": "node", "kind": "step", "id": f"step_{s.step_id}",
                "agent": s.agent_id, "text": s.action,
            })

    assignments, unresolved_needs = compute_match_assignments(
        items, plans, offers, room_adjacency, max_rounds=auction_rounds,
    )

    print(f"  [AUCTION] {len(assignments)}개 확정 "
          f"(선언 {sum(1 for a in assignments if a.source_step_id is not None)}개 + "
          f"auction매칭 {sum(1 for a in assignments if a.source_step_id is None)}개), "
          f"{len(unresolved_needs)}개 미해결")
    handoff_conflicts: List[ConflictEntry] = []
    for a in assignments:
        declared = a.source_step_id is not None
        tag = "선언됨" if declared else f"bid={a.weight:.2f}"
        if a.target_kind == "PROVIDE":
            print(f"    [HANDOFF] {a.target_agent} → {a.need_agent} : {a.target_text} ({tag})")
            conflict, _ = apply_handoff(a, plans)
            events.append({
                "type": "edge", "kind": "handoff", "from": a.target_agent, "to": a.need_agent,
                "label": a.target_text, "weight": a.weight, "declared": declared,
            })
            if conflict:
                handoff_conflicts.append(conflict)
        else:
            print(f"    [STATE_DEP] {a.need_agent} needs state from {a.target_agent}'s "
                  f"step '{a.target_text}' ({tag})")
            apply_state_dependency(a, plans)
            events.append({
                "type": "edge", "kind": "state_dependency",
                "from": f"step_{a.target_id}", "to": a.need_agent,
                "label": a.target_text, "weight": a.weight, "declared": declared,
            })
    for nid in unresolved_needs:
        events.append({"type": "unresolved_need", "id": nid})

    # ── 1.5) orphan PASS 정리 ──────────────────────────────────────────────────
    cleaned = cleanup_orphan_pass(plans)
    if cleaned:
        print(f"  [CLEANUP] 수신자 없는 PASS {len(cleaned)}개 → 평범한 스텝으로 정리: {cleaned}")
        for sid in cleaned:
            events.append({"type": "orphan_pass_cleaned", "id": f"step_{sid}"})

    # ── 2) SAME_ROOM 엣지 구축 + Conflict 워크리스트 ───────────────────────────
    same_room_edges = build_same_room_edges(plans)
    conflict_queue = deque(detect_conflicts_via_edges(plans, offers, same_room_edges))
    resolved: List[ConflictEntry] = []
    rounds = 0
    while conflict_queue and rounds < max_rounds:
        rounds += 1
        c = conflict_queue.popleft()
        affected = resolve_conflict(c, plans)
        resolved.append(c)
        events.append({
            "type": "conflict_resolved", "conflict_type": c.conflict_type,
            "step_ids": c.step_ids, "description": c.description,
        })
        if affected:
            same_room_edges = build_same_room_edges(plans)
            new_conflicts = detect_conflicts_via_edges(plans, offers, same_room_edges, only_step_ids=affected)
            for nc in new_conflicts:
                if nc not in resolved:
                    conflict_queue.append(nc)

    print(f"  [CONFLICT] {len(resolved)}개 자동 해결, {len(handoff_conflicts)}개 미해결(DEPENDENCY)")

    # ── 3) Merge (Kahn's algorithm) ───────────────────────────────────────────
    joint_plan, broken_cycle_edges = merge_joint_plan(plans)
    for from_id, to_id in broken_cycle_edges:
        events.append({"type": "cycle_broken", "from": f"step_{from_id}", "to": f"step_{to_id}"})
    events.append({
        "type": "final_order",
        "order": [s["step_id"] for s in joint_plan],
    })

    return {
        "updated_plans": plans,
        "assignments": assignments,
        "conflicts_resolved": resolved,
        "conflicts_unresolved": handoff_conflicts + [
            ConflictEntry(ConflictType.DEPENDENCY, [], [], f"unmatched need item: {nid}", "no target found")
            for nid in unresolved_needs
        ],
        "broken_cycle_edges": broken_cycle_edges,
        "orphan_pass_cleaned": cleaned,
        "joint_plan": joint_plan,
        "joint_plan_text": format_joint_plan(joint_plan),
        "graph_events": events,
    }
