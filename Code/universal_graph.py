# universal_graph.py  (v4 — embedding + CBAA-style auction matching (P2P) + verify/resolve conflict check)
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
#   설계 원칙 (v4) — "전역성"과 "판단(reasoning)"을 분리한다:
#     본 시스템의 각 단계는 (a) 전역 정보가 필요한가, (b) 그 자리에서 뭔가를
#     "판단"하는가라는 두 축으로 분류된다. 이 둘은 서로 독립적이다 —
#     전역 정보가 필요하다고 해서 반드시 판단(=LLM 호출/최적화)이 있는 건
#     아니고, 판단이 있다고 해서 반드시 전역 정보가 필요한 것도 아니다.
#
#       매칭(Auction)         : 로컬(전역 정보 불필요) + 판단 있음(bid 비교)
#                                → 완전 decentralized
#       Conflict Detection    : 전역(SAME_ROOM 엣지 순회 필요) + 판단 없음
#                                (겹치는지 "확인"만 함, "verify")
#       Conflict Resolution   : 로컬(영향받은 노드만) + 판단 있음
#                                (사전 정의된 고정 규칙 적용, "rule-based
#                                resolve" — LLM 없음, 매번 같은 입력엔
#                                같은 출력)
#       Kahn's Cycle Check    : 전역(그래프 전체를 봐야 사이클 유무를 앎)
#                                + 판단 없음 ("verify")
#       Kahn's Cycle Break    : 로컬(끊을 엣지 하나만 선택) + 판단 있음
#                                (cross-agent 엣지 우선이라는 고정 규칙,
#                                "rule-based resolve")
#
#     즉 이 시스템에서 "전역 정보가 필요한 지점"(verify)에는 판단이 없고,
#     "판단이 일어나는 지점"(resolve)은 전역 정보가 필요 없다 — 대부분의
#     중앙집중형 시스템이 "전역 정보를 보고 + 그 자리에서 판단"을 같이
#     하는 것과 근본적으로 다른 구조. 전역 verify는 순수 확인이라 값싸고
#     (엣지 순회 기준 비용, 전체 재스캔 아님), 판단은 전부 로컬 + 결정론적
#     규칙이라 예측 가능하고 감사 가능함.
#
#   알고리즘:
#     - 매칭(v4, CBAA 스타일 auction): 이전 버전(node-centric greedy)은
#       "먼저 처리된 need가 임자"라는 처리 순서(리스트 인덱스)가 승자를
#       결정했음 — 이건 "전체를 보는 계산 주체가 없다"는 P2P 요건과 미묘하게
#       어긋남(순서 자체가 암묵적인 전역 중재자 역할을 했기 때문). v4는
#       승자를 순서가 아니라 bid 값(임베딩 유사도)으로 결정한다. 모든
#       need-agent는 이미 broadcast를 통해 동일한 정보(offer, plan, 임베딩
#       유사도 행렬)를 갖고 있고 — 이것 자체는 유지한다, "정보 공유"를 막을
#       필요는 없고 "판단 주체"만 분산되면 된다 — 이 공통 정보 위에서 각
#       need가 스스로 bid를 계산하고, 매 라운드 "자기가 이길 수 있는 가장
#       좋은 target"에 도전한다. 기존 낙찰자보다 bid가 높으면 낙찰자를
#       교체하고(밀려난 need는 다음 라운드에 재도전), 동률이면 item_id
#       사전순으로 고정 tie-break한다 — 이래야 "누가 언제 계산하든 같은
#       결론"에 도달하고(consensus), 특정 프로세스가 중재자로서 순서를
#       정하지 않는다. 이 루프는 코드 상 하나의 함수로 구현되어 있지만,
#       개념적으로는 각 need-agent가 독립적으로 수행하는 로컬 판단을
#       시뮬레이션한 것이다 — CBAA 원 논문(Choi, Brunet, How, 2009,
#       IEEE T-RO)처럼 실제 분산 환경에서는 각 agent가 이 루프의 자기
#       지분만 수행하고 "현재 낙찰 현황" 메시지만 주고받으면 됨. 전역
#       최적해는 보장하지 않는다(트레이드오프로 감수, greedy auction의
#       일반적 특성).
#     - Conflict: SAME_ROOM 엣지를 미리 만들어두고, 그 엣지를 순회하며
#       "확인"만 한다(verify, 판단 없음). 문제가 확인되면 사전 정의된 고정
#       규칙으로 해소한다(resolve, 로컬 — 영향받은 노드의 이웃 엣지만
#       재확인, 1-hop 전파).
#     - Merge: Kahn's algorithm. 사이클 유무 확인은 전역(verify)이지만,
#       사이클 발견 시 "어떤 엣지를 끊을지"는 cross-agent 엣지 우선이라는
#       고정 규칙(resolve, 로컬)으로 결정 — LLM 개입 없음.
#
# 설계 원칙: 텍스트 임베딩(사전학습 모델, 로컬 계산)만 사용. LLM 호출 없음.
#           sentence-transformers가 없으면 TF-IDF 코사인 유사도로 자동 대체
#           (약한 대체재 — 실제 의미 유사도 포착력은 sentence-transformers보다
#           떨어짐. Colab 등 인터넷 되는 환경에서는 sentence-transformers 설치
#           권장: `pip install sentence-transformers`).
#
# P2P 정리: 관찰/Offer/Local Plan + 이 매칭 단계(v4: auction)는 완전
#           분산(P2P) — 계산 주체가 항상 "그 need를 가진 agent 자신"이고,
#           승자 결정도 순서가 아니라 bid consensus로 이루어짐(중앙
#           중재자 없음). Conflict/Kahn's의 "verify" 단계만 전역 정보가
#           필요하지만 이건 판단이 아니라 확인이고, "resolve" 단계는
#           판단이지만 전역 정보 없이 로컬 + 고정 규칙으로 이루어짐
#           (완전 P2P는 아니고, 판단이 필요한 곳엔 전역성이 없고 전역성이
#           필요한 곳엔 판단이 없는 하이브리드 구조).

from __future__ import annotations

import copy
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

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
# MATCH 후보 그래프 + CBAA 스타일 auction 매칭 (P2P, v4)
# ══════════════════════════════════════════════════════════════════════════════

MIN_MATCH_SIM_ST    = 0.35   # sentence-transformers 사용 시 (코사인 유사도 스케일이 큼)
MIN_MATCH_SIM_TFIDF = 0.05   # TF-IDF 폴백 시 (실측 결과 관련 있는 쌍도 0.08~0.11 수준이라
                              # 낮게 잡음. 무관한 쌍은 0.000으로 명확히 구분됨)
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


def apply_declared_handoffs(
    plans: Dict[str, LocalPlan],
) -> List[MatchAssignment]:
    """
    1단계 — Local Plan 생성 단계에서 LLM이 이미 명시적으로 정한 PASS 타겟
    (handoff_type="PASS" + target_agent 확정)을 auction보다 먼저 신뢰한다.
    이걸 무시하고 auction을 처음부터 다시 돌리면, 이미 정해진 의도를
    엉뚱한 agent한테 재배정해버리는 문제가 생길 수 있음 (실측으로 확인됨).
    이 우선순위 계층 자체가 (auction+graph 조합 자체보다) 본 시스템의
    핵심 차별점 — LLM이 스스로 정한 선언적 의도를 auction이 뒤엎지 않도록
    보호함.
    """
    declared: List[MatchAssignment] = []
    for agent_id, plan in plans.items():
        for s in plan.steps:
            if s.handoff_type == "PASS" and s.target_agent and s.target_agent in plans:
                item_text = s.action
                declared.append(MatchAssignment(
                    need_item_id=f"__declared__{s.step_id}",
                    need_agent=s.target_agent,
                    need_text=item_text,
                    target_kind="PROVIDE",
                    target_id=f"__declared__{s.step_id}",
                    target_agent=agent_id,
                    target_text=item_text,
                    weight=1.0,  # 로컬 플랜에서 이미 확정된 것이므로 최고 신뢰도
                    source_step_id=s.step_id,
                ))
    return declared


def _already_declared_agent_pairs(declared: List[MatchAssignment]) -> Set[Tuple[str, str]]:
    """이미 선언된 (provide_agent, need_agent) 쌍 — auction 대상에서 제외용."""
    return {(a.target_agent, a.need_agent) for a in declared}


def compute_match_assignments(
    items: List[ItemNode],
    plans: Dict[str, LocalPlan],
    offers: Dict[str, Offer],
    room_adjacency: Optional[Dict[Tuple[str, str], int]] = None,
    max_rounds: int = DEFAULT_AUCTION_ROUNDS,
) -> Tuple[List[MatchAssignment], List[str]]:
    """
    NEED × (PROVIDE 아이템 ∪ 다른 agent의 Step) 매칭.

    v4 — CBAA(Consensus-Based Auction Algorithm) 스타일 auction으로 전환.
    이전 버전(node-centric greedy, v3)은 "먼저 처리된 need가 임자"라는
    처리 순서(리스트 인덱스)가 승자를 결정했음 — 이건 "전체를 보는 계산
    주체가 없다"는 P2P 요건과 미묘하게 어긋남(순서 자체가 암묵적인 전역
    중재자 역할을 했기 때문). v4는 승자를 순서가 아니라 bid 값으로
    결정한다:

      - 모든 need-agent는 이미 broadcast를 통해 동일한 정보(offer, plan,
        임베딩 유사도)를 갖고 있음 — 정보 공유 자체는 그대로 유지한다
        ("판단 주체"만 분산되면 되고, 정보 공유를 막을 필요는 없음).
      - 각 need는 이 공통 정보 위에서 "내가 이 target을 얼마나 원하는가"
        (bid = 임베딩 유사도)를 스스로 계산.
      - 라운드마다 각 need가 "자기가 이길 수 있는 가장 좋은 target"에
        도전하고, 기존 낙찰자보다 bid가 높으면 낙찰자를 교체한다(밀려난
        need는 다음 라운드에 재도전). 동률이면 item_id 사전순으로 고정
        tie-break — 이래야 "누가 언제 계산하든 같은 결론"에 도달하고
        (consensus), 특정 프로세스가 중재자로서 순서를 정하지 않는다.
      - 수렴할 때까지(또는 max_rounds) 반복. 이 루프는 코드 상 하나의
        함수로 구현되어 있지만, 개념적으로는 각 need-agent가 독립적으로
        수행하는 로컬 판단을 시뮬레이션한 것 — 실제 분산 환경에서는
        각 agent가 이 루프의 자기 지분만 수행하고 "현재 낙찰 현황"
        메시지만 주고받으면 됨 (CBAA 원 논문, Choi, Brunet, How, 2009,
        IEEE T-RO).

    Local Plan에서 이미 명시적으로 확정된 PASS(target_agent 포함)는
    1단계(apply_declared_handoffs)에서 먼저 신뢰하고, 그 agent 쌍은
    이 단계에서 건드리지 않는다.

    반환: (확정된 MatchAssignment 리스트 — 선언분+auction매칭분, 매칭 안 된 need_item_id 리스트)
    """
    declared = apply_declared_handoffs(plans)
    declared_pairs = _already_declared_agent_pairs(declared)
    # 이미 선언된 handoff를 "받은" agent들 — 이 agent들의 실제 need는
    # (텍스트 매칭 없이도) 그 handoff가 채워주는 것으로 간주해 매칭 완료 처리한다.
    agents_with_declared_incoming = {a.need_agent for a in declared}

    needs = [it for it in items if it.kind == "NEED"]
    provides = [it for it in items if it.kind == "PROVIDE"]
    all_steps = [s for p in plans.values() for s in p.steps]

    assignments: List[MatchAssignment] = list(declared)
    matched_need_ids: Set[str] = {a.need_item_id for a in declared}
    matched_need_ids |= {n.item_id for n in needs if n.agent_id in agents_with_declared_incoming}

    remaining_needs = [n for n in needs if n.item_id not in matched_need_ids]
    if not remaining_needs:
        return assignments, []

    targets: List[Tuple[str, str, str, str]] = []  # (kind, id, agent_id, text)
    for p in provides:
        targets.append(("PROVIDE", p.item_id, p.agent_id, p.text))
    for s in all_steps:
        targets.append(("STEP", str(s.step_id), s.agent_id, s.action))

    if not targets:
        unresolved = [n.item_id for n in needs if n.item_id not in matched_need_ids]
        return assignments, unresolved

    need_texts = [n.text for n in remaining_needs]
    target_texts = [t[3] for t in targets]
    combined_vecs = embed_texts(need_texts + target_texts)
    need_vecs = combined_vecs[: len(need_texts)]
    target_vecs = combined_vecs[len(need_texts):]
    # sim = "broadcast된 공통 정보" — 모든 need-agent가 동일하게 접근 가능한
    # 값. 이 값 자체를 누가 계산했는지는 P2P 성격을 해치지 않는다 (Offer
    # 텍스트 자체가 이미 broadcast되어 있으므로, 그 위에서 유사도를 로컬
    # 재계산하는 것은 각 agent가 스스로 할 수 있는 연산).
    sim = cosine_sim_matrix(need_vecs, target_vecs)

    threshold = _min_match_sim()

    # 각 target(candidate)에 대해 threshold를 넘고, agent 제약(자기 자신 제외,
    # 이미 declared된 쌍 제외)을 통과하는 valid bidder 목록을 미리 걸러둠
    valid_bidders: Dict[int, List[int]] = defaultdict(list)  # target_idx -> [need_idx, ...]
    for j, (tkind, tid, tagent, ttext) in enumerate(targets):
        for i, n in enumerate(remaining_needs):
            if tagent == n.agent_id:
                continue  # 자기 자신은 매칭 대상 아님
            if (tagent, n.agent_id) in declared_pairs:
                continue  # 이미 로컬 플랜에서 이 쌍끼리 확정됨
            if float(sim[i, j]) < threshold:
                continue
            valid_bidders[j].append(i)

    def _beats(challenger_score: float, challenger_idx: int,
               holder_score: float, holder_idx: int) -> bool:
        """challenger가 현재 holder를 이기는가. 동률이면 item_id 사전순 —
        어느 need-agent가 이 판단을 하든 항상 같은 결론에 도달함(consensus).
        """
        if challenger_score != holder_score:
            return challenger_score > holder_score
        return remaining_needs[challenger_idx].item_id < remaining_needs[holder_idx].item_id

    # current_winner: target_idx -> (need_idx, bid) — 이건 "현재 낙찰 현황"이지
    # 중앙 중재자의 결정이 아님. 실제 분산 구현에서는 이게 각 target 후보를
    # broadcast 채널에 올라온 최신 메시지로 대체됨.
    current_winner: Dict[int, Tuple[int, float]] = {}
    matched: Set[int] = set()  # 현재 뭔가를 낙찰받은 need_idx

    for _round in range(max_rounds):
        changed = False
        # need_idx 순회 순서도 item_id 사전순으로 고정 — 순서 자체가 결과에
        # 영향을 주지 않도록 함 (v3의 "리스트 인덱스가 승자를 정한다"는
        # 문제를 여기서도 재현하지 않기 위함)
        order = sorted(range(len(remaining_needs)), key=lambda i: remaining_needs[i].item_id)
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
                    continue  # 이 target은 이미 나보다 강한(또는 동률+선순위) bid가 있음
                if score > best_score:
                    best_score, best_j = score, j
            if best_j is not None:
                prev = current_winner.get(best_j)
                if prev is not None:
                    matched.discard(prev[0])  # 밀려난 need는 다음 라운드에 재도전
                current_winner[best_j] = (i, best_score)
                matched.add(i)
                changed = True
        if not changed:
            break  # 수렴 — 더 이상 아무도 낙찰 현황을 바꾸지 못함

    for j, (i, score) in current_winner.items():
        n = remaining_needs[i]
        tkind, tid, tagent, ttext = targets[j]
        assignments.append(MatchAssignment(
            need_item_id=n.item_id, need_agent=n.agent_id, need_text=n.text,
            target_kind=tkind, target_id=tid, target_agent=tagent,
            target_text=ttext, weight=score,
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
    # 원문 그대로 비교하면 "carry X to kitchen doorway for Y pickup" 같은
    # 상투적 문구(kitchen/doorway/pickup 등)가 겹쳐서 서로 다른 아이템끼리
    # 오매칭될 수 있음 — 동사/목적지 문구를 걷어낸 순수 아이템명으로 비교
    item_kw = _kw(_clean_item_phrase(item_text))
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


_SEND_VERB_PREFIX_RE = re.compile(r"^(carry|bring|deliver|transport|pass)\s+", re.IGNORECASE)
_DOORWAY_SUFFIX_RE = re.compile(r"\s+(to|for)\s+.*(doorway|pickup|delivery).*$", re.IGNORECASE)


def _clean_item_phrase(text: str) -> str:
    """PASS 액션 문장('bring laptop to doorway')에서 동사/목적지 문구를 제거해
    깔끔한 아이템 설명('laptop')만 남김. 이미 깔끔한 텍스트(Offer의 can_provide
    항목 등)는 그대로 통과."""
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
    """
    step_id의 시각을 min_time 이상으로 밀고, 그 스텝에 의존하는(depends_on에
    포함하는) 다른 모든 스텝(agent 무관)도 필요하면 연쇄적으로 뒤로 민다.
    한 곳만 밀고 끝나면 그 뒤에 매달린 스텝들이 시간상 모순되게 남을 수 있어서
    (예: PASS 스텝이 그 준비 스텝보다 먼저인 채로 남는 경우) 이 전파가 필요함.
    """
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
    """
    이 plan 안에서 PASS 스텝 자신 + 그 PASS로 이어지는 depends_on 조상들의
    step_id 집합. 예: prep(A) -> prep(B) -> PASS(C)면 {A,B,C} 전부 포함.
    이 집합에 속한 스텝은 "받은 아이템을 쓰는 스텝"으로 취급하면 안 됨 —
    얘네는 (같은 이름이라도 다른 물건일 수 있는) 자기 아이템을 내보내는
    파이프라인이지, 받은 걸 소비하는 스텝이 아니기 때문.
    """
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
        # 선언된(declared) handoff — Local Plan에서 이미 정확히 어떤 스텝인지
        # 알고 있으므로 키워드 검색 없이 그 스텝을 바로 사용
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
            depends_on=[send_step.step_id],  # 보내는 스텝보다 반드시 뒤에 오도록 강제
            handoff_type=None,
            target_agent=None,
            uncertainty=0.2,
            notes="auto-inserted by conflict check (DEPENDENCY)",
        )
        need_plan.steps.append(recv_step)
        need_plan.steps.sort(key=lambda s: (s.time_min, s.step_id))
        affected.add(new_id)

    # recv_step이 새로 만들어졌든(위에서 depends_on 지정) LLM이 이미 만들어뒀던 것이든,
    # 반드시 send_step보다 뒤에 오도록 의존성을 강제한다 (여기서 한 번 더 보장).
    if send_step.step_id not in recv_step.depends_on:
        recv_step.depends_on = list(recv_step.depends_on) + [send_step.step_id]
        affected.add(recv_step.step_id)
    if recv_step.time_min <= send_step.time_min:
        _propagate_time_forward(plans, recv_step.step_id, send_step.time_min + 2, affected)

    # 이 아이템을 실제로 쓰는 다른 스텝이 받기 전에 와 있으면 순서 재조정
    # (그 스텝에 의존하는 후속 스텝들까지 연쇄적으로 전파)
    # target_text 원문이 아니라 정제된 아이템명으로 비교 — "kitchen doorway for
    # Y pickup" 같은 상투적 문구가 겹쳐서 다른 아이템끼리 오매칭되는 걸 방지
    item_kw = _kw(_clean_item_phrase(a.target_text))
    outgoing_ids = _outgoing_chain_step_ids(need_plan)
    for s in need_plan.steps:
        if s.step_id == recv_step.step_id:
            continue
        first = s.action.lower().split()[0] if s.action.strip() else ""
        if first in _RECEIVE_VERBS:
            continue
        if s.step_id in outgoing_ids:
            # 이 스텝은 (자신이 PASS든, PASS로 이어지는 준비 단계든) 뭔가를
            # 내보내는 파이프라인 소속 — 이름이 같아도 다른 물건일 수 있으므로
            # "받은 아이템을 쓰는 스텝"으로 강제 순서 조정하지 않음
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
        _propagate_time_forward(plans, consumer.step_id, provide_step.time_min + 1, affected)

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
# CONFLICT 탐지 (global verify — 전역이지만 판단 없음, "확인"만 함)
# SAME_ROOM 엣지를 순회 (전체 쌍 재계산 아님, 엣지 기준 비용)
# ══════════════════════════════════════════════════════════════════════════════

_TEMPORAL_WINDOW_MIN = 3
REDUNDANCY_SIM_THRESH = 0.92  # 임베딩 유사도 기준 (LLM 아님)


def detect_conflicts_via_edges(
    plans: Dict[str, LocalPlan],
    offers: Dict[str, Offer],
    same_room_edges: Dict[int, List[int]],
    only_step_ids: Optional[Set[int]] = None,
) -> List[ConflictEntry]:
    """global verify 단계 — 겹치는지/충돌인지 "확인"만 함. 뭘 할지는
    resolve_conflict(로컬 + 고정 규칙)에서 결정한다. 이 함수 자체는 판단을
    내리지 않음 (판단 없는 전역 연산)."""
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
    """rule-based resolve 단계 — 전역 정보 불필요(영향받은 노드만 다룸),
    판단은 있지만 사전 정의된 고정 규칙 적용(LLM 없음, 매번 같은 입력엔
    같은 출력)."""
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
# Kahn's Algorithm — 사이클 체크(global verify) + 위상 정렬/사이클 해소
# (rule-based resolve: cross-agent 엣지 우선 절단)
# ══════════════════════════════════════════════════════════════════════════════

def kahn_topological_order(plans: Dict[str, LocalPlan]) -> Tuple[List[int], bool, List[Tuple[int, int]]]:
    """
    위상 정렬 + 사이클 자동 해소.
    사이클 유무 확인(in-degree=0 노드가 더 없는데 남은 스텝이 있는가)은
    전역 정보가 필요한 verify — 그래프 전체를 봐야 사이클을 알 수 있다는
    건 수학적으로 불가피함, 어떤 로컬 정보로도 우회 불가.
    사이클이 감지되면, 남은 것 중 time_min이 가장 이른 스텝을 골라 그
    스텝으로 들어오는 depends_on 엣지 하나를 끊는다 — 이건 resolve 단계로,
    cross-agent 엣지(STATE_DEPENDENCY 추론으로 생긴 것)를 우선 끊는다는
    고정 규칙이다(로컬 판단, 전역 정보 재확인 불필요). 같은 agent 안
    엣지는 원래 Local Plan에서 LLM이 명시한 의도라서 더 신뢰할 수 있고,
    cross-agent 엣지는 그래프 리즈닝이 사후에 추론한 것이라 끊어도
    상대적으로 안전함.
    반환: (정렬된 step_id 리스트, 사이클 없었는지 여부, 끊긴 엣지 리스트[(from,to)])
    """
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

        # 사이클 발생 — 남은 것 중 time_min이 가장 이른 스텝을 강제로 진입시킴
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

    # 사이클 해제 후에도 depends_on 기준으로 시간이 여전히 모순될 수 있어서,
    # 최종 위상순을 따라가며 "의존 대상보다 반드시 뒤에 오도록" 한 번 더 정합성 보정
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
    events: List[dict] = []  # 데모/시각화용 구조화된 이벤트 로그 (실제 실행 데이터)

    agent_ids = [getattr(a, "agent_id", a) for a in active_agents]
    for aid in agent_ids:
        events.append({"type": "node", "kind": "agent", "id": aid})

    # ── 1) 매칭 (CBAA 스타일 auction, P2P — 로컬 + 판단, 전역 정보 불필요) ────
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

    # ── 2) SAME_ROOM 엣지 구축 + Conflict 워크리스트 (global verify → rule-based resolve) ──
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
            same_room_edges = build_same_room_edges(plans)  # 스텝 변경 반영해 재구축
            new_conflicts = detect_conflicts_via_edges(plans, offers, same_room_edges, only_step_ids=affected)
            for nc in new_conflicts:
                if nc not in resolved:
                    conflict_queue.append(nc)

    print(f"  [CONFLICT] {len(resolved)}개 자동 해결, {len(handoff_conflicts)}개 미해결(DEPENDENCY)")

    # ── 3) Merge (Kahn's algorithm — global verify → rule-based resolve) ──────
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
        "joint_plan": joint_plan,
        "joint_plan_text": format_joint_plan(joint_plan),
        "graph_events": events,  # 데모/시각화(graph_demo.py)에서 그대로 사용
    }
