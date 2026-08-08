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
#                     (REDUNDANCY 체크가 이 엣지를 "순회"하도록. TEMPORAL은 v9에서 제거됨)
#     - DEPENDS_ON  : 같은 agent 안(원래) + STATE_DEPENDENCY로 추가되는
#                     cross-agent 순서 제약
#
#   설계 원칙 — "전역성"과 "판단(reasoning)"을 분리한다:
#     본 시스템의 각 단계는 (a) 전역 정보가 필요한가, (b) 그 자리에서 뭔가를
#     "판단"하는가라는 두 축으로 분류된다. 이 둘은 서로 독립적이다.
#
#       매칭(Auction)         : 로컬(Offer + Local Plan 스텝 텍스트 — 매칭 전
#                                이미 전부 공개됨) + 판단 있음(bid 비교)
#                                → 예외 없이 완전 decentralized
#                                  (PROVIDE 타겟이든 STEP 타겟이든 동일)
#       Conflict Detection    : 전역(SAME_ROOM 엣지 순회 필요) + 판단 없음(verify)
#       Conflict Resolution   : 로컬(영향받은 노드만) + 판단 있음(rule-based resolve)
#       Kahn's Cycle Check    : 전역(그래프 전체를 봐야 사이클 유무를 앎) + 판단 없음(verify)
#       Kahn's Cycle Break    : 로컬(끊을 엣지 하나만 선택) + 판단 있음(rule-based resolve)
#
#     정리: 매칭(4단계)은 전부 decentralized, 그래프(5단계)는 매칭을
#     전혀 하지 않고 오직 verify+resolve("체크")만 한다 — 이 둘의 역할이
#     이제 완전히 분리된다.
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


def _auction_phase(
    need_idx: int,
    sim: np.ndarray,
    valid_bidders: Dict[int, List[int]],
    current_winner: Dict[int, Tuple[int, float]],
    needs: List[ItemNode],
) -> Optional[Tuple[int, float]]:
    """
    경매 단계(Auction Phase) — GCAA(Braquet & Bakolas, 2021)의 2단계 구조를
    본떠 명시적으로 분리함.

    need_idx 하나만의 완전히 로컬인 연산: 유효하게 입찰 가능한 target들 중,
    "직전 라운드까지 확정되어 공개된 낙찰 현황"(current_winner)을 기준으로
    자신이 이길 수 있는(또는 아직 아무도 없는) target 중 bid가 가장 높은
    것 하나를 "제안"으로 고른다.

    current_winner를 참조하는 게 "로컬"이라는 원칙을 깨지 않는 이유:
    이건 이번 라운드에 다른 need가 몰래 뭘 제안하는지 훔쳐보는 게 아니라,
    이미 이전 라운드에 합의(consensus)되어 공개 broadcast된 정보를 읽는
    것이다 — CBAA/GCAA 에이전트도 정확히 이렇게 동작한다: 매 라운드
    "공개된 현재 낙찰 현황 대비 내가 이길 수 있는 가장 좋은 target"에
    다시 도전한다. 이 스냅샷을 안 보면(초기 버전의 버그), 이길 수 없는
    target에 영원히 재도전만 하다가 대안으로 못 넘어가는 문제가 생김
    (실측으로 확인됨 — 아래 unit test 참고).
    """
    def _beats(challenger_score: float, challenger_idx: int,
               holder_score: float, holder_idx: int) -> bool:
        if challenger_score != holder_score:
            return challenger_score > holder_score
        return needs[challenger_idx].item_id < needs[holder_idx].item_id

    best_j: Optional[int] = None
    best_score = -1.0
    for j, bidders in valid_bidders.items():
        if need_idx not in bidders:
            continue
        score = float(sim[need_idx, j])
        holder = current_winner.get(j)
        if holder is not None and holder[0] != need_idx and not _beats(score, need_idx, holder[1], holder[0]):
            continue  # 공개된 현재 낙찰자를 못 이기는 target은 제안 후보에서 제외
        if score > best_score:
            best_score, best_j = score, j
    if best_j is None:
        return None
    return best_j, best_score


def _consensus_phase(
    proposals: Dict[int, Tuple[int, float]],
    current_winner: Dict[int, Tuple[int, float]],
    matched: Set[int],
    needs: List[ItemNode],
) -> bool:
    """
    합의 단계(Consensus Phase) — 이번 라운드에 모인 모든 제안(proposals)을
    훑으며 승자를 확정한다.

    승자 판정 규칙은 "bid 값이 더 높은가, 동률이면 item_id 사전순"뿐이라,
    이 규칙 자체는 공개되어 있고 결정론적이다 — 그래서 어느 need-agent가
    이 규칙을 스스로 적용해도(즉 이 함수를 각자 로컬로 돌려도) 항상 같은
    결론에 도달한다. 특정 프로세스가 "중재자"로서 승패를 결정하는 게
    아니라, 모두가 같은 공개 규칙 위에서 같은 답에 수렴하는 것 — 이게
    "합의(consensus)"의 정확한 의미다. 참조 구현은 명료성을 위해 이걸
    하나의 함수로 순차 처리하지만, 결과는 각 need-agent가 독립적으로
    이 규칙을 적용한 것과 동일하다.

    반환: 이번 라운드에 낙찰 현황이 하나라도 바뀌었는지 여부(수렴 판정용).
    """
    def _beats(challenger_score: float, challenger_idx: int,
               holder_score: float, holder_idx: int) -> bool:
        if challenger_score != holder_score:
            return challenger_score > holder_score
        return needs[challenger_idx].item_id < needs[holder_idx].item_id

    changed = False
    order = sorted(proposals.keys(), key=lambda i: needs[i].item_id)
    for i in order:
        target_j, score = proposals[i]
        holder = current_winner.get(target_j)
        if holder is not None and not _beats(score, i, holder[1], holder[0]):
            continue
        prev = current_winner.get(target_j)
        if prev is not None:
            matched.discard(prev[0])
        current_winner[target_j] = (i, score)
        matched.add(i)
        changed = True
    return changed


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

    v8 — Local Plan 스텝 텍스트도 Offer와 동일하게 "생성 후 공개되는
    정보"로 취급한다. 이전 버전은 Local Plan이 Coordinator에게만
    제출되고 팀원 간 비공개라고 전제해, PROVIDE 타겟 매칭(Local)과
    STEP 타겟 매칭(Global — Coordinator만 계산 가능)을 구분했다. 이제는
    Local Plan 생성이 끝나는 즉시 그 스텝 텍스트도 Offer처럼 팀 전체에
    공개된다고 정의하므로, 이 구분이 사라진다 — PROVIDE든 STEP이든
    매칭에 필요한 정보(Offer + Local Plan 스텝 텍스트)가 매칭 시작 전에
    이미 전부 공개돼 있고, 계산 방식(bid+합의)도 원래 동일했으므로,
    Auction Matching 전체가 예외 없이 decentralized(개념적으로 각
    need-agent가 로컬로 계산 가능)라고 부를 수 있다.

    (참고: 이전 버전에서 이 부분에 "Local vs Global" 구분이 있었음 — 이
    구분은 정보 공개 여부에 대한 설계 선택이 바뀌면서 더 이상 필요 없음.
    그래프(5단계)는 이제 매칭을 전혀 수행하지 않고, 순수 검증(verify)과
    해소(resolve)만 담당한다 — "그래프는 체크만 한다"는 원칙.)

    declared PASS는 별도 우선순위 계층이 아니라 auction 내부의 bid
    보너스로 통합되어 있다. Local Plan에서 LLM이 스스로 만든 PASS
    선언은:
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

    current_winner: Dict[int, Tuple[int, float]] = {}
    matched: Set[int] = set()

    for _round in range(max_rounds):
        # ── 경매 단계: 모든 미매칭 need가, 이번 라운드 시작 시점의 낙찰
        #    현황만 보고, 서로 뭘 제안하는지 모른 채(동시에) 자기 제안을
        #    계산한다 — 완전히 로컬인 연산 ─────────────────────────────────
        proposals: Dict[int, Tuple[int, float]] = {}
        for i in range(len(needs)):
            if i in matched:
                continue
            proposal = _auction_phase(i, sim, valid_bidders, current_winner, needs)
            if proposal is not None:
                proposals[i] = proposal

        if not proposals:
            break

        # ── 합의 단계: 이번 라운드에 모인 제안들을 공개된 결정론적 규칙으로
        #    한꺼번에 반영 → 승자 확정 (중재자 없음, 규칙만 있음) ─────────────
        changed = _consensus_phase(proposals, current_winner, matched, needs)
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


@dataclass
class MissingReceive:
    """4단계(Matching)에서 HANDOFF는 확정됐지만, 받는 쪽 로컬 플랜에 대응하는
    receive 스텝이 아직 없는 경우를 나타내는 신호. 이 스텝을 실제로
    만들어 삽입하는 건 apply_handoff(4단계, matching 결정)의 몫이 아니라
    resolve_missing_receive(5단계, resolve — "insert step" 규칙)의 몫이다.
    """
    need_agent:    str
    send_step_id:  int
    target_text:   str


def apply_handoff(
    a: MatchAssignment, plans: Dict[str, LocalPlan],
) -> Tuple[Optional[ConflictEntry], Set[int], Optional["MissingReceive"]]:
    """
    PROVIDE 타겟 매칭 → HANDOFF 엣지 연결.

    v7 — "받는 스텝이 없으면 새로 만든다"는 로직을 이 함수에서 제거했다.
    이건 matching 결정(4단계: "누가 누구에게 보내는가")이 아니라, 그
    결정이 실제 플랜 구조와 안 맞을 때 고정 규칙으로 고치는 resolve
    행위(5단계: "insert step")이기 때문이다 — 5단계 Verify+Resolve
    다이어그램에서 이미 "insert step"을 resolve 규칙 중 하나로 명시해
    뒀는데, 실제 코드는 이 로직이 4단계 쪽에 섞여 있어 불일치가 있었다.

    이제 이 함수는 순수하게 "이미 존재하는 노드들 사이의 엣지 연결"만
    담당한다: send_step을 찾고, recv_step이 *이미 존재하면* 엣지를
    연결하고 순서를 맞춘다. recv_step이 없으면 여기서 만들지 않고
    MissingReceive를 반환해 5단계로 넘긴다.

    반환: (conflict, affected, missing_receive) — 셋 중 실제로 값이
    있는 건 상황에 따라 하나뿐이다.
    """
    need_plan = plans[a.need_agent]
    provide_plan = plans[a.target_agent]
    affected: Set[int] = set()

    if a.source_step_id is not None:
        send_step = next((s for s in provide_plan.steps if s.step_id == a.source_step_id), None)
    else:
        send_step = _find_step_by_verb(provide_plan, a.target_text, _SEND_VERBS)

    if send_step is None:
        return ConflictEntry(
            ConflictType.DEPENDENCY, [], [a.target_agent],  # Match Failure — 확정된 매칭인데 실제 송신 스텝이 없음
            f"{a.target_agent} matched to provide '{a.target_text}' but has no send step",
            "notified — no auto-fix (target agent should be informed manually)",
        ), affected, None

    recv_step = _find_step_by_verb(need_plan, a.target_text, _RECEIVE_VERBS)
    if recv_step is None:
        # 여기서 만들지 않음 — 5단계(resolve_missing_receive)로 위임
        return None, affected, MissingReceive(
            need_agent=a.need_agent,
            send_step_id=send_step.step_id,
            target_text=a.target_text,
        )

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

    return None, affected, None


def resolve_missing_receive(
    mr: "MissingReceive", plans: Dict[str, LocalPlan],
) -> Set[int]:
    """
    5단계 Resolve의 "insert step" 규칙 — 4단계에서 확정된 HANDOFF가 받는
    쪽 로컬 플랜과 구조적으로 안 맞을 때(대응 receive 스텝 없음), 고정
    규칙으로 새 스텝을 만들어 끼워 넣는다. 로직 자체는 이전 버전의
    apply_handoff 안에 있던 것과 동일 — 어디서 실행되는지(4단계 매칭
    처리 중이 아니라 5단계 resolve 단계)만 바뀌었다.
    """
    need_plan = plans[mr.need_agent]
    send_step = next(
        (s for p in plans.values() for s in p.steps if s.step_id == mr.send_step_id), None,
    )
    affected: Set[int] = set()
    if send_step is None:
        return affected

    clean_item = _clean_item_phrase(mr.target_text)
    new_id = max([s.step_id for s in need_plan.steps], default=0) + 1
    recv_step = PlanStep(
        step_id=new_id,
        time_min=send_step.time_min + 2,
        room=need_plan.steps[0].room if need_plan.steps else "",
        agent_id=mr.need_agent,
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

    if recv_step.time_min <= send_step.time_min:
        _propagate_time_forward(plans, recv_step.step_id, send_step.time_min + 2, affected)

    item_kw = _kw(clean_item)
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

    return affected


def _outgoing_chain_prep_ids(plan: LocalPlan) -> Set[int]:
    """
    이 plan 안에서 PASS 스텝으로 "이어지는" depends_on 조상들의 step_id
    집합 — PASS 스텝 "자신"은 포함하지 않는다 (_outgoing_chain_step_ids와의
    차이점).

    이 구분이 중요한 이유(v3 버그 수정): 준비 스텝(예: "arrange tray")은
    받는 쪽 상태와 무관하게 자기 아이템을 만드는 파이프라인이라, 엉뚱한
    need와 키워드가 우연히 겹쳐도 소비자로 오인되면 안 됨 — 이건 그대로
    제외 대상이다. 반면 PASS 스텝 자신(예: "carry tray to doorway")은
    "받는 쪽이 준비될 때까지 실제로 기다려야 하는" 정당한 소비자일 수
    있다 (예: 상대방 테이블이 치워지기 전까지 배달을 미뤄야 하는 경우).
    이전 버전(_outgoing_chain_step_ids 그대로 재사용)은 PASS 스텝까지
    통째로 배제해서, 이런 정당한 state dependency가 조용히 사라지는
    부작용이 있었음 (실측으로 확인됨).
    """
    by_id = {s.step_id: s for s in plan.steps}
    result: Set[int] = set()

    def walk(sid: int) -> None:
        if sid not in by_id or sid in result:
            return
        step = by_id[sid]
        if step.handoff_type != "PASS":
            result.add(sid)
        for dep in step.depends_on:
            walk(dep)

    for s in plan.steps:
        if s.handoff_type == "PASS":
            for dep in s.depends_on:
                walk(dep)
    return result


def apply_state_dependency(
    a: MatchAssignment, plans: Dict[str, LocalPlan],
) -> Set[int]:
    """
    STEP 타겟 매칭 → need-agent의 "소비 스텝"을 provide-step 이후로 오도록
    cross-agent DEPENDS_ON 엣지만 추가한다.

    v3 — 소비자 후보에서 제외하는 대상을 "PASS로 이어지는 준비 스텝"만으로
    좁힘 (PASS 스텝 자신은 제외하지 않음). 자세한 이유는
    _outgoing_chain_prep_ids 참고.
    """
    need_plan = plans[a.need_agent]
    provide_step_id = int(a.target_id)
    provide_step = next(
        (s for p in plans.values() for s in p.steps if s.step_id == provide_step_id), None,
    )
    if provide_step is None:
        return set()

    outgoing_prep_ids = _outgoing_chain_prep_ids(need_plan)
    consumer = _find_consuming_step(need_plan, a.need_text, exclude_ids=outgoing_prep_ids)
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
# v10 — SAME_ROOM 엣지 기반 conflict 탐지(build_same_room_edges,
# detect_conflicts_via_edges, resolve_conflict)를 전체 제거함.
#
# 순서대로 있었던 5종 conflict 중 남은 게 하나도 없어졌기 때문:
#   - TEMPORAL: v9에서 제거 (SAME_ROOM 엣지 자체가 room 구조상 거의 발생 안 함)
#   - CANNOT_DO, OBSERVABILITY: v9에서 localplan.py 로컬 자체검사로 이동
#   - REDUNDANCY: v10에서 제거. SAME_ROOM 조건에 얹혀 있던 게 애초에 설계
#     오류였음(중복 낭비는 물리적 위치와 무관한 문제인데 왜 같은 room일
#     때만 확인하는지 정당화가 안 됐음) + 이 태스크 세팅에서는 SAME_ROOM
#     엣지가 사실상 전혀 발생하지 않아 검증 자체가 불가능했음.
#
# 그 결과 SAME_ROOM 엣지를 사용하는 conflict 유형이 하나도 안 남아서,
# 이 엣지 구축/순회 로직 자체가 죽은 코드가 됨 — 통째로 제거.
#
# 5단계(그래프)에 남는 건 이제 정확히 둘뿐이다:
#   - Match Failure : matching 실패(unresolved need, send step 못 찾음 등)로
#     별도 누적되는 것 — 자동 해소 없이 사람에게 통보만 함 (내부 상수는
#     하위 호환을 위해 여전히 ConflictType.DEPENDENCY를 씀)
#   - Cycle         : Kahn's algorithm이 DEPENDS_ON 그래프에서 탐지 + 해소
#     (검증=전역·무판단, 해소="break edge" 규칙)
#   - Missing Receive : declared HANDOFF는 확정됐지만 받는 쪽 로컬 플랜에
#     대응 스텝이 없는 경우 — 검증은 4단계 matching 처리 중 발견되고,
#     해소("insert step" 규칙)는 5단계에서 수행 (내부 상수는 기존에
#     미사용이던 ConflictType.HANDOFF를 재사용)
# ══════════════════════════════════════════════════════════════════════════════


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
    max_rounds: int = 10,  # v10부터 미사용 (conflict 재시도 루프 제거됨) — 하위 호환용으로만 유지
    auction_rounds: int = DEFAULT_AUCTION_ROUNDS,
) -> dict:
    plans = copy.deepcopy(plans)
    events: List[dict] = []

    agent_ids = [getattr(a, "agent_id", a) for a in active_agents]
    for aid in agent_ids:
        events.append({"type": "node", "kind": "agent", "id": aid})

    # ── 1) 매칭 (CBAA 스타일 auction, 전부 decentralized) ──────────────────────
    # 이 시점(각 agent의 Local Plan 생성이 모두 끝난 직후)부터 Local Plan
    # 스텝 텍스트도 Offer와 동일하게 "공개된 정보"로 취급한다 — Coordinator
    # 전용이 아니다. PROVIDE 타겟이든 STEP 타겟이든 매칭에 필요한 정보가
    # 이 시점에 전부 공개돼 있으므로, 이후 매칭 전체가 decentralized다.
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
    missing_receives: List[MissingReceive] = []
    for a in assignments:
        declared = a.source_step_id is not None
        tag = "선언됨" if declared else f"bid={a.weight:.2f}"
        if a.target_kind == "PROVIDE":
            print(f"    [HANDOFF] {a.target_agent} → {a.need_agent} : {a.target_text} ({tag})")
            conflict, _, missing = apply_handoff(a, plans)
            events.append({
                "type": "edge", "kind": "handoff", "from": a.target_agent, "to": a.need_agent,
                "label": a.target_text, "weight": a.weight, "declared": declared,
            })
            if conflict:
                handoff_conflicts.append(conflict)
            if missing:
                missing_receives.append(missing)
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

    # ── 1.5) Resolve: 받는 쪽에 대응 스텝이 없는 HANDOFF는 여기서 삽입
    #        ("insert step" 규칙 — 5단계 Resolve 소속. 4단계 matching은
    #        엣지 연결까지만 담당하고, 구조적 불일치를 고치는 건 여기서).
    #        v11 — Missing Receive를 정식 conflict 카테고리로 승격.
    #        내부 상수는 ConflictType.HANDOFF를 재사용(기존에 미사용 상수) ──
    resolved: List[ConflictEntry] = []
    for mr in missing_receives:
        resolve_missing_receive(mr, plans)
        print(f"    [RESOLVE] insert step: '{mr.target_text}' 수신 스텝을 "
              f"{mr.need_agent}에 자동 삽입 (step{mr.send_step_id} 이후)")
        resolved.append(ConflictEntry(
            ConflictType.HANDOFF, [mr.send_step_id], [mr.need_agent],
            f"declared HANDOFF confirmed but {mr.need_agent} had no receive step for "
            f"'{mr.target_text}'",
            "insert step",
        ))
        events.append({
            "type": "resolve_insert_step", "agent": mr.need_agent,
            "label": mr.target_text, "after_step": f"step_{mr.send_step_id}",
        })

    # ── 1.6) orphan PASS 정리 ──────────────────────────────────────────────────
    cleaned = cleanup_orphan_pass(plans)
    if cleaned:
        print(f"  [CLEANUP] 수신자 없는 PASS {len(cleaned)}개 → 평범한 스텝으로 정리: {cleaned}")
        for sid in cleaned:
            events.append({"type": "orphan_pass_cleaned", "id": f"step_{sid}"})

    # v10 — SAME_ROOM 엣지 기반 conflict 워크리스트(구 2단계) 제거됨. 5단계에
    # 남은 conflict는 Missing Receive(위), Cycle(바로 아래), Match Failure
    # (핸드오프에 4단계에서 그대로 누적된 것)뿐이다.
    print(f"  [MATCH FAILURE] {len(handoff_conflicts) + len(unresolved_needs)}개 — 자동 해소 불가, 통보만 함")

    # ── 2) Merge (Kahn's algorithm — global verify(cycle 탐지) + rule-based resolve) ──
    joint_plan, broken_cycle_edges = merge_joint_plan(plans)
    for from_id, to_id in broken_cycle_edges:
        events.append({"type": "cycle_broken", "from": f"step_{from_id}", "to": f"step_{to_id}"})
        resolved.append(ConflictEntry(
            ConflictType.REDUNDANCY, [from_id, to_id], [],  # 내부 상수 재사용, 표시명은 "Cycle"
            f"dependency cycle detected — edge {from_id}\u2192{to_id} broken (cross-agent edge preferred)",
            "break edge",
        ))
    events.append({
        "type": "final_order",
        "order": [s["step_id"] for s in joint_plan],
    })

    match_failures = handoff_conflicts + [
        ConflictEntry(
            ConflictType.DEPENDENCY, [], [], f"unmatched need item: {nid}",
            "notified — no auto-fix (no target could satisfy this need)",  # Match Failure — 사람에게 통보만, 자동 해소 없음
        )
        for nid in unresolved_needs
    ]

    return {
        "updated_plans": plans,
        "assignments": assignments,
        "conflicts_resolved": resolved,
        "conflicts_unresolved": match_failures,
        "broken_cycle_edges": broken_cycle_edges,
        "orphan_pass_cleaned": cleaned,
        "joint_plan": joint_plan,
        "joint_plan_text": format_joint_plan(joint_plan),
        "graph_events": events,
    }
