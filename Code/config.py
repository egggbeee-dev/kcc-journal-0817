MAX_NEW_TOKENS         = 2048
TEMPERATURE            = 0.0   # 재현성을 위해 최대한 결정론적으로 (완전한 결정론 보장은 아님)
MAX_CAN_DO             = 8
MAX_CANNOT_DO          = 5
UNCERTAINTY_THRESH     = 0.50
HQ_TOP_K               = 3
MAX_NEGOTIATION_ROUNDS = 3
AGENT_B_STEP_OFFSET    = 100   # 하위 호환용 (2-agent 코드에서 참조 시). N-agent는 AGENT_STEP_STRIDE 사용
AUTO_HQ_ANSWER: str | None = None

VALID_REASONS  = {"NO_OBJECT", "NO_CAPABILITY", "UNCERTAIN"}
VALID_AGENTS   = {"agent_A", "agent_B"}   # 하위 호환용. N-agent는 build_agents()가 동적으로 생성
VALID_HANDOFFS = {"PASS", "INFORM"}
VALID_PROPOSAL_FIELDS = {"time_min", "action", "depends_on", "delete"}

# ── N-AGENT 확장 설정 ─────────────────────────────────────────────────────────
MIN_AGENTS                          = 2
MAX_AGENTS                          = 8
MIN_ZONE_IMAGES                     = 2
MAX_ZONE_IMAGES                     = 4
AGENT_STEP_STRIDE                   = 100   # agent_index * STRIDE = step_id offset
MIN_ACTIVE_AGENTS                   = 2     # participation 이후 최소 활성 agent 수
PARTICIPATION_CONFIDENCE_THRESHOLD  = 50    # 0-100. OUT 판단의 신뢰도 하한 (미만이면 IN으로 재편입)

# 물리적으로 전달 불가능한 키워드 (can_provide 필터링용, offer._is_passable())
#
# v2 — 가구/고정 설비 키워드 보강. 원래 목록에 "sink/counter/shelf" 같은
# 붙박이 표면류는 있었지만 "table/chair/sofa" 같은 (붙박이는 아니지만
# 통상 손으로 나를 수 없는) 가구류가 빠져 있었음. 그 결과 living_room
# agent의 Offer가 can_provide=["clean dining table"]을 그대로 통과시켰고,
# 이게 뒤 단계(Local Plan)에서 "식탁을 doorway로 나른다"는 PASS 스텝으로
# 이어져 원래 STATE_DEPENDENCY(그래프 매칭)로 처리됐어야 할 need를
# 물리적 handoff로 잘못 표현하게 만든 근본 원인이었음. (auction/그래프
# 로직 자체의 버그가 아니라 이 필터 누락이 원인 — universal_graph.py는
# can_provide에 뭐가 들어오든 그걸 신뢰하고 그대로 처리하기 때문에,
# 여기서 걸러주지 않으면 하류로 그대로 전파됨.)
#
# 이 목록은 여전히 블록리스트(놓치기 쉬움) 방식임 — 새로운 가구/설비
# 이름이 나올 때마다 여기 추가해야 함. 장기적으로는 화이트리스트
# (음식/음료/도구/문서류만 허용)로 바꾸는 게 더 견고하지만, 우선 이번
# 버그를 막는 최소 수정으로 보강함.
NON_PASSABLE_KW = {
    # 표면/구조 요소 (기존)
    "sink", "counter", "shelf", "surface", "floor", "wall",
    # 상태/추상 개념 (기존)
    "cleaned", "wiped", "organized", "confirmation", "confirm",
    "status", "space", "area", "cleared", "tidied", "done",
    # 가구 (신규 — 이번 버그의 직접 원인)
    "table", "chair", "sofa", "couch", "armchair", "bed", "desk",
    "cabinet", "drawer", "dresser", "wardrobe", "bookshelf",
    # 욕실 고정 설비 (신규)
    "bathtub", "tub", "toilet", "shower", "mirror", "rack",
    # 기타 고정 가전/구조물 (신규)
    "tv", "television", "lamp", "window", "curtain", "door",
    "doorway", "carpet", "rug",
}

FUZZY_STOPWORDS = {
    "the", "a", "an", "and", "or", "with", "on", "in", "at", "to", "of",
    "set", "up", "get", "put", "make", "do", "move", "check", "use",
    "take", "open", "close", "place", "arrange", "clean",
}
