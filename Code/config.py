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

# 물리적으로 전달 불가능한 키워드 (can_provide 필터링용)
NON_PASSABLE_KW = {
    "sink", "counter", "shelf", "surface", "floor", "wall",
    "cleaned", "wiped", "organized", "confirmation", "confirm",
    "status", "space", "area", "cleared", "tidied", "done",
}

FUZZY_STOPWORDS = {
    "the", "a", "an", "and", "or", "with", "on", "in", "at", "to", "of",
    "set", "up", "get", "put", "make", "do", "move", "check", "use",
    "take", "open", "close", "place", "arrange", "clean",
}
