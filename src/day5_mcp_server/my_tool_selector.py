# -*- coding: utf-8 -*-
"""
my_mcp_server - Tool Selector (LLM 기반 Tool Selection)

[이 모듈의 역할]
사용자 자연어 질문을 받아 LLM으로 Tool Plan을 '생성'합니다.
Tool을 '실행'하지 않습니다. 생성된 plan은 my_tool_executor가 검증 후 PASS일 때만 실행합니다.

[Tool Selection 전략]
- 기본 흐름: 사용자 질문 → build_selection_prompt → LLM → {"plan": [...]}
- LLM 호출은 프로젝트 공통 seam(src/llm_client.generate_json_response)을 재사용합니다.
- llm_client를 함수 내부에서 '지연 import'하여 MCP Server 전체 import가 깨지지 않게 합니다.

[rule fallback 전략]
크롤러 서버는 항상 4개 tool 전부를 사용하는 것이 최선이므로,
LLM 실패 시 rule fallback도 4개 tool 전체를 실행하는 단순 plan을 만듭니다.
(제조 시나리오의 복잡한 분기 선택과 달리, 개발 정보 수집은 항상 전 채널 수집이 유용)

[보안]
- API Key / endpoint / 환경변수 값은 절대 출력하지 않습니다.
- LLM 응답이 JSON dict가 아니면 실행 가능한 plan으로 취급하지 않습니다.
"""
import inspect
import re

from my_contracts import (
    ALLOWED_TOOLS, TOOL_CONTRACTS,
    TOPIC_KEYWORDS, TOPIC_ENRICHMENT,
)


# ---------------------------------------------------------------------------
# 주제 감지 / 쿼리 확장 헬퍼
# ---------------------------------------------------------------------------

def detect_topics(query: str) -> list:
    """질문에서 HW/ROS/AMR/RFM 주제 키워드를 감지해 목록을 돌려준다."""
    lower = query.lower()
    return [t for t, kws in TOPIC_KEYWORDS.items() if any(k in lower for k in kws)]


def enrich_query(query: str, topics: list) -> str:
    """감지된 모든 주제의 확장 어휘를 쿼리에 덧붙인다."""
    if not topics:
        return query
    extra = " ".join(TOPIC_ENRICHMENT[t] for t in topics if t in TOPIC_ENRICHMENT)
    return f"{query} {extra}".strip() if extra else query


# ---------------------------------------------------------------------------
# LLM 호출 헬퍼
# ---------------------------------------------------------------------------

def build_selection_prompt(user_query: str) -> str:
    """LLM에게 크롤러 Tool Plan 생성을 요청하는 prompt를 만든다.

    [구성]
        1) 역할/제약 선언 (허용 Tool만, 이름·인자 임의 생성 금지)
        2) 사용 가능한 Tool 목록 (이름 + purpose)
        3) Tool Contract 요약 (required/optional)
        4) 사용자 질문
        5) 출력 형식 (JSON 하나만, markdown/코드펜스 금지)
    """
    tool_lines = []
    for name in ALLOWED_TOOLS:
        contract = TOOL_CONTRACTS.get(name, {})
        tool_lines.append(f"- {name}: {contract.get('purpose', '')}")
    available_tools = "\n".join(tool_lines)

    contract_lines = []
    for name in ALLOWED_TOOLS:
        contract = TOOL_CONTRACTS.get(name, {})
        required = contract.get("required_arguments", []) or []
        optional = contract.get("optional_arguments", []) or []
        defaults = contract.get("default_values", {}) or {}
        parts = [name]
        parts.append(f"  - when_to_use: {contract.get('when_to_use', '')}")
        parts.append(f"  - required: {', '.join(required)}")
        if optional:
            parts.append(f"  - optional: {', '.join(optional)} (기본값: {defaults})")
        contract_lines.append("\n".join(parts))
    tool_contracts = "\n".join(contract_lines)

    return (
        "당신은 로봇 개발(HW, ROS, AMR, RFM) 정보를 수집하는 크롤러 MCP Tool 선택 에이전트입니다.\n"
        "사용자 질문을 분석해 사용 가능한 크롤러 Tool 중 필요한 Tool을 선택하고,\n"
        "실행 계획(Tool Plan)을 JSON으로 만드세요.\n"
        "- 제공된 Tool 목록에 있는 Tool만 사용하세요.\n"
        "- 목록에 없는 Tool 이름을 만들지 마세요.\n"
        "- arguments key는 Tool Contract에 정의된 이름만 사용하세요.\n"
        "- query는 반드시 채우세요. 사용자 질문을 그대로 쓰거나 영문으로 번역해 사용하세요.\n"
        "- 개발 정보 수집이 목적이므로 일반적으로 4개 Tool 모두를 실행하는 것이 좋습니다.\n\n"
        f"=== 사용 가능한 Tool 목록 ===\n{available_tools}\n\n"
        f"=== Tool Contract (요약) ===\n{tool_contracts}\n\n"
        f"=== 사용자 질문 ===\n{user_query}\n\n"
        "=== 출력 형식 ===\n"
        "아래 JSON 형식 하나만 출력하세요. markdown, 코드펜스, 주석은 출력하지 마세요.\n"
        '{"plan": [{"step": 1, "tool_name": "", "arguments": {}, '
        '"condition": "always", "reason": ""}]}\n'
        '처리 가능한 Tool이 없으면 {"plan": []} 를 출력하세요.\n'
    )


def call_llm_json(prompt: str):
    """프로젝트 공통 llm_client.generate_json_response를 재사용해 JSON dict를 받는다.

    [반환] (result_dict_or_None, source)
        source: "llm" | "llm_error" | "unavailable"

    [mock LLM 처리]
    mock LLM은 제조 알람 Markdown을 반환하므로 Tool Plan JSON을 생성할 수 없다.
    mock 모드에서는 LLM 호출을 시도하지 않고 즉시 "unavailable"을 반환한다.
    실제 LLM(local/cloud)이 설정된 경우에만 LLM 호출을 시도한다.
    """
    try:
        from llm_client import get_llm_provider
        if get_llm_provider(strict=False) == "mock":
            return None, "unavailable"
    except Exception:
        return None, "unavailable"

    try:
        from llm_client import generate_json_response
    except Exception:
        return None, "unavailable"
    try:
        signature = inspect.signature(generate_json_response)
        if "allow_fallback" in signature.parameters:
            result = generate_json_response(prompt, allow_fallback=False)
        else:
            result = generate_json_response(prompt)
    except Exception:
        return None, "llm_error"
    if not isinstance(result, dict):
        return None, "llm_error"
    if result.get("llm_error_code"):
        return result, "llm_error"
    return result, "llm"


# ---------------------------------------------------------------------------
# Rule fallback plan (LLM 미사용 / 실패 시)
# ---------------------------------------------------------------------------

def build_rule_fallback_plan(user_query: str) -> dict:
    """LLM 사용 불가 시 사용하는 최소 rule 기반 Tool Plan.

    [전략]
        크롤러 4개(Reddit/arXiv/영문뉴스/로봇신문)는 항상 포함한다.
        search_robot_db 는 step 5 로 항상 포함한다.
        LanceDB 미준비 시에도 빈 결과로 안전 fallback 되어 크롤러가 보완한다.

    [반환]
        {"plan": [ {step, tool_name, arguments, condition, reason}, ... ]}
    """
    topics   = detect_topics(user_query)
    enriched = enrich_query(user_query, topics)
    topic_str = ", ".join(topics) if topics else "general"

    return {"plan": [
        {
            "step": 1,
            "tool_name": "search_reddit",
            "arguments": {"query": user_query},
            "condition": "always",
            "reason": f"r/robotics 커뮤니티 최신 토론 및 동향 수집 (주제: {topic_str})",
        },
        {
            "step": 2,
            "tool_name": "search_arxiv",
            "arguments": {"query": enriched},
            "condition": "always",
            "reason": f"arXiv cs.RO 최신 논문으로 기술 트렌드 파악 (주제: {topic_str})",
        },
        {
            "step": 3,
            "tool_name": "search_robot_news",
            "arguments": {"query": user_query},
            "condition": "always",
            "reason": "영문 로봇 전문 뉴스 RSS(Robot Report/Robohub) 수집",
        },
        {
            "step": 4,
            "tool_name": "search_irobotnews",
            "arguments": {"query": user_query},
            "condition": "always",
            "reason": "로봇신문(irobotnews.com) 한국어 뉴스 수집",
        },
        {
            "step": 5,
            "tool_name": "search_robot_db",
            "arguments": {"query": user_query, "data_type": "papers"},
            "condition": "always",
            "reason": "D:/lance_db 로컬 캐시에서 기존 수집 논문/뉴스 재조회(미준비 시 빈 결과)",
        },
    ]}


def build_llm_error_plan(error_code: str, message: str = "") -> dict:
    """strict fallback mode 전용 LLM 실패 plan(빈 plan)."""
    return {
        "llm_error_code":    error_code or "unknown_llm_error",
        "llm_error_message": message or "LLM 실패, strict mode로 rule fallback 미사용",
        "plan": [],
    }


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def select_tool_plan_with_llm(user_query: str, fallback_mode: str = "rule") -> dict:
    """LLM으로 크롤러 Tool Plan을 생성한다(Selection 단계 진입점).

    [입력]
        user_query   : 사용자 자연어 질문.
        fallback_mode: "rule"(기본) 또는 "strict".
            - "rule"  : LLM 실패 시 build_rule_fallback_plan으로 대체(강의 안정성).
            - "strict": LLM 실패 시 build_llm_error_plan(빈 plan) 반환.
    [반환]
        {
          "tool_plan": {"plan": [...]},
          "generation_source": "llm"|"fallback_rule"|"llm_error"|"unavailable",
          "fallback_used": bool,
          "error_message": str | None
        }
    """
    prompt = build_selection_prompt(user_query)
    result, source = call_llm_json(prompt)

    # 1) 정상 LLM 응답 (plan 키 존재)
    if source == "llm" and isinstance(result, dict) and "plan" in result:
        return {
            "tool_plan":         result,
            "generation_source": "llm",
            "fallback_used":     False,
            "error_message":     None,
        }

    # 2) LLM 실패 → fallback_mode에 따라 분기
    if fallback_mode == "strict":
        safe_source = source if source in ("llm_error", "unavailable") else "llm_error"
        return {
            "tool_plan":         build_llm_error_plan(safe_source, "LLM 실패(strict mode)"),
            "generation_source": safe_source,
            "fallback_used":     False,
            "error_message":     "LLM 실패 또는 비정상 응답으로 strict mode 처리",
        }

    # rule(기본): rule fallback plan으로 대체
    return {
        "tool_plan":         build_rule_fallback_plan(user_query),
        "generation_source": "fallback_rule",
        "fallback_used":     True,
        "error_message":     "LLM 사용 불가 또는 실패로 rule fallback 사용",
    }


if __name__ == "__main__":
    q = "ROS2 AMR SLAM 최신 기술 동향"
    plan_result = select_tool_plan_with_llm(q, fallback_mode="rule")
    print(f"[my_tool_selector] source={plan_result['generation_source']}")
    for item in plan_result["tool_plan"].get("plan", []):
        print(f"  step={item['step']} tool={item['tool_name']} "
              f"args={list(item['arguments'].keys())}")
