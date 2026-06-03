# -*- coding: utf-8 -*-
"""
my_mcp_server - Security (민감정보 보호 / 안전한 반환)

[이 모듈의 역할]
크롤러 Tool 결과나 에러 메시지가 외부로 나가기 전에 민감정보를 제거하고,
반환 데이터를 안전한 요약 형태로 제한하는 leaf 계층입니다.

[설계 원칙]
1. leaf 모듈: 표준 라이브러리(re)만 사용. 다른 my_* 모듈을 import하지 않는다.
2. 절대 출력/반환 금지: API Key, token, password, endpoint, URL 전체.
3. sanitize_result()는 tool_name별 '허용 필드 화이트리스트'만 남긴다.
4. 크롤러 결과는 list[dict] 형태이므로 각 항목을 개별 정제한다.
"""
import re

# ---------------------------------------------------------------------------
# 민감정보 key 이름 마커
# ---------------------------------------------------------------------------
SENSITIVE_KEY_MARKERS = (
    "token", "password", "passwd", "pwd", "secret",
    "api_key", "apikey", "authorization", "bearer",
    "credential", "endpoint", "host", "access_key",
)

REDACTED = "[REDACTED]"

_KEYVALUE_PATTERN = re.compile(
    r"(?i)\b("
    r"token|password|passwd|pwd|secret|api[_-]?key|apikey|authorization|"
    r"bearer|credential|endpoint|host|url|access[_-]?key"
    r")\b(\s*[=:]\s*)(\S+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]+=*")
_URL_PATTERN = re.compile(r"(?i)\bhttps?://\S+")


def mask_sensitive_text(text):
    """문자열 안의 민감정보(토큰/비밀번호/URL 등)를 [REDACTED]로 치환한다.

    에러 메시지·로그·인자값처럼 내부 API 엔드포인트가 노출될 수 있는 컨텍스트에 사용한다.
    공개 뉴스 콘텐츠(title, preview 등)에는 mask_content_text를 사용한다.
    """
    s = str(text)
    s = _BEARER_PATTERN.sub("Bearer " + REDACTED, s)
    s = _URL_PATTERN.sub("[REDACTED_URL]", s)
    s = _KEYVALUE_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", s)
    return s


def mask_content_text(text):
    """공개 크롤링 콘텐츠용 마스킹: Bearer·API Key만 제거하고 URL은 그대로 유지한다.

    뉴스 title·preview·summary 같이 공개 URL을 포함하는 텍스트에 사용한다.
    에러 메시지·로그에는 mask_sensitive_text를 사용한다.
    """
    s = str(text)
    s = _BEARER_PATTERN.sub("Bearer " + REDACTED, s)
    s = _KEYVALUE_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", s)
    return s


def safe_error_message(error):
    """예외/에러를 외부로 내보내기 안전한 짧은 문자열로 변환한다."""
    if isinstance(error, BaseException):
        raw = f"{type(error).__name__}: {error}"
    else:
        raw = str(error)
    masked = mask_sensitive_text(raw)
    if len(masked) > 300:
        masked = masked[:300] + "...(생략)"
    return masked


def enforce_limit(value, default=5, maximum=20):
    """조회 limit 값을 안전 범위로 강제한다(과도 전체 조회 방지)."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    if n < 1:
        return default
    if n > maximum:
        return maximum
    return n


def sanitize_arguments(arguments):
    """Tool 인자 dict에서 민감 key의 값을 마스킹한다."""
    if not isinstance(arguments, dict):
        return {}
    safe = {}
    for key, value in arguments.items():
        key_lower = str(key).lower()
        if any(marker in key_lower for marker in SENSITIVE_KEY_MARKERS):
            safe[key] = REDACTED
        elif isinstance(value, str):
            safe[key] = mask_sensitive_text(value)
        else:
            safe[key] = value
    return safe


# ---------------------------------------------------------------------------
# tool_name별 반환 허용 필드 화이트리스트
# ---------------------------------------------------------------------------
# 크롤러 결과는 list[dict]이므로, 각 항목(dict)에서 이 필드만 남긴다.
# URL은 허용 필드에 포함(개발 정보 접근에 필수)하되,
# mask_sensitive_text는 적용하지 않는다(URL이 결과의 핵심 값이기 때문).
_RERANK_FIELDS = ("original_rank", "rerank_score", "rerank_reasons")

RESULT_FIELD_WHITELIST = {
    "search_reddit":     ("title", "url", "date", "preview", "source")       + _RERANK_FIELDS,
    "search_arxiv":      ("title", "authors", "published", "summary", "url") + _RERANK_FIELDS,
    "search_robot_news": ("title", "url", "date", "source", "preview")       + _RERANK_FIELDS,
    "search_irobotnews": ("title", "url", "date", "preview", "source")       + _RERANK_FIELDS,
    # LanceDB 캐시 조회 결과 (robot_multi_source_matrix 테이블 컬럼 기준)
    # vector 컬럼은 my_repositories 에서 이미 제외됨.
    "search_robot_db": (
        "title", "content", "source_type", "url", "origin", "publish_date",
        "data_source", "error",
    ) + _RERANK_FIELDS,
}


def _mask_item_value(value):
    """허용 필드 값을 타입별로 안전하게 마스킹하는 내부 헬퍼.

    크롤러 결과의 공개 콘텐츠(title, preview 등)에 사용하므로
    URL은 마스킹하지 않고 Bearer/API Key만 제거하는 mask_content_text를 사용한다.
    """
    if isinstance(value, str):
        return mask_content_text(value)
    if isinstance(value, list):
        return [_mask_item_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _mask_item_value(v) for k, v in value.items()}
    return value


def _sanitize_single_item(tool_name, item):
    """결과 목록의 항목 dict 하나를 허용 필드만 남겨 정제한다."""
    if not isinstance(item, dict):
        return {"safe_summary": mask_sensitive_text(str(item)[:200])}
    allowed = RESULT_FIELD_WHITELIST.get(tool_name)
    if allowed is None:
        return {"safe_summary": "허용되지 않은 tool 결과입니다."}
    safe = {}
    for key in allowed:
        if key in item:
            val = item[key]
            # url은 마스킹하지 않고 그대로 전달 (개발 링크로 사용)
            safe[key] = val if key == "url" else _mask_item_value(val)
    # 에러 항목은 안전 메시지로 교체
    if "error" in item:
        safe["error"] = mask_sensitive_text(str(item["error"])[:200])
    return safe


def sanitize_result(tool_name, payload):
    """Tool 응답(list[dict])을 tool_name별 허용 필드만 남긴 안전한 리스트로 변환한다.

    [입력]
        tool_name: 실행한 Tool 이름.
        payload  : crawler 함수가 반환한 list[dict].
    [반환]
        list[dict] — 항목마다 허용 필드만 남긴 안전한 결과.
        payload가 list가 아니면 [{safe_summary: ...}]로 축약한다.
    """
    if isinstance(payload, list):
        return [_sanitize_single_item(tool_name, item) for item in payload]
    # 비정상 반환(list가 아님) → 축약
    return [{"safe_summary": mask_sensitive_text(str(payload)[:200])}]


if __name__ == "__main__":
    print("[my_security] mask_sensitive_text 테스트")
    print(mask_sensitive_text("endpoint=https://api.example.com Bearer abc123"))
    print(sanitize_arguments({"query": "ROS2 AMR", "api_key": "secret123"}))
    sample = [{"title": "Test", "url": "https://example.com", "date": "2026-06-01",
               "preview": "token=abc123 in text", "source": "test"}]
    print(sanitize_result("search_reddit", sample))
