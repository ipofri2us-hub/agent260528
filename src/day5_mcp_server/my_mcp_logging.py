# -*- coding: utf-8 -*-
"""
my_mcp_server - MCP Logging (안전한 JSONL 호출 이력 저장)

[이 모듈의 역할]
크롤러 MCP Tool 호출 이벤트를 한 줄 단위 JSONL 로그 파일로 저장합니다.
민감정보(API Key, URL 전체, 쿼리 전문 등)는 절대 남기지 않습니다.

[로그 저장 위치]
  outputs/my_mcp/logs/my_mcp_server.jsonl  (프로젝트 루트 기준)
  환경변수 MY_MCP_LOG_DIR 설정 시 그 경로로 override.

[설계 원칙]
1. import 방향: my_mcp_logging → my_security (단방향). 순환 import 없음.
2. 로그 저장 실패는 Tool 실행 결과에 영향을 주지 않는다(예외 흡수).
3. 허용 필드 화이트리스트만 저장한다(raw query/result 미기록).
4. JSONL은 BOM 없는 utf-8로 기록, 읽을 때는 utf-8-sig로 흡수.
"""
import json
import os
import datetime
from pathlib import Path

from my_security import mask_sensitive_text, sanitize_arguments

# ---------------------------------------------------------------------------
# 경로 / 환경변수 상수
# ---------------------------------------------------------------------------
# 이 파일 위치: <root>/src/my_mcp_logging.py
# parents[0]=src, parents[1]=<root>
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_DIR_ENV   = "MY_MCP_LOG_DIR"
LOG_FILE_NAME = "my_mcp_server.jsonl"


def get_log_dir():
    """로그 디렉터리 경로(Path)를 돌려준다."""
    override = os.environ.get(LOG_DIR_ENV, "").strip()
    if override:
        return Path(override)
    return _PROJECT_ROOT / "outputs" / "my_mcp" / "logs"


def get_log_path():
    """로그 파일 전체 경로를 돌려준다."""
    return get_log_dir() / LOG_FILE_NAME


def ensure_log_dir():
    """로그 디렉터리가 없으면 생성한다. 성공 True / 실패 False."""
    try:
        get_log_dir().mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 로그 허용 필드 화이트리스트
# ---------------------------------------------------------------------------
_ALLOWED_LOG_FIELDS = {
    "timestamp", "event_type",
    "tool_name", "status", "validation_status", "execution_status",
    "executed_count", "generation_source", "fallback_used",
    "user_query_preview", "arguments_preview", "issue_types",
    "error_type", "safe_error_message", "message", "has_fastmcp",
    "result_count",  # 크롤러 전용: 수집된 항목 수
}

_MAX_QUERY_PREVIEW = 80
_MAX_STR = 300


def _query_preview(value):
    """user_query 류 문자열을 '마스킹 + 80자 preview'로 축약한다."""
    text = mask_sensitive_text(str(value))
    if len(text) > _MAX_QUERY_PREVIEW:
        return text[:_MAX_QUERY_PREVIEW] + "...(생략)"
    return text


def _mask_log_value(value):
    """로그 값 하나를 타입별로 안전하게 마스킹한다."""
    if isinstance(value, str):
        masked = mask_sensitive_text(value)
        return masked[:_MAX_STR] + "...(생략)" if len(masked) > _MAX_STR else masked
    if isinstance(value, dict):
        return {k: _mask_log_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask_log_value(v) for v in value]
    return value


def sanitize_log_payload(payload):
    """로그 payload를 '저장해도 되는 안전 필드'만 남긴 dict로 변환한다."""
    if not isinstance(payload, dict):
        return {}
    safe = {}
    # user_query는 preview로만 (원문 전체 금지)
    if "user_query" in payload or "user_query_preview" in payload:
        raw = payload.get("user_query_preview", payload.get("user_query"))
        if raw is not None:
            safe["user_query_preview"] = _query_preview(raw)
    # arguments는 sanitize_arguments 결과로만
    if "arguments" in payload or "arguments_preview" in payload:
        raw_args = payload.get("arguments_preview", payload.get("arguments"))
        if isinstance(raw_args, dict):
            safe["arguments_preview"] = sanitize_arguments(raw_args)
        elif raw_args is not None:
            safe["arguments_preview"] = _mask_log_value(raw_args)
    # 나머지 허용 필드만 통과
    for key, value in payload.items():
        if key in ("user_query", "user_query_preview", "arguments", "arguments_preview"):
            continue
        if key not in _ALLOWED_LOG_FIELDS:
            continue
        safe[key] = _mask_log_value(value)
    return safe


def log_event(event_type, payload):
    """MCP 호출 이벤트 한 건을 JSONL 한 줄로 저장한다.

    [반환] True: 성공, False: 실패(예외 흡수, 메인 실행 보호).
    """
    try:
        record = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "event_type": str(event_type),
        }
        safe = sanitize_log_payload(payload or {})
        safe.pop("event_type", None)
        safe.pop("timestamp", None)
        record.update(safe)
        if not ensure_log_dir():
            return False
        line = json.dumps(record, ensure_ascii=False)
        with open(get_log_path(), "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return True
    except Exception:
        return False


def read_recent_logs(limit=50):
    """최근 로그 N건을 dict 리스트로 돌려준다(내부 디버깅 전용).

    [중요] FastMCP Tool로 노출하지 않는다.
    """
    try:
        path = get_log_path()
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        records = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
        return records
    except Exception:
        return []


if __name__ == "__main__":
    ok = log_event("test_event", {
        "tool_name": "search_arxiv",
        "user_query": "ROS2 AMR navigation planning",
        "arguments": {"query": "ROS2 AMR", "api_key": "secret"},
        "status": "executed",
        "result_count": 5,
    })
    print(f"[my_mcp_logging] log_event: {'성공' if ok else '실패'}")
    logs = read_recent_logs(1)
    if logs:
        print(f"  마지막 로그: {logs[-1]}")
