# -*- coding: utf-8 -*-
"""
로봇 정보 검색 Tool Plan 검증 모듈

Tool Contract 기준으로 plan을 검증해 PASS / WARNING / FAIL을 반환합니다.

FAIL 유형 (실행 중단 권고):
  unknown_tool      : ALLOWED_TOOLS에 없는 tool 사용
  missing_argument  : required argument 누락 또는 빈 값
  schema_error      : plan item이 dict가 아니거나 필수 필드 부재

WARNING 유형 (실행은 가능하나 품질 문제):
  unknown_argument  : Tool Contract에 없는 argument key 사용
  empty_plan        : plan이 비어 있음 (실행할 tool 없음)
  weak_reason       : reason 필드가 없거나 10자 미만
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from my_tool_selection import TOOL_CONTRACTS, ALLOWED_TOOLS

MIN_REASON_LEN = 10


# ───────────────────────────────────────────────────────────
# 공개 인터페이스
# ───────────────────────────────────────────────────────────
def validate_plan(plan: list) -> dict:
    """
    plan 리스트를 Tool Contract 기준으로 검증합니다.

    매개변수:
      plan : normalize_plan()이 반환한 plan_result["plan"].

    반환:
      {
        "status":  "PASS" | "WARNING" | "FAIL",
        "issues":  list[{"type": str, "severity": str, "message": str}]
      }

    status 결정 우선순위:
      1. FAIL issue가 하나라도 있으면 → "FAIL"
      2. WARNING issue만 있으면      → "WARNING"
      3. 이슈 없음                   → "PASS"

    검증 항목:
      - plan 자체가 비어 있는지 (WARNING)
      - 각 item이 dict인지 (FAIL: schema_error)
      - tool_name이 ALLOWED_TOOLS에 있는지 (FAIL: unknown_tool)
      - required_arguments가 모두 있고 값이 있는지 (FAIL: missing_argument)
      - arguments에 contract 외 key가 있는지 (WARNING: unknown_argument)
      - reason 필드 길이가 MIN_REASON_LEN 이상인지 (WARNING: weak_reason)
    """
    issues    = []
    has_fail  = False
    has_warn  = False

    def _add(severity: str, issue_type: str, message: str):
        nonlocal has_fail, has_warn
        issues.append({"type": issue_type, "severity": severity, "message": message})
        if severity == "FAIL":
            has_fail = True
        else:
            has_warn = True

    # ── 빈 plan 검사 ──────────────────────────────────────
    if not plan:
        _add("WARNING", "empty_plan", "plan이 비어 있습니다. 실행할 tool이 없습니다.")

    for idx, item in enumerate(plan):
        label = f"step {idx + 1}"

        # ── item 타입 검사 ────────────────────────────────
        if not isinstance(item, dict):
            _add("FAIL", "schema_error", f"{label}: plan item이 dict가 아닙니다.")
            continue

        tool_name = item.get("tool_name", "")
        arguments = item.get("arguments") or {}
        reason    = item.get("reason", "")

        # ── tool 이름 검사 ────────────────────────────────
        if tool_name not in ALLOWED_TOOLS:
            _add("FAIL", "unknown_tool",
                 f"{label}: '{tool_name}'은 ALLOWED_TOOLS에 없습니다.")
            continue  # contract가 없으므로 이하 검사 불가

        contract = TOOL_CONTRACTS[tool_name]
        required = contract.get("required_arguments", [])
        optional = contract.get("optional_arguments", [])
        defaults = contract.get("default_values", {})
        known    = set(required) | set(optional) | set(defaults)

        # ── required argument 검사 ────────────────────────
        for key in required:
            val = arguments.get(key)
            if val is None or val == "":
                _add("FAIL", "missing_argument",
                     f"{label} '{tool_name}': 필수 argument '{key}'가 누락됐습니다.")

        # ── unknown argument 검사 ─────────────────────────
        for key in arguments:
            if key not in known:
                _add("WARNING", "unknown_argument",
                     f"{label} '{tool_name}': contract에 없는 argument '{key}' 사용.")

        # ── reason 길이 검사 ──────────────────────────────
        if not reason or not reason.strip():
            _add("WARNING", "weak_reason",
                 f"{label} '{tool_name}': reason 필드가 비어 있습니다.")
        elif len(reason.strip()) < MIN_REASON_LEN:
            _add("WARNING", "weak_reason",
                 f"{label} '{tool_name}': reason이 {MIN_REASON_LEN}자 미만입니다.")

    status = "FAIL" if has_fail else ("WARNING" if has_warn else "PASS")
    return {"status": status, "issues": issues}


# ───────────────────────────────────────────────────────────
# 단독 실행용 self-test
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from my_tool_selection import select_tools
    from my_tool_normalizer import normalize_plan

    # 정상 케이스
    plan_result = normalize_plan(select_tools("ROS2 AMR lidar SLAM"))
    result = validate_plan(plan_result["plan"])
    print(f"[정상] status={result['status']} | issues={result['issues']}")

    # 오류 케이스: required argument 누락
    bad_plan = [
        {
            "step": 1,
            "tool_name": "search_reddit",
            "arguments": {},            # query 누락
            "condition": "always",
            "reason": "테스트",
        },
        {
            "step": 2,
            "tool_name": "nonexistent_tool",  # 허용되지 않는 tool
            "arguments": {"query": "test"},
            "condition": "always",
            "reason": "테스트용 잘못된 tool",
        },
    ]
    result = validate_plan(bad_plan)
    print(f"\n[오류 케이스] status={result['status']}")
    for issue in result["issues"]:
        marker = "❌" if issue["severity"] == "FAIL" else "⚠️ "
        print(f"  {marker} [{issue['type']}] {issue['message']}")
