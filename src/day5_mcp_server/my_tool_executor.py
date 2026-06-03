# -*- coding: utf-8 -*-
"""
my_mcp_server - Tool Executor (검증된 Tool Plan 실행 계층)

[이 모듈의 역할]
Tool Plan을 받아 간단한 검증(허용 Tool + 필수 인자 확인)을 수행하고,
PASS일 때만 my_crawlers의 함수를 실행하는 공통 실행 계층입니다.

[Tool Selection과 Tool Execution의 분리]
- Selection(어떤 Tool을 쓸지 결정)은 my_tool_selector가 담당합니다.
- Execution(검증된 plan을 실제 크롤러 호출로 실행)은 이 모듈이 담당합니다.

[검증 게이트 정책]
    1) tool_name이 ALLOWED_TOOLS에 있는가?
    2) required_arguments가 모두 채워져 있는가?
    → PASS  : 실행
    → FAIL  : rejected (실행 안 함)
    → dry_run=True : 실제 호출 없이 계획만 확인

[기본값 채우기]
Tool Contract의 default_values로 누락된 optional argument를 자동 보완합니다.

[보안]
- ALLOWED_TOOLS 밖의 tool은 절대 실행하지 않습니다.
- 각 Tool 결과는 security.sanitize_result()로 정제해 반환합니다.
- 예외는 security.safe_error_message()로 안전하게 변환합니다.
"""
from my_contracts import ALLOWED_TOOLS, TOOL_CONTRACTS, MIN_REASON_LENGTH
from my_crawlers import CRAWLER_DISPATCH
from my_security import sanitize_result, safe_error_message, sanitize_arguments
from my_reranking import rerank_crawler_results


# ---------------------------------------------------------------------------
# 내부 검증 / 정규화 헬퍼
# ---------------------------------------------------------------------------

def _fill_defaults(tool_name: str, arguments: dict) -> dict:
    """Tool Contract의 default_values로 누락된 optional argument를 채운다."""
    defaults = TOOL_CONTRACTS.get(tool_name, {}).get("default_values", {}) or {}
    args = dict(arguments or {})
    for key, default_val in defaults.items():
        if args.get(key) in (None, ""):
            args[key] = default_val
    return args


def _validate_tool_item(item: dict) -> tuple:
    """plan 항목 하나를 검증한다.

    [반환] (validation_status, issues)
        validation_status: "PASS" | "FAIL"
        issues: list[{"type": str, "severity": str, "message": str}]
    """
    issues = []

    if not isinstance(item, dict):
        return "FAIL", [{"type": "schema_error", "severity": "FAIL",
                         "message": "plan item이 dict가 아닙니다."}]

    tool_name = item.get("tool_name", "")
    arguments = item.get("arguments") or {}

    # 1) 허용 Tool 확인
    if tool_name not in ALLOWED_TOOLS:
        issues.append({
            "type":     "unknown_tool",
            "severity": "FAIL",
            "message":  f"허용 목록에 없는 Tool입니다: {tool_name!r}",
        })

    # 2) 필수 argument 확인
    contract = TOOL_CONTRACTS.get(tool_name, {})
    for arg in contract.get("required_arguments", []):
        if not arguments.get(arg):
            issues.append({
                "type":     "missing_argument",
                "severity": "FAIL",
                "message":  f"{tool_name}: 필수 인자 '{arg}'가 없거나 비어 있습니다.",
            })

    # 3) reason 품질 확인 (WARNING)
    reason = item.get("reason", "")
    if len(reason) < MIN_REASON_LENGTH:
        issues.append({
            "type":     "weak_reason",
            "severity": "WARNING",
            "message":  f"reason이 {MIN_REASON_LENGTH}자 미만입니다 ({len(reason)}자).",
        })

    has_fail = any(i["severity"] == "FAIL" for i in issues)
    return ("FAIL" if has_fail else ("PASS" if not issues else "WARNING")), issues


def _validate_plan(tool_plan: dict) -> tuple:
    """Tool Plan 전체를 검증한다.

    [반환] (validation_status, all_issues, plan_items)
    """
    if not isinstance(tool_plan, dict):
        return "FAIL", [{"type": "schema_error", "severity": "FAIL",
                         "message": "tool_plan이 dict가 아닙니다."}], []

    plan_items = tool_plan.get("plan", [])
    if not isinstance(plan_items, list):
        return "FAIL", [{"type": "schema_error", "severity": "FAIL",
                         "message": "tool_plan.plan이 list가 아닙니다."}], []

    if not plan_items:
        return "WARNING", [{"type": "empty_plan", "severity": "WARNING",
                            "message": "plan이 비어 있습니다."}], []

    all_issues = []
    worst = 0  # PASS=0, WARNING=1, FAIL=2
    severity_rank = {"PASS": 0, "WARNING": 1, "FAIL": 2}

    for item in plan_items:
        status, issues = _validate_tool_item(item)
        all_issues.extend(issues)
        rank = severity_rank.get(status, 0)
        if rank > worst:
            worst = rank

    status_map = {0: "PASS", 1: "WARNING", 2: "FAIL"}
    return status_map[worst], all_issues, plan_items


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def build_single_tool_plan(tool_name: str, arguments: dict) -> dict:
    """단일 Tool 호출을 표준 Tool Plan 형태로 감싼다.

    [반환]
        {"plan": [{"step":1, "tool_name":..., "arguments":...,
                   "condition":"always", "reason":"<설명>"}]}
    """
    return {
        "plan": [{
            "step":      1,
            "tool_name": tool_name,
            "arguments": dict(arguments or {}),
            "condition": "always",
            "reason":    f"{tool_name} 단일 Tool을 직접 실행하기 위한 요청입니다.",
        }]
    }


def execute_single_tool(tool_name: str, arguments: dict, dry_run: bool = False) -> dict:
    """단일 Tool을 실제로 실행(또는 dry_run)하고 안전한 결과 dict를 반환한다.

    [반환]
        {
          "tool_name": ...,
          "status": "executed"|"dry_run"|"rejected"|"error",
          "results": list[dict],    # 실행 시 sanitize_result 정제 결과
          "result_count": int,
          "message": str
        }
    """
    args = _fill_defaults(tool_name, arguments)

    # 1) 허용 Tool 차단
    if tool_name not in ALLOWED_TOOLS:
        return {
            "tool_name":    tool_name,
            "status":       "rejected",
            "results":      [],
            "result_count": 0,
            "message":      "허용 목록에 없는 Tool이므로 실행하지 않습니다.",
        }

    # 2) dry_run
    if dry_run:
        return {
            "tool_name":         tool_name,
            "status":            "dry_run",
            "results":           [],
            "result_count":      0,
            "message":           "dry_run 모드: 실제 크롤링을 수행하지 않았습니다.",
            "arguments_preview": sanitize_arguments(args),
        }

    # 3) 실제 실행
    try:
        dispatch = CRAWLER_DISPATCH.get(tool_name)
        if dispatch is None:
            return {
                "tool_name":    tool_name,
                "status":       "rejected",
                "results":      [],
                "result_count": 0,
                "message":      "크롤러 함수가 등록되지 않은 Tool입니다.",
            }
        func, arg_names = dispatch
        call_kwargs = {k: args[k] for k in arg_names if k in args}
        raw_results = func(**call_kwargs)
        query = args.get("query", "")
        reranked = rerank_crawler_results(query, raw_results, tool_name=tool_name)
        safe_results = sanitize_result(tool_name, reranked)
        return {
            "tool_name":    tool_name,
            "status":       "executed",
            "results":      safe_results,
            "result_count": len(safe_results),
            "message":      f"{len(safe_results)}건 수집 완료.",
        }
    except Exception as error:
        return {
            "tool_name":    tool_name,
            "status":       "error",
            "results":      [],
            "result_count": 0,
            "message":      safe_error_message(error),
        }


def execute_tool_plan(tool_plan: dict, user_query: str = "",
                      allow_warning_execute: bool = True,
                      dry_run: bool = False) -> dict:
    """Tool Plan을 검증한 뒤, PASS(또는 WARNING)일 때만 순서대로 실행한다.

    [입력]
        tool_plan            : {"plan": [...]} 형태.
        user_query           : 로깅용(실행에 영향 없음).
        allow_warning_execute: True(기본)면 WARNING도 실행(크롤러는 WARNING 허용).
        dry_run              : True면 실제 크롤링 없이 실행 계획만 확인.
    [반환]
        {
          "status": "executed"|"dry_run"|"needs_review"|"rejected"|"error",
          "validation_status": "PASS"|"WARNING"|"FAIL",
          "executed_count": int,
          "results": [ {tool_name, status, results, result_count, message}, ... ],
          "issues": [...],
          "message": str
        }
    """
    try:
        validation_status, issues, plan_items = _validate_plan(tool_plan)
    except Exception as error:
        return {
            "status":            "error",
            "validation_status": "FAIL",
            "executed_count":    0,
            "results":           [],
            "issues":            [],
            "message":           safe_error_message(error),
        }

    # 빈 plan
    if not plan_items:
        return {
            "status":            "rejected",
            "validation_status": validation_status,
            "executed_count":    0,
            "results":           [],
            "issues":            issues,
            "message":           "실행할 Tool이 없습니다(빈 plan).",
        }

    # FAIL → rejected
    if validation_status == "FAIL":
        return {
            "status":            "rejected",
            "validation_status": validation_status,
            "executed_count":    0,
            "results":           [],
            "issues":            issues,
            "message":           "검증 결과가 FAIL이므로 실행하지 않습니다.",
        }

    # WARNING + allow_warning_execute=False → needs_review
    if validation_status == "WARNING" and not allow_warning_execute:
        return {
            "status":            "needs_review",
            "validation_status": validation_status,
            "executed_count":    0,
            "results":           [],
            "issues":            issues,
            "message":           "검증 결과가 WARNING입니다. 실행하지 않고 검토가 필요합니다.",
        }

    # PASS(또는 허용된 WARNING) → step 순서대로 실행
    results = []
    executed_count = 0
    for idx, item in enumerate(plan_items):
        if not isinstance(item, dict):
            continue
        tool_name = item.get("tool_name")
        arguments = item.get("arguments") or {}
        step      = item.get("step") if item.get("step") is not None else idx + 1

        single = execute_single_tool(tool_name, arguments, dry_run=dry_run)
        single["step"] = step
        results.append(single)
        if single.get("status") in ("executed", "dry_run"):
            executed_count += 1

    overall = "dry_run" if dry_run else "executed"
    return {
        "status":            overall,
        "validation_status": validation_status,
        "executed_count":    executed_count,
        "results":           results,
        "issues":            issues,
        "message":           (
            "검증을 통과하여 Tool Plan을 실행했습니다."
            if not dry_run else "dry_run 모드로 실행 계획만 확인했습니다."
        ),
    }


if __name__ == "__main__":
    plan = build_single_tool_plan("search_irobotnews", {"query": "ROS2 AMR"})
    result = execute_tool_plan(plan, dry_run=True)
    print(f"[my_tool_executor] dry_run status={result['status']} "
          f"validation={result['validation_status']}")
    for r in result["results"]:
        print(f"  tool={r['tool_name']} status={r['status']}")
