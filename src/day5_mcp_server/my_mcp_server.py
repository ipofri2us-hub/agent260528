# -*- coding: utf-8 -*-
"""
my_mcp_server - MCP Server 진입점 (로봇 개발 정보 크롤러)

[이 모듈의 역할]
로봇 개발(HW, ROS, AMR, RFM) 정보를 수집하는 크롤러 MCP Server의 진입점입니다.
arXiv, Reddit r/robotics, 로봇신문(irobotnews.com), Robot Report/Robohub RSS를
대상으로 개발 관련 뉴스·논문·커뮤니티 정보를 수집합니다.

[크롤링 대상]
  1. arXiv cs.RO   : 최신 로봇공학 논문 (Atom API)
  2. Reddit r/robotics : 커뮤니티 토론 (Google News RSS 경유)
  3. 로봇신문(irobotnews.com) : 한국어 뉴스 (Google News RSS 경유)
  4. Robot Report / Robohub : 영문 뉴스 (RSS)

[제공하는 진입 함수]
- list_mcp_tools()                    : 노출 Tool schema 목록
- call_mcp_tool(tool_name, arguments) : 개별 Tool 직접 호출
- call_select_tools_for_query(query)  : LLM으로 Tool Plan 생성(실행 안 함)
- call_validate_tool_plan(tool_plan)  : 검증 결과만 반환
- call_execute_selected_tools(plan)   : 이미 만들어진 Tool Plan 실행
- call_execute_query(user_query)      : LLM 생성 → 검증 → 실행(end-to-end)

[책임 분리]
- Selection(생성)     : my_tool_selector
- Execution(실행)     : my_tool_executor
- Schema 노출         : my_mcp_schemas
- 호출 이력 로깅      : my_mcp_logging

[FastMCP 선택 의존성]
FastMCP가 설치돼 있으면 create_fastmcp_server()로 Tool wrapper를 등록해 사용한다.
설치돼 있지 않아도 함수형 API는 정상 동작한다.

[보안]
API Key / token / endpoint / 환경변수 값은 반환/로깅하지 않습니다.
"""
from my_contracts import ALLOWED_TOOLS
from my_mcp_schemas import build_mcp_tool_schemas
from my_tool_selector import select_tool_plan_with_llm
from my_tool_executor import build_single_tool_plan, execute_tool_plan
from my_mcp_logging import log_event

# ---------------------------------------------------------------------------
# FastMCP 선택 의존성 (없어도 함수형 API는 정상)
# ---------------------------------------------------------------------------
try:
    from fastmcp import FastMCP
    _HAS_FASTMCP = True
except Exception:
    FastMCP = None
    _HAS_FASTMCP = False


# ---------------------------------------------------------------------------
# 로깅 헬퍼
# ---------------------------------------------------------------------------

def _issue_types(issues):
    """validation issues(list[dict])에서 'type' 값만 추려 list[str]로 돌려준다."""
    if not isinstance(issues, list):
        return []
    return [i.get("type") for i in issues if isinstance(i, dict) and i.get("type")]


def _total_result_count(results):
    """results 목록에서 수집된 전체 항목 수를 합산한다."""
    if not isinstance(results, list):
        return 0
    return sum(r.get("result_count", 0) for r in results if isinstance(r, dict))


def _log_execution_result(tool_name: str, result: dict):
    """실행 결과 dict를 보고 상태별 로그 이벤트를 1건 남긴다."""
    if not isinstance(result, dict):
        return
    status = result.get("status")
    event_map = {
        "executed":     "tool_executed",
        "dry_run":      "tool_executed",
        "rejected":     "tool_rejected",
        "needs_review": "tool_needs_review",
        "error":        "tool_error",
    }
    log_event(event_map.get(status, "tool_executed"), {
        "tool_name":         tool_name,
        "status":            status,
        "validation_status": result.get("validation_status"),
        "executed_count":    result.get("executed_count"),
        "result_count":      _total_result_count(result.get("results", [])),
        "issue_types":       _issue_types(result.get("issues")),
    })


# ---------------------------------------------------------------------------
# 함수형 API (FastMCP 없이도 직접 호출 가능)
# ---------------------------------------------------------------------------

def list_mcp_tools() -> list:
    """노출할 MCP Tool schema 목록을 반환한다(my_contracts 기준).

    [반환] list[dict] — build_mcp_tool_schemas() 결과.
    [용도] MCP Client가 사용 가능한 Tool과 input schema를 확인할 때 사용한다.
    """
    return build_mcp_tool_schemas()


def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """개별 MCP Tool을 직접 호출한다(검증 후 PASS일 때만 실행).

    [입력]
        tool_name: 호출할 Tool 이름.
        arguments: 호출 인자 dict.
    [반환]
        execute_tool_plan()의 결과 dict.

    [동작]
        1) 허용 목록에 없으면 즉시 rejected.
        2) build_single_tool_plan으로 단일 Tool Plan 생성.
        3) execute_tool_plan으로 검증 → PASS면 실행.
    """
    log_event("tool_called", {
        "tool_name": tool_name,
        "arguments": arguments,
    })

    if tool_name not in ALLOWED_TOOLS:
        result = {
            "status":            "rejected",
            "validation_status": "FAIL",
            "executed_count":    0,
            "results":           [],
            "issues":            [],
            "message":           "허용 목록에 없는 Tool이므로 실행하지 않습니다.",
        }
        _log_execution_result(tool_name, result)
        return result

    plan = build_single_tool_plan(tool_name, arguments)
    result = execute_tool_plan(plan, user_query="")
    _log_execution_result(tool_name, result)
    return result


def call_select_tools_for_query(user_query: str,
                                fallback_mode: str = "rule") -> dict:
    """사용자 질문으로 LLM Tool Plan을 '생성'만 한다(실행하지 않음).

    [반환]
        my_tool_selector.select_tool_plan_with_llm() 결과
        (tool_plan / generation_source / fallback_used / error_message).
    """
    selection = select_tool_plan_with_llm(user_query, fallback_mode=fallback_mode)
    log_event("tool_selected", {
        "user_query":        user_query,
        "generation_source": selection.get("generation_source"),
        "fallback_used":     selection.get("fallback_used"),
    })
    return selection


def call_validate_tool_plan(tool_plan: dict, user_query: str = "") -> dict:
    """주어진 Tool Plan을 검증한 결과를 반환한다(실행 안 함).

    [반환]
        {
          "validation_status": "PASS"|"WARNING"|"FAIL",
          "issues": [...],
          "executable": bool
        }
    """
    exec_result = execute_tool_plan(tool_plan, user_query=user_query, dry_run=True)
    validation_status = exec_result.get("validation_status", "FAIL")
    issues = exec_result.get("issues", [])
    log_event("tool_validated", {
        "validation_status": validation_status,
        "issue_types":       _issue_types(issues),
    })
    return {
        "validation_status": validation_status,
        "issues":            issues,
        "executable":        validation_status in ("PASS", "WARNING"),
    }


def call_execute_selected_tools(tool_plan: dict, user_query: str = "",
                                dry_run: bool = False) -> dict:
    """이미 만들어진 Tool Plan을 실행한다(검증 후 PASS면 실행).

    [용도]
        Selection과 Execution을 분리해 호출하고 싶을 때 사용한다.
    """
    result = execute_tool_plan(tool_plan, user_query=user_query, dry_run=dry_run)
    _log_execution_result("(tool_plan)", result)
    return result


def call_execute_query(user_query: str, fallback_mode: str = "rule",
                       dry_run: bool = False) -> dict:
    """사용자 질문 → LLM Tool Plan 생성 → 검증 → 실행하는 end-to-end 진입점.

    [입력]
        user_query   : 사용자 자연어 질문 (한국어/영어 모두 가능).
        fallback_mode: "rule"(기본) 또는 "strict".
        dry_run      : True면 실제 크롤링 없이 실행 계획만 확인.
    [반환]
        execute_tool_plan 결과 dict + generation_source/fallback_used.

    [동작]
        1) my_tool_selector로 LLM Tool Plan 생성(실패 시 fallback_mode 정책 적용).
        2) my_tool_executor로 검증 → PASS면 실행.
        3) Selection 출처 정보를 결과에 함께 담는다.
    """
    log_event("tool_called", {"user_query": user_query})

    # 1) Selection
    selection = select_tool_plan_with_llm(user_query, fallback_mode=fallback_mode)
    log_event("tool_selected", {
        "generation_source": selection.get("generation_source"),
        "fallback_used":     selection.get("fallback_used"),
    })

    # 2) Execution
    exec_result = execute_tool_plan(
        selection.get("tool_plan", {"plan": []}),
        user_query=user_query,
        dry_run=dry_run,
    )

    # 3) Selection 출처 정보 추가
    exec_result["generation_source"]       = selection.get("generation_source")
    exec_result["fallback_used"]           = selection.get("fallback_used")
    exec_result["selection_error_message"] = selection.get("error_message")
    exec_result["llm_error_code"]          = selection.get("tool_plan", {}).get("llm_error_code")

    _log_execution_result("(query)", exec_result)
    return exec_result


# ---------------------------------------------------------------------------
# FastMCP wrapper 계층 (선택 의존성 — 설치돼 있을 때만 사용)
# ---------------------------------------------------------------------------
# 아래 _fastmcp_* 함수는 위 함수형 API를 '그대로 호출'하는 얇은 계층이다.
# 로그는 기존 진입 함수에서 이미 남기므로, wrapper는 중복 기록하지 않는다.

def has_fastmcp() -> bool:
    """FastMCP 설치/임포트 가능 여부를 돌려준다."""
    return _HAS_FASTMCP


def _fastmcp_execute_query(user_query: str, fallback_mode: str = "rule",
                           dry_run: bool = False) -> dict:
    """[MCP Tool: execute_query] 질문 → 생성 → 실행(end-to-end 핵심)."""
    return call_execute_query(user_query, fallback_mode=fallback_mode, dry_run=dry_run)


def _fastmcp_select_tools_for_query(user_query: str,
                                    fallback_mode: str = "rule") -> dict:
    """[MCP Tool: select_tools_for_query] LLM Tool Selection만 수행(실행 안 함)."""
    return call_select_tools_for_query(user_query, fallback_mode=fallback_mode)


def _fastmcp_validate_tool_plan(tool_plan: dict, user_query: str = "") -> dict:
    """[MCP Tool: validate_tool_plan] Tool Plan 검증 결과만 반환."""
    return call_validate_tool_plan(tool_plan, user_query=user_query)


def _fastmcp_execute_selected_tools(tool_plan: dict, user_query: str = "",
                                    dry_run: bool = False) -> dict:
    """[MCP Tool: execute_selected_tools] 이미 만들어진 Tool Plan을 실행."""
    return call_execute_selected_tools(tool_plan, user_query=user_query, dry_run=dry_run)


def _fastmcp_call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """[MCP Tool: call_mcp_tool] 개별 Tool 직접 호출."""
    return call_mcp_tool(tool_name, arguments)


def _fastmcp_list_mcp_tools() -> list:
    """[MCP Tool: list_mcp_tools] 노출 Tool schema 목록(교육/디버깅용)."""
    return list_mcp_tools()


def create_fastmcp_server():
    """FastMCP server 객체를 생성하고 Tool wrapper를 등록해 돌려준다.

    [반환]
        - FastMCP 설치 + 생성 성공: FastMCP server 객체.
        - 미설치 또는 오류: None.

    [등록 Tool]
        execute_query / select_tools_for_query / validate_tool_plan /
        execute_selected_tools / call_mcp_tool / list_mcp_tools
    """
    if not _HAS_FASTMCP:
        return None
    try:
        server = FastMCP("my_mcp_server")
        server.tool(name="execute_query")(_fastmcp_execute_query)
        server.tool(name="select_tools_for_query")(_fastmcp_select_tools_for_query)
        server.tool(name="validate_tool_plan")(_fastmcp_validate_tool_plan)
        server.tool(name="execute_selected_tools")(_fastmcp_execute_selected_tools)
        server.tool(name="call_mcp_tool")(_fastmcp_call_mcp_tool)
        server.tool(name="list_mcp_tools")(_fastmcp_list_mcp_tools)
        return server
    except Exception:
        return None


def run_fastmcp_server():
    """FastMCP server를 실행한다(설치돼 있을 때만).

    [반환]
        - 미설치: 안내 dict(예외 없음).
        - 정상 실행 후 종료: 종료 안내 dict.

    [주의]
        server.run()은 블로킹 실행이다.
        미설치 환경에서는 함수형 API를 그대로 사용하면 된다.
    """
    if not _HAS_FASTMCP:
        log_event("fastmcp_unavailable", {
            "message": "FastMCP가 설치되어 있지 않습니다. 함수형 API를 사용하세요.",
        })
        return {
            "status":  "unavailable",
            "message": "FastMCP가 설치되어 있지 않습니다. 함수형 API를 사용하세요.",
        }

    server = create_fastmcp_server()
    if server is None:
        return {"status": "unavailable", "message": "FastMCP server 생성에 실패했습니다."}

    log_event("server_started", {"has_fastmcp": True})
    try:
        server.run()
    except Exception as error:
        return {"status": "error", "message": type(error).__name__}
    return {"status": "stopped"}


# ---------------------------------------------------------------------------
# __main__ : 함수형 API 동작 확인 또는 FastMCP 기동
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    args = sys.argv[1:]

    if "--server" in args:
        # FastMCP 서버 기동 모드
        print(f"[my_mcp_server] FastMCP 기동 시도 (has_fastmcp={has_fastmcp()})")
        result = run_fastmcp_server()
        print(result)

    elif "--list" in args:
        # Tool schema 확인
        schemas = list_mcp_tools()
        print(f"[my_mcp_server] 노출 Tool 수: {len(schemas)}")
        for s in schemas:
            print(f"  {s['name']}: {s['description'][:60]}")

    else:
        # 기본: 실제 크롤링 실행 후 결과 출력
        dry = "--dry-run" in args
        query = " ".join(a for a in args if not a.startswith("--")) or "ROS2 AMR SLAM 최신 동향"

        import sys as _sys
        if hasattr(_sys.stdout, "reconfigure"):
            try:
                _sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass

        mode_label = "[dry_run]" if dry else "[실행]"
        print(f"\n{'=' * 64}")
        print(f"  로봇 개발 정보 수집  {mode_label}")
        print(f"  쿼리: {query}")
        print(f"{'=' * 64}")

        result = call_execute_query(query, dry_run=dry)
        src = result.get("generation_source", "")
        print(f"  Tool 선택: {src}  |  실행: {result['executed_count']}개  |  검증: {result['validation_status']}")

        for r in result.get("results", []):
            tool  = r["tool_name"]
            step  = r.get("step", "?")
            items = r.get("results", [])
            count = r.get("result_count", len(items))

            label_map = {
                "search_reddit":     f"r/robotics ({count}건)",
                "search_arxiv":      f"arXiv cs.RO ({count}건)",
                "search_robot_news": f"영문 뉴스 RSS ({count}건)",
                "search_irobotnews": f"로봇신문 ({count}건)",
            }
            print(f"\n  [{step}] {label_map.get(tool, tool)}")
            print(f"  {'─' * 58}")

            if dry:
                print("       (dry_run: 실제 수집하지 않음)")
                continue

            if not items:
                print("       결과 없음")
                continue

            for idx, item in enumerate(items[:3], 1):
                if "error" in item:
                    print(f"  {idx}. ⚠  {item['error']}")
                    continue
                title = item.get("title", "(제목 없음)")[:60]
                date  = item.get("date", item.get("published", ""))[:10]
                url   = item.get("url", "")
                print(f"  {idx}. {title}")
                if date:
                    print(f"       [{date}]")
                # arXiv는 authors/summary 별도 출력
                if tool == "search_arxiv":
                    authors = ", ".join(item.get("authors", []))
                    if authors:
                        print(f"       저자: {authors[:55]}")
                    summary = item.get("summary", "")
                    if summary:
                        print(f"       {summary[:100]}...")
                else:
                    preview = item.get("preview", "")
                    if preview:
                        print(f"       {preview[:90]}")
                if url:
                    print(f"       >> {url[:80]}")

        if result.get("issues"):
            print(f"\n  ⚠  검증 이슈:")
            for i in result["issues"]:
                print(f"     [{i['severity']}] {i['type']}: {i['message']}")
        print(f"\n{'=' * 64}")
