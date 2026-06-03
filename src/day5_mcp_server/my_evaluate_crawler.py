# -*- coding: utf-8 -*-
"""
day5_mcp_server - 크롤러 검색 성능 평가 스크립트

[위치]
이 파일은 src/day5_mcp_server/ 에 있습니다.
실행: uv run python src/day5_mcp_server/my_evaluate_crawler.py [--limit N] [--case-id CRAWL-EVAL-001]

[목적]
my_evaluation_cases.json 평가 케이스를 '그대로 재사용'하되,
검색 실행 경로는 day5_mcp_server 의 실제 MCP Tool 흐름
(call_execute_query → my_tool_selector → my_tool_executor → my_crawlers → my_reranking)
을 사용해 크롤러 검색 성능을 재측정한다.

[평가 지표]
- keyword_any_hit      : 쿼리 키워드가 결과 title/preview 에 하나라도 포함되는가
- keyword_all_hit      : 쿼리 키워드가 결과에 모두 포함되는가
- tool_selection_match : 실행된 Tool 이 expected_tools 와 하나라도 겹치는가
- real_source_count    : 실제 데이터(비-에러, 비-fallback)를 반환한 소스 수
- rerank_applied       : reranking 이 적용되었는가(rerank_score 필드 존재)
- top_result_keyword_hit : 최상위 결과(rerank 기준)에 키워드가 있는가
- forbidden_field_found  : 금지 필드/패턴 노출 여부(보안)

[오버피팅 방지 — 매우 중요]
- expected_tools / expected_keywords 는 채점/분석에만 사용하고
  쿼리 실행·rerank·선택 로직에는 사용하지 않는다.
- 특정 case_id 전용 분기를 두지 않는다.

[이 파일의 성격 — 교육용으로 꼭 이해할 점]
- 이 스크립트는 '합격/불합격을 가르는 단위 테스트'가 아니라 크롤러 검색 품질을 '측정·기록'하는
  평가 하베스트(harness)입니다.
- 검색 실행은 같은 프로세스 안에서 call_execute_query() 를 직접 호출합니다(in-process).
  별도 서버 프로세스/transport, async/await, 호출 timeout 은 존재하지 않습니다.

[전체 실행 흐름 한눈에]
main() → run() → load_cases() 로 케이스 로드
  → 케이스마다 evaluate_case() 호출
  → call_execute_query() 로 크롤러 실행
  → evaluate_*() 지표 함수로 채점 → judge_status() 로 케이스 상태 판정
  → build_summary() / build_report() 로 집계·보고서 작성
  → outputs/day5_mcp_server/crawler_eval/ 저장
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 경로 설정 (항상 프로젝트 루트 기준, 어디서 실행해도 동일)
# ---------------------------------------------------------------------------
# 이 파일: <root>/src/day5_mcp_server/my_evaluate_crawler.py
#   parents[0]=day5_mcp_server, [1]=src, [2]=<root>
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

CASES_PATH = _THIS_DIR / "my_evaluation_cases.json"
OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "day5_mcp_server" / "crawler_eval"

# ---------------------------------------------------------------------------
# 금지 필드 / 패턴 (보안 검증)
# ---------------------------------------------------------------------------
# 결과 dict 에 절대 섞여 나오면 안 되는 내부 key.
# rerank_score/rerank_reasons/original_rank 는 허용 필드이므로 금지 목록에 없다.
FORBIDDEN_KEYS = ("api_key", "token", "password", "secret", "bearer", "credential")
# 에러 메시지 등에 credential 패턴이 들어 있으면 보안 위협으로 본다.
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(token|password|api[_-]?key|bearer|secret|credential)\b\s*[=:]\s*\S+",
)

# ---------------------------------------------------------------------------
# quality gate 임계값 (리포트용 기준선; exit code 는 실패로 만들지 않음)
# ---------------------------------------------------------------------------
QUALITY_GATE_THRESHOLDS = {
    "min_keyword_any_hit_rate":   0.70,  # 70% 이상 케이스: 키워드가 결과에 하나라도 포함
    "min_rerank_applied_rate":    0.80,  # 80% 이상 케이스: reranking 이 적용됨
    "min_real_source_rate":       0.50,  # 50% 이상 케이스: 실제 데이터(비-에러/비-fallback) 소스 존재
    "max_forbidden_field_case_count": 0, # 보안 — 금지 패턴이 노출된 케이스 수는 0
    "max_fail_count":             3,     # FAIL 케이스 수 최대 3
}


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def _norm(value) -> str:
    """비교용 정규화: 문자열화 + 앞뒤 공백 제거 + 소문자."""
    return str(value if value is not None else "").strip().lower()


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def load_cases() -> list[dict]:
    """평가 케이스 JSON 을 읽어 list[dict] 로 돌려준다(읽기 전용).

    encoding='utf-8-sig' 는 Windows BOM 처리를 위한 이 저장소 공통 규칙이다.
    """
    raw = CASES_PATH.read_text(encoding="utf-8-sig")
    data = json.loads(raw)
    return [c for c in data if isinstance(c, dict)]


# ---------------------------------------------------------------------------
# 채점 함수
# ---------------------------------------------------------------------------

def _item_text(item: dict) -> str:
    """크롤러 결과 항목 1건의 title + preview/summary 를 소문자 합본으로 반환한다."""
    return " ".join([
        str(item.get("title",   "") or ""),
        str(item.get("preview", item.get("summary", "")) or ""),
    ]).lower()


def _all_items(tool_results: list) -> list[dict]:
    """tool_results 목록에서 모든 결과 항목을 펼쳐 flat list 로 만든다."""
    items = []
    for tr in tool_results:
        for item in tr.get("results", []):
            if isinstance(item, dict):
                items.append(item)
    return items


def evaluate_keywords(expected_keywords: list, tool_results: list) -> dict:
    """expected_keywords 가 결과(어느 항목이든)에 포함되는지 본다(any/all, 단순 포함 기준).

    [의미] keyword_any_hit=하나라도 포함, keyword_all_hit=모두 포함.
    [실패하면 의심할 것] 크롤러가 엉뚱한 결과를 가져왔거나, 키워드가 너무 특정적으로 설정됨.
    """
    keywords = [k for k in (expected_keywords or []) if str(k).strip()]
    if not keywords:
        return {"keyword_any_hit": None, "keyword_all_hit": None,
                "matched_keywords": [], "missing_keywords": []}

    combined_all = " ".join(_item_text(item) for item in _all_items(tool_results))
    matched = [kw for kw in keywords if _norm(kw) in combined_all]
    missing = [kw for kw in keywords if _norm(kw) not in combined_all]
    return {
        "keyword_any_hit": len(matched) >= 1,
        "keyword_all_hit": len(missing) == 0,
        "matched_keywords": matched,
        "missing_keywords": missing,
    }


def evaluate_tool_selection(expected_tools: list, tool_results: list,
                            generation_source: str) -> dict:
    """실행된 tool 이 expected_tools 와 겹치는지 본다.

    [주의] fallback_rule 사용 시 4개 tool 이 모두 실행되므로 any_match 는 항상 True.
    generation_source 가 'llm' 인 경우에만 정밀한 선택 비교가 의미 있다.
    """
    executed = {tr["tool_name"] for tr in tool_results if isinstance(tr, dict)}
    expected = set(expected_tools or [])
    overlap = expected & executed
    return {
        "tool_selection_any_match": len(overlap) > 0,
        "tool_selection_all_match": expected <= executed,
        "matched_tools": sorted(overlap),
        "missing_tools": sorted(expected - executed),
        "extra_tools": sorted(executed - expected),
        "generation_source": generation_source,
    }


def evaluate_source_coverage(tool_results: list) -> dict:
    """각 소스가 실제 데이터를 반환했는지 분류한다.

    - real   : 결과가 있고 fallback 이 아닌 항목이 하나라도 있는 소스
    - fallback: 결과가 있지만 모두 "[fallback]" preview 인 소스
    - error  : 결과가 없거나 모두 error 항목인 소스
    """
    real, fallback, error = [], [], []
    for tr in tool_results:
        if not isinstance(tr, dict):
            continue
        name = tr.get("tool_name", "unknown")
        items = [i for i in tr.get("results", []) if isinstance(i, dict)]
        if tr.get("status") == "error" or not items:
            error.append(name)
            continue
        all_fallback = all(
            str(i.get("preview", "") or "").startswith("[fallback]")
            for i in items if "error" not in i
        )
        all_error = all("error" in i for i in items)
        if all_error:
            error.append(name)
        elif all_fallback:
            fallback.append(name)
        else:
            real.append(name)
    return {
        "real_source_count":     len(real),
        "real_sources":          real,
        "fallback_source_count": len(fallback),
        "fallback_sources":      fallback,
        "error_source_count":    len(error),
        "error_sources":         error,
    }


def evaluate_reranking(expected_keywords: list, tool_results: list) -> dict:
    """reranking 적용 여부와 최상위 결과 관련성을 본다.

    [rerank_applied] 어느 항목이든 rerank_score 필드가 있으면 True.
    [top_result_keyword_hit] rerank_score 기준 최상위 항목에 키워드가 있으면 True.
    [점수 범위] rerank_score 의 min/max 를 기록해 점수 분포를 파악한다.
    """
    items = _all_items(tool_results)
    rerank_applied = any("rerank_score" in item for item in items)

    if not rerank_applied or not items:
        return {
            "rerank_applied": rerank_applied,
            "top_result_keyword_hit": None,
            "rerank_score_min": None,
            "rerank_score_max": None,
        }

    # rerank_score 가 있는 항목 중 최상위 항목을 선택한다.
    scored = [i for i in items if "rerank_score" in i]
    top_item = max(scored, key=lambda x: x.get("rerank_score", 0)) if scored else {}
    top_text = _item_text(top_item)
    keywords = [k for k in (expected_keywords or []) if str(k).strip()]
    top_hit = any(_norm(kw) in top_text for kw in keywords) if keywords else None

    scores = [i["rerank_score"] for i in scored]
    return {
        "rerank_applied": True,
        "top_result_keyword_hit": top_hit,
        "rerank_score_min": min(scores) if scores else None,
        "rerank_score_max": max(scores) if scores else None,
    }


def scan_forbidden(exec_result: dict) -> dict:
    """실행 결과 전체에서 금지 key/패턴이 있는지 검사한다(보안 검증).

    [왜 필요한가] 크롤러 에러 메시지나 디버그 출력에 credential 이 섞여 나오면 안 된다.
    [실패하면 의심할 것] my_security.safe_error_message 가 빠졌거나 새 로그 경로가 생겼을 때.
    """
    found_keys: set[str] = set()
    found_credential = False

    def walk(node):
        nonlocal found_credential
        if isinstance(node, dict):
            for key, val in node.items():
                if str(key).lower() in FORBIDDEN_KEYS:
                    found_keys.add(str(key))
                walk(val)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            if _CREDENTIAL_PATTERN.search(node):
                found_credential = True

    walk(exec_result)
    forbidden = sorted(found_keys)
    if found_credential:
        forbidden.append("credential_pattern")
    return {
        "forbidden_field_found": len(forbidden) > 0,
        "forbidden_fields": forbidden,
    }


# ---------------------------------------------------------------------------
# 판정
# ---------------------------------------------------------------------------

def judge_status(case: dict, m: dict) -> tuple[str, str]:
    """케이스 유형/지표로 PASS/WARNING/FAIL 판정. (status, note) 반환.

    [판정 우선순위]
      ① 보안(금지 필드) → ② 실행 실패(no tool ran) → ③ multi_source 기대 충족 여부
      → ④ keyword_any_hit + real_source 기준

    아래 if 순서를 바꾸면 같은 결과라도 다른 status 가 나오므로 순서 자체가 판정 규칙의 일부다.
    """
    cdt = _norm(case.get("case_design_type"))
    is_multi = cdt == "multi_source"

    # ① 보안 최우선
    if m["forbidden_field_found"]:
        return "FAIL", "금지 필드/패턴이 결과에 노출됨"

    # ② 실행 자체 실패
    if m.get("executed_count", 0) == 0:
        return "FAIL", "실행된 Tool 이 없음(plan 검증 FAIL 또는 rejected)"

    # ③ multi_source: 실제 데이터 소스가 2개 이상이어야 PASS
    if is_multi:
        if m.get("keyword_any_hit") and m.get("real_source_count", 0) >= 2:
            return "PASS", "multi_source: 키워드 hit + 실제 데이터 소스 2개 이상"
        if m.get("keyword_any_hit"):
            return "WARNING", "multi_source: 키워드 hit 이지만 실제 소스가 부족함"
        if m.get("real_source_count", 0) >= 1:
            return "WARNING", "multi_source: 실제 소스는 있지만 키워드 미매칭"
        return "FAIL", "multi_source: 실제 데이터 소스 없음 + 키워드 미매칭"

    # ④ single_source 및 기타
    if m.get("keyword_any_hit") and m.get("real_source_count", 0) >= 1:
        return "PASS", "키워드 hit + 실제 데이터 소스 존재"
    if m.get("keyword_any_hit"):
        return "WARNING", "키워드 hit 이지만 실제 소스 없음(fallback 또는 error)"
    if m.get("real_source_count", 0) >= 1:
        return "WARNING", "실제 소스는 있지만 키워드 미매칭"
    return "FAIL", "키워드 미매칭 + 실제 데이터 소스 없음"


# ---------------------------------------------------------------------------
# 케이스 평가
# ---------------------------------------------------------------------------

def evaluate_case(case: dict) -> dict:
    """한 케이스를 day5_mcp_server call_execute_query 로 실행·평가한다.

    [이 함수가 사실상 '한 건의 테스트']
    - Given: 평가 케이스 1건(user_query + expected_* 라벨)과 실행 가능한 day5_mcp_server.
    - When : query 로 call_execute_query() 를 호출해 크롤러 결과 목록을 받는다.
    - Then : 받은 결과를 evaluate_*() 로 채점하고 judge_status() 로 PASS/WARNING/FAIL 을 매긴다.

    [주의] 지연 import: 서버 모듈을 함수 안에서 불러온다.
    import 단계의 부작용/실패가 스크립트 전체를 막지 않게 하기 위함이다.
    이 호출은 같은 프로세스 안의 직접 함수 호출이다(별도 서버 프로세스/transport 가 아님).
    """
    from my_mcp_server import call_execute_query  # 지연 import

    user_query = str(case.get("user_query", "") or "")
    expected_tools = case.get("expected_tools") or []
    expected_keywords = case.get("expected_keywords") or []

    record: dict = {
        "case_id":             case.get("case_id"),
        "query":               user_query,
        "category":            case.get("category"),
        "case_design_type":    case.get("case_design_type"),
        "difficulty":          case.get("difficulty"),
        "expected_tools":      expected_tools,
        "expected_keywords":   expected_keywords,
        "teaching_point":      case.get("teaching_point"),
        "error_message":       None,
    }

    # [When] call_execute_query 로 tool selection → validation → crawler 실행 전 파이프라인 실행.
    # 네트워크 실패·예외가 발생해도 이 케이스만 ERROR 로 기록하고 다음 케이스로 넘어간다.
    try:
        exec_result = call_execute_query(user_query)
    except Exception as error:
        record.update({
            "status": "ERROR", "status_note": "call_execute_query 실행 실패",
            "executed_count": 0,
            "error_message": f"{type(error).__name__}: {str(error)[:200]}",
            "forbidden_field_found": False, "forbidden_fields": [],
        })
        return record

    tool_results = exec_result.get("results", [])
    generation_source = exec_result.get("generation_source", "unknown")

    # [Then] 받은 결과를 여러 채점 함수로 평가해 한 dict 로 합친다(**로 펼쳐 병합).
    metrics: dict = {
        "executed_count":     exec_result.get("executed_count", 0),
        "validation_status":  exec_result.get("validation_status", "unknown"),
        "generation_source":  generation_source,
        "fallback_used":      bool(exec_result.get("fallback_used")),
        **scan_forbidden(exec_result),
        **evaluate_keywords(expected_keywords, tool_results),
        **evaluate_tool_selection(expected_tools, tool_results, generation_source),
        **evaluate_source_coverage(tool_results),
        **evaluate_reranking(expected_keywords, tool_results),
    }

    status, note = judge_status(case, metrics)
    record.update(metrics)
    record["status"] = status
    record["status_note"] = note
    return record


# ---------------------------------------------------------------------------
# 집계 / 품질 게이트 / 보고서
# ---------------------------------------------------------------------------

def build_quality_gate(summary: dict) -> dict:
    """현재 summary 값으로 quality gate 통과 여부를 계산한다(리포트용)."""
    actual_map = {
        "min_keyword_any_hit_rate":       summary["keyword_any_hit_rate"],
        "min_rerank_applied_rate":        summary["rerank_applied_rate"],
        "min_real_source_rate":           summary["real_source_rate"],
        "max_forbidden_field_case_count": summary["forbidden_field_case_count"],
        "max_fail_count":                 summary["fail_count"],
    }
    checks = {}
    all_passed = True
    for name, threshold in QUALITY_GATE_THRESHOLDS.items():
        actual = actual_map[name]
        passed = (actual >= threshold) if name.startswith("min_") else (actual <= threshold)
        checks[name] = {"threshold": threshold, "actual": actual, "passed": bool(passed)}
        all_passed = all_passed and passed
    return {"passed": bool(all_passed), "checks": checks}


def build_summary(results: list[dict]) -> dict:
    """케이스 결과 집계."""
    total = len(results)
    status_counts = {"PASS": 0, "WARNING": 0, "FAIL": 0, "ERROR": 0}
    for r in results:
        key = r.get("status", "ERROR")
        status_counts[key] = status_counts.get(key, 0) + 1

    kw_den  = [r for r in results if r.get("keyword_any_hit") is not None]
    rr_den  = [r for r in results if r.get("rerank_applied") is not None]
    top_den = [r for r in results if r.get("top_result_keyword_hit") is not None]
    rs_den  = [r for r in results if r.get("real_source_count") is not None]

    summary = {
        "total_cases":   total,
        "pass_count":    status_counts["PASS"],
        "warning_count": status_counts["WARNING"],
        "fail_count":    status_counts["FAIL"],
        "error_count":   status_counts["ERROR"],

        "keyword_any_hit_rate":    _rate(sum(1 for r in kw_den  if r["keyword_any_hit"]),    len(kw_den)),
        "keyword_all_hit_rate":    _rate(sum(1 for r in kw_den  if r.get("keyword_all_hit")), len(kw_den)),

        "rerank_applied_rate":     _rate(sum(1 for r in rr_den  if r["rerank_applied"]),      len(rr_den)),
        "top_result_keyword_hit_rate": _rate(sum(1 for r in top_den if r["top_result_keyword_hit"]), len(top_den)),

        "real_source_rate":        _rate(sum(1 for r in rs_den  if r.get("real_source_count", 0) >= 1), len(rs_den)),
        "avg_real_source_count":   _rate(sum(r.get("real_source_count", 0) for r in rs_den), len(rs_den)),
        "avg_fallback_source_count": _rate(sum(r.get("fallback_source_count", 0) for r in rs_den), len(rs_den)),
        "avg_error_source_count":  _rate(sum(r.get("error_source_count", 0) for r in rs_den), len(rs_den)),

        "tool_selection_any_match_rate": _rate(
            sum(1 for r in results if r.get("tool_selection_any_match")), total
        ),
        "llm_selection_count":   sum(1 for r in results if r.get("generation_source") == "llm"),
        "rule_selection_count":  sum(1 for r in results if "rule" in str(r.get("generation_source", ""))),

        "forbidden_field_case_count": sum(1 for r in results if r.get("forbidden_field_found")),
    }
    summary["quality_gate"] = build_quality_gate(summary)
    return summary


def _md(text) -> str:
    """Markdown 표 셀용 간단 escape."""
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def build_report(results: list[dict], summary: dict) -> str:
    """사람이 읽기 좋은 Markdown 보고서를 만든다."""
    L = []
    L.append("# day5_mcp_server 크롤러 검색 성능 평가 보고서\n")

    L.append("## 1. 평가 개요")
    L.append("- 평가 대상: `src/day5_mcp_server/` — RSS/arXiv 크롤러 + reranking 파이프라인.")
    L.append("- 평가 케이스: `my_evaluation_cases.json` (10건 — 학술/커뮤니티/한국어/업계/복합).")
    L.append("- 실행 경로: `call_execute_query` → tool_selector → tool_executor → crawlers → **my_reranking**.")
    L.append("- quality gate 실패해도 exit code 는 0 (CI 연동 시 --fail-on-gate TODO).\n")

    L.append("## 2. 주요 지표 요약")
    L.append("| 지표 | 값 |")
    L.append("|---|---|")
    for key in ("total_cases", "pass_count", "warning_count", "fail_count", "error_count",
                "keyword_any_hit_rate", "keyword_all_hit_rate",
                "rerank_applied_rate", "top_result_keyword_hit_rate",
                "real_source_rate", "avg_real_source_count",
                "avg_fallback_source_count", "avg_error_source_count",
                "tool_selection_any_match_rate", "llm_selection_count", "rule_selection_count",
                "forbidden_field_case_count"):
        L.append(f"| {key} | {summary.get(key)} |")
    L.append("")

    L.append("## 3. quality gate 결과 (리포트용)")
    L.append(f"- 전체 통과: **{summary['quality_gate']['passed']}** (gate 실패해도 exit code=0)")
    L.append("| gate | threshold | actual | passed |")
    L.append("|---|---|---|---|")
    for name, chk in summary["quality_gate"]["checks"].items():
        L.append(f"| {name} | {chk['threshold']} | {chk['actual']} | {chk['passed']} |")
    L.append("")

    L.append("## 4. 케이스별 결과")
    L.append("| case_id | type | diff | status | kw_any | kw_all | rerank | top_kw | real_src | fallback_src | err_src | sel_match | gen_src |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        L.append(
            "| {cid} | {ct} | {df} | {st} | {ka} | {kl} | {rr} | {tk} | {rs} | {fs} | {es} | {sm} | {gs} |".format(
                cid=_md(r.get("case_id")), ct=_md(r.get("case_design_type")),
                df=_md(r.get("difficulty")), st=r.get("status"),
                ka=r.get("keyword_any_hit"), kl=r.get("keyword_all_hit"),
                rr=r.get("rerank_applied"), tk=r.get("top_result_keyword_hit"),
                rs=r.get("real_source_count"), fs=r.get("fallback_source_count"),
                es=r.get("error_source_count"), sm=r.get("tool_selection_any_match"),
                gs=_md(r.get("generation_source")),
            )
        )
    L.append("")

    L.append("## 5. 키워드 미매칭 케이스 분석")
    for r in results:
        if r.get("missing_keywords"):
            L.append(
                f"- **{r['case_id']}** [{r['status']}] "
                f"matched={_md(r.get('matched_keywords'))} "
                f"missing={_md(r.get('missing_keywords'))}"
            )
    L.append("")

    L.append("## 6. 소스 커버리지 분석")
    for r in results:
        real = r.get("real_source_count", 0)
        fb   = r.get("fallback_source_count", 0)
        err  = r.get("error_source_count", 0)
        L.append(
            f"- **{r['case_id']}** [{r['status']}] "
            f"real={real}{_md(r.get('real_sources'))} "
            f"fallback={fb} error={err}{_md(r.get('error_sources'))}"
        )
    L.append("")

    L.append("## 7. reranking 진단")
    L.append(f"- rerank_applied_rate={summary['rerank_applied_rate']} "
             f"(my_reranking.rerank_crawler_results 가 실행됐는가)")
    L.append(f"- top_result_keyword_hit_rate={summary['top_result_keyword_hit_rate']} "
             f"(rerank 최상위 결과에 키워드가 있는가)")
    for r in results:
        if r.get("rerank_applied"):
            L.append(
                f"  - {r['case_id']}: score_range=[{r.get('rerank_score_min')}, {r.get('rerank_score_max')}] "
                f"top_kw_hit={r.get('top_result_keyword_hit')}"
            )
    L.append("")

    L.append("## 8. 개선 제안")
    L.append("- rerank 보강 시 오버피팅 금지: expected_* 라벨/case_id 전용 분기 금지, 일반 키워드만.")
    L.append("- Google News RSS 차단 시 Reddit/로봇신문 fallback 이 동작 — 실제 데이터가 0건이면 별도 소스 확보 필요.")
    L.append("- LLM Tool 선택이 실패해 fallback_rule 이 4개 tool 모두 실행하는 상황 개선 → llm_client 경로 수정.")
    L.append("- TODO: --fail-on-gate 옵션으로 quality gate 실패 시 exit code 1 처리.\n")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 실행 진입점
# ---------------------------------------------------------------------------

def run(limit=None, case_id=None) -> dict:
    """평가를 실행하고 결과/요약/보고서를 파일로 저장한다(반환: summary).

    [흐름] 케이스 로드 → (옵션) case_id/limit 로 추림
         → 케이스마다 evaluate_case() → 집계/보고서 → 저장.
    """
    cases = load_cases()
    if case_id:
        cases = [c for c in cases if c.get("case_id") == case_id]
    if limit is not None:
        cases = cases[:limit]

    results = []
    for case in cases:
        record = evaluate_case(case)
        results.append(record)
        print(
            f"[{record.get('case_id')}] {record.get('status')} "
            f"(kw={record.get('keyword_any_hit')}, "
            f"rerank={record.get('rerank_applied')}, "
            f"real_src={record.get('real_source_count')}, "
            f"gen={record.get('generation_source')})"
        )

    summary = build_summary(results)
    report  = build_report(results, summary)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8-sig")

    gate = summary["quality_gate"]
    print("\n=== SUMMARY ===")
    print(
        f"total={summary['total_cases']} PASS={summary['pass_count']} "
        f"WARNING={summary['warning_count']} FAIL={summary['fail_count']} ERROR={summary['error_count']}"
    )
    print(
        f"keyword: any_hit={summary['keyword_any_hit_rate']} "
        f"all_hit={summary['keyword_all_hit_rate']}"
    )
    print(
        f"rerank: applied={summary['rerank_applied_rate']} "
        f"top_kw_hit={summary['top_result_keyword_hit_rate']}"
    )
    print(
        f"source: real={summary['real_source_rate']} "
        f"avg_real={summary['avg_real_source_count']} "
        f"avg_fallback={summary['avg_fallback_source_count']} "
        f"avg_error={summary['avg_error_source_count']}"
    )
    print(f"forbidden_cases={summary['forbidden_field_case_count']}")
    print(
        f"quality_gate.passed={gate['passed']} "
        f"(gate 실패해도 exit code=0; CI 용 --fail-on-gate 는 TODO)"
    )
    print(f"outputs -> {OUTPUT_DIR}")
    return summary


def main():
    """CLI 진입점. uv run python src/day5_mcp_server/my_evaluate_crawler.py 로 실행."""
    parser = argparse.ArgumentParser(
        description="day5_mcp_server 크롤러 검색 성능 평가"
    )
    parser.add_argument("--limit",   type=int, default=None, help="평가할 케이스 수 제한")
    parser.add_argument("--case-id", type=str, default=None, help="특정 case_id 만 평가")
    args = parser.parse_args()
    run(limit=args.limit, case_id=args.case_id)


if __name__ == "__main__":
    main()
