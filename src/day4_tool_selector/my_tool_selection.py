# -*- coding: utf-8 -*-
"""
로봇 개발(HW, RFM, ROS, AMR) 뉴스·정보 수집 파이프라인
Tool Selection → Normalize → Validate → Execute

제약 조건(Tool Contract, 주제 키워드, RSS 주소 등)은
my_tool_contracts.json에서 로드합니다.
Python 코드를 수정하지 않고 JSON만 편집해 tool·주제를 추가할 수 있습니다.

검색 소스:
  1. Reddit      : r/robotics (Google News RSS 경유 — 봇 차단 우회)
  2. arXiv       : cs.RO 카테고리 논문 (arXiv Atom API)
  3. 영문 뉴스   : The Robot Report / Robohub (RSS)
  4. 한국어 뉴스 : 로봇신문 irobotnews.com (Google News RSS 한국어 로케일)

실행:
  python my_tool_selection.py
  python my_tool_selection.py --query "ROS2 AMR navigation"
  python my_tool_selection.py --query "그리퍼 센서 HW"
"""
import sys
import json
import time
import argparse
import xml.etree.ElementTree as ET
import urllib.parse
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════
# JSON 제약 조건 로드
# ═══════════════════════════════════════════════════════════
_CONTRACTS_PATH = Path(__file__).parent / "my_tool_contracts.json"


def load_contracts(path: Path = _CONTRACTS_PATH) -> dict:
    """my_tool_contracts.json에서 tool 제약 조건과 설정을 로드합니다.

    반환 dict 구조:
      tools              : tool 이름 → contract 정의
      topic_keywords     : 주제 → 감지 키워드 목록
      topic_enrichment   : 주제 → 검색어 확장 문자열
      rss_sources        : 소스 이름 → RSS URL
      reddit_fallback_topics: [[제목, URL], ...] Google News 0건 시 대체 토픽
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_CFG              = load_contracts()
TOOL_CONTRACTS    = _CFG["tools"]
ALLOWED_TOOLS     = set(TOOL_CONTRACTS.keys())
TOPIC_KEYWORDS    = _CFG["topic_keywords"]
TOPIC_ENRICHMENT  = _CFG["topic_enrichment"]
_RSS_SOURCES      = _CFG["rss_sources"]
_REDDIT_FALLBACKS = _CFG["reddit_fallback_topics"]   # [[title, url], ...]

# ───────────────────────────────────────────────────────────
# Rule 기반 Tool Plan 생성
# ───────────────────────────────────────────────────────────
def detect_topics(query: str) -> list:
    lower = query.lower()
    return [t for t, kws in TOPIC_KEYWORDS.items() if any(k in lower for k in kws)]


def enrich_query(query: str, topics: list) -> str:
    if not topics:
        return query
    return f"{query} {TOPIC_ENRICHMENT.get(topics[0], '')}".strip()


def select_tools(query: str) -> dict:
    """
    키워드 기반으로 4단계 Tool Plan을 생성합니다.

    Step 1 search_reddit      : r/robotics 커뮤니티 동향 (Google News 경유)
    Step 2 search_arxiv       : arXiv cs.RO 최신 논문
    Step 3 search_robot_news  : 영문 로봇 뉴스 RSS (Robot Report / Robohub)
    Step 4 search_irobotnews  : 한국어 로봇신문 irobotnews.com

    tool 목록은 TOOL_CONTRACTS(JSON)에 등록된 것만 허용됩니다.
    JSON에 새 tool을 추가하면 execute_tool에 dispatch 함수만 추가하면 됩니다.
    """
    topics   = detect_topics(query)
    enriched = enrich_query(query, topics)

    plan = [
        {
            "step": 1,
            "tool_name": "search_reddit",
            "arguments": {"query": query},
            "condition": "always",
            "reason": f"r/robotics 최신 토론 및 커뮤니티 동향 (주제: {topics or ['general']})",
        },
        {
            "step": 2,
            "tool_name": "search_arxiv",
            "arguments": {"query": query},
            "condition": "always",
            "reason": "arXiv cs.RO 최신 논문으로 기술 동향 파악",
        },
        {
            "step": 3,
            "tool_name": "search_robot_news",
            "arguments": {"query": query},
            "condition": "always",
            "reason": "영문 로봇 전문 뉴스 RSS (Robot Report / Robohub) 수집",
        },
        {
            "step": 4,
            "tool_name": "search_irobotnews",
            "arguments": {"query": query},
            "condition": "always",
            "reason": "로봇신문(irobotnews.com) 한국어 뉴스 수집",
        },
    ]

    return {"plan": plan, "detected_topics": topics, "enriched_query": enriched}


# ═══════════════════════════════════════════════════════════
# 공통 HTTP 유틸
# ═══════════════════════════════════════════════════════════
_BROWSER_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _requests_get(url: str, params: dict = None, timeout: int = 12):
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("pip install requests 필요")
    resp = requests.get(url, headers=_BROWSER_UA, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


def _urllib_get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers=_BROWSER_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ═══════════════════════════════════════════════════════════
# Tool 실행 함수
# ═══════════════════════════════════════════════════════════
def search_reddit(query: str, limit: int = 5, subreddit: str = "robotics") -> list:
    """Google News RSS로 r/robotics 게시물을 수집합니다 (Reddit 봇 차단 우회).

    Reddit API는 2023년 이후 인증 필요. Google News의 site:reddit.com 필터로 우회.
    0건이면 my_tool_contracts.json의 reddit_fallback_topics로 보완합니다.
    """
    rss_query = urllib.parse.quote(f"{subreddit} {query} site:reddit.com")
    rss_url   = (f"https://news.google.com/rss/search"
                 f"?q={rss_query}&hl=en-US&gl=US&ceid=US:en")
    results = []
    try:
        raw  = _urllib_get(rss_url)
        root = ET.fromstring(raw)
        for item in root.findall(".//item")[:limit]:
            desc_el = item.find("description")
            results.append({
                "title":   (item.findtext("title") or "").strip(),
                "url":     (item.findtext("link")  or "").strip(),
                "date":    (item.findtext("pubDate") or "")[:22],
                "preview": (desc_el.text or "").strip()[:180] if desc_el is not None else "",
                "source":  f"r/{subreddit}",
            })
    except Exception:
        pass

    if not results:
        for title, url in _REDDIT_FALLBACKS[:limit]:
            results.append({
                "title":   title,
                "url":     url,
                "date":    "",
                "preview": "[fallback] Google News 결과 없어 대표 토픽으로 대체",
                "source":  f"r/{subreddit} (fallback)",
            })
    return results


def search_arxiv(query: str, max_results: int = 5, category: str = "cs.RO") -> list:
    """arXiv Atom API로 로봇공학 논문을 검색합니다."""
    try:
        resp = _requests_get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"cat:{category} AND all:{query}",
                "max_results":  max_results,
                "sortBy":       "submittedDate",
                "sortOrder":    "descending",
            },
            timeout=15,
        )
        ns  = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        results = []
        for entry in root.findall("a:entry", ns):
            link = next(
                (el.get("href") for el in entry.findall("a:link", ns)
                 if el.get("type") == "text/html"), "")
            results.append({
                "title":     (entry.findtext("a:title", "", ns) or "").replace("\n", " ").strip(),
                "authors":   [a.findtext("a:name", "", ns)
                              for a in entry.findall("a:author", ns)][:3],
                "published": (entry.findtext("a:published", "", ns) or "")[:10],
                "summary":   (entry.findtext("a:summary", "", ns) or "").strip()[:280],
                "url":       link,
            })
        return results
    except Exception as exc:
        return [{"error": f"arXiv 오류: {exc}"}]


def search_robot_news(query: str, limit: int = 5, source: str = "all") -> list:
    """Robot Report / Robohub RSS에서 영문 로봇 뉴스를 검색합니다.

    RSS 소스 목록은 my_tool_contracts.json의 rss_sources에서 로드됩니다.
    """
    feeds = _RSS_SOURCES if source == "all" else {source: _RSS_SOURCES.get(source, "")}
    words = [w for w in query.lower().split() if len(w) > 3]
    results = []

    for name, url in feeds.items():
        if not url:
            continue
        try:
            resp    = _requests_get(url, timeout=10)
            root    = ET.fromstring(resp.content)
            channel = root.find("channel")
            if channel is None:
                continue
            for item in channel.findall("item"):
                title = item.findtext("title", "")
                desc  = item.findtext("description", "")
                if not words or any(w in (title + desc).lower() for w in words):
                    results.append({
                        "title":   title,
                        "url":     item.findtext("link", ""),
                        "date":    (item.findtext("pubDate") or "")[:22],
                        "source":  name,
                        "preview": desc[:180].strip(),
                    })
            time.sleep(0.3)
        except Exception as exc:
            results.append({"error": f"{name}: {exc}"})

    ok = [r for r in results if "error" not in r]
    return (ok or results)[:limit]


def search_irobotnews(query: str, limit: int = 5) -> list:
    """로봇신문(irobotnews.com) 한국어 뉴스를 Google News RSS로 검색합니다.

    한국어 로케일(hl=ko&gl=KR&ceid=KR:ko)을 사용해 한글 제목·요약을 반환합니다.
    my_rag_app3.py의 site:irobotnews.com 접근 방식을 그대로 적용합니다.
    """
    rss_query = urllib.parse.quote(f"{query} site:irobotnews.com")
    rss_url   = (f"https://news.google.com/rss/search"
                 f"?q={rss_query}&hl=ko&gl=KR&ceid=KR:ko")
    results = []
    try:
        raw  = _urllib_get(rss_url)
        root = ET.fromstring(raw)
        for item in root.findall(".//item")[:limit]:
            desc_el = item.find("description")
            results.append({
                "title":   (item.findtext("title") or "").strip(),
                "url":     (item.findtext("link")  or "").strip(),
                "date":    (item.findtext("pubDate") or "")[:22],
                "preview": (desc_el.text or "").strip()[:200] if desc_el is not None else "",
                "source":  "로봇신문",
            })
    except Exception as exc:
        results.append({"error": f"로봇신문 오류: {exc}"})

    if not results:
        results.append({"error": "로봇신문 검색 결과 없음 (Google News 인덱싱 없는 쿼리)"})
    return results


# ═══════════════════════════════════════════════════════════
# Tool 실행 디스패처
# ═══════════════════════════════════════════════════════════
def execute_tool(tool_name: str, arguments: dict) -> list:
    """tool_name에 따라 해당 함수를 호출합니다.

    기본값은 TOOL_CONTRACTS(JSON)의 default_values에서 병합합니다.
    새 tool을 JSON에 추가한 경우 이 함수에도 분기를 추가하세요.
    """
    defaults = TOOL_CONTRACTS.get(tool_name, {}).get("default_values", {})
    args     = {**defaults, **arguments}

    if tool_name == "search_reddit":
        return search_reddit(args["query"], args.get("limit", 5), args.get("subreddit", "robotics"))
    if tool_name == "search_arxiv":
        return search_arxiv(args["query"], args.get("max_results", 5), args.get("category", "cs.RO"))
    if tool_name == "search_robot_news":
        return search_robot_news(args["query"], args.get("limit", 5), args.get("source", "all"))
    if tool_name == "search_irobotnews":
        return search_irobotnews(args["query"], args.get("limit", 5))
    return [{"error": f"알 수 없는 tool: {tool_name}"}]


# ═══════════════════════════════════════════════════════════
# 결과 출력 포매터
# ═══════════════════════════════════════════════════════════
def _line(char="─", n=64):
    print(char * n)


def print_results(plan_result: dict, executed: list):
    print()
    print("═" * 64)
    print("🤖  로봇 개발 정보 검색 결과")
    topics = plan_result.get("detected_topics") or []
    print(f"    원본 쿼리 : {plan_result.get('enriched_query', '')}")
    print(f"    감지 주제 : {', '.join(topics) if topics else 'general (주제 미감지)'}")
    print("═" * 64)

    for step_info in executed:
        item    = step_info["plan_item"]
        results = step_info["results"]
        tool    = item["tool_name"]

        print(f"\n▶ [Step {item['step']}]  {tool}")
        print(f"   {item['reason']}")
        _line()

        if not results:
            print("   결과 없음")
            continue

        for idx, r in enumerate(results, 1):
            if "error" in r:
                print(f"  {idx}. ⚠️  {r['error']}")
                continue

            print(f"\n  {idx}. {r.get('title', '(제목 없음)')}")

            if tool == "search_reddit":
                src = r.get("source", "")
                print(f"     출처: {src}  📅 {r.get('date', '')}")
                if r.get("preview"):
                    print(f"     {r['preview'][:100]}")

            elif tool == "search_arxiv":
                authors = ", ".join(r.get("authors", []))
                print(f"     저자: {authors or '미상'}  게재: {r.get('published', '')}")
                if r.get("summary"):
                    print(f"     {r['summary'][:160]}...")

            elif tool in ("search_robot_news", "search_irobotnews"):
                label = "로봇신문" if tool == "search_irobotnews" else r.get("source", "")
                print(f"     출처: {label}  📅 {r.get('date', '')}")
                if r.get("preview"):
                    print(f"     {r['preview'][:100]}")

            url = r.get("url", "")
            if url:
                print(f"     🔗 {url}")

    print()
    _line("═")


# ═══════════════════════════════════════════════════════════
# __main__  :  select → normalize → validate → execute → print
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="로봇 개발(HW/RFM/ROS/AMR) 뉴스·논문·커뮤니티 검색"
    )
    parser.add_argument(
        "--query", "-q",
        default="ROS2 AMR autonomous mobile robot navigation",
        help="검색 쿼리",
    )
    cli = parser.parse_args()

    if not REQUESTS_AVAILABLE:
        print("[오류] pip install requests 필요")
        sys.exit(1)

    from my_tool_normalizer import normalize_plan
    from my_tool_validation import validate_plan

    print(f"\n쿼리: {cli.query!r}")
    print("─" * 64)

    # ① 선택
    plan_result = select_tools(cli.query)
    print(f"[1/4] 선택 완료 | 주제: {plan_result['detected_topics'] or ['general']}")
    print(f"       확장 쿼리: {plan_result['enriched_query']}")

    # ② 정규화
    plan_result = normalize_plan(plan_result)
    print(f"[2/4] 정규화 완료 | plan 항목: {len(plan_result['plan'])}개")

    # ③ 검증
    val = validate_plan(plan_result["plan"])
    print(f"[3/4] 검증 완료 | status={val['status']} | issues={len(val['issues'])}")
    for issue in val["issues"]:
        mark = "❌" if issue["severity"] == "FAIL" else "⚠️ "
        print(f"       {mark} [{issue['type']}] {issue['message']}")

    if val["status"] == "FAIL":
        print("\n[중단] 검증 실패")
        sys.exit(1)

    # ④ 실행
    print("[4/4] 각 tool 실행 중...")
    executed = []
    for item in plan_result["plan"]:
        print(f"       → {item['tool_name']} ...", end=" ", flush=True)
        results = execute_tool(item["tool_name"], item["arguments"])
        ok = sum(1 for r in results if "error" not in r)
        print(f"{ok}건")
        executed.append({"plan_item": item, "results": results})
        time.sleep(1.0)

    # ⑤ 출력
    print_results(plan_result, executed)
