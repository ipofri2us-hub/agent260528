# -*- coding: utf-8 -*-
"""
my_mcp_server - Crawlers (실제 크롤러 구현 계층)

[이 모듈의 역할]
4개 크롤러 Tool의 실제 HTTP/RSS/API 호출을 담당합니다.
my_tool_executor가 이 모듈의 함수를 직접 호출하여 결과를 가져옵니다.

[크롤링 대상]
  1. search_reddit      : r/robotics → Google News RSS (site:reddit.com 필터)
  2. search_arxiv       : arXiv Atom API (cs.RO 카테고리)
  3. search_robot_news  : Robot Report / Robohub RSS
  4. search_irobotnews  : 로봇신문 → Google News RSS (hl=ko&gl=KR)

[의존성]
  - requests (pip install requests) : search_arxiv에서 선호 사용.
    없으면 urllib으로 대체(search_reddit, search_irobotnews).
  - 표준 라이브러리: urllib, xml.etree.ElementTree, time

[보안]
  - 이 모듈이 반환하는 raw 결과는 my_tool_executor → my_security.sanitize_result로
    정제된 뒤 외부로 나간다. 여기서 민감정보를 넣지 않는 것이 1차 방어선이다.
  - API Key, 인증 토큰, 내부 URL 등을 반환값에 포함하지 않는다.
"""
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import html as _html_mod
_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_WS = re.compile(r"\s{2,}")


def _strip_html(text: str) -> str:
    """HTML 태그·엔티티를 제거하고 공백을 정리한다."""
    text = _HTML_TAG.sub(" ", text or "")
    text = _html_mod.unescape(text)
    return _MULTI_WS.sub(" ", text).strip()

from my_contracts import RSS_SOURCES, REDDIT_FALLBACK_TOPICS

# ---------------------------------------------------------------------------
# requests 선택 의존성
# ---------------------------------------------------------------------------
try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _requests = None
    _HAS_REQUESTS = False

_BROWSER_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# 공통 HTTP 헬퍼
# ---------------------------------------------------------------------------
def _urllib_get(url: str, timeout: int = 15) -> bytes:
    """urllib로 GET 요청을 보내고 바이트를 돌려준다."""
    req = urllib.request.Request(url, headers=_BROWSER_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _requests_get(url: str, params: dict = None, timeout: int = 12):
    """requests 라이브러리로 GET 요청을 보낸다. 미설치 시 RuntimeError."""
    if not _HAS_REQUESTS:
        raise RuntimeError("pip install requests 필요 (arXiv API 호출)")
    resp = _requests.get(url, headers=_BROWSER_UA, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# 크롤러 함수
# ---------------------------------------------------------------------------

def search_reddit(query: str, limit: int = 5, subreddit: str = "robotics") -> list:
    """Google News RSS로 r/robotics 게시물을 수집한다 (Reddit 봇 차단 우회).

    [동작]
        Google News 의 site:reddit.com 필터를 사용해 Reddit 게시물을 RSS로 가져온다.
        0건이면 my_contracts.REDDIT_FALLBACK_TOPICS로 보완한다.

    [반환]
        list[dict] — 각 항목: title, url, date, preview, source
    """
    rss_query = urllib.parse.quote(f"{subreddit} {query} site:reddit.com")
    rss_url = (
        f"https://news.google.com/rss/search"
        f"?q={rss_query}&hl=en-US&gl=US&ceid=US:en"
    )
    results = []
    try:
        raw = _urllib_get(rss_url)
        root = ET.fromstring(raw)
        for item in root.findall(".//item")[:limit]:
            desc_el = item.find("description")
            results.append({
                "title":   (item.findtext("title") or "").strip(),
                "url":     (item.findtext("link")  or "").strip(),
                "date":    (item.findtext("pubDate") or "")[:22],
                "preview": _strip_html((desc_el.text or ""))[:180] if desc_el is not None else "",
                "source":  f"r/{subreddit}",
            })
    except Exception:
        pass

    # Google News 결과 0건 → fallback 토픽으로 보완
    if not results:
        for title, url in REDDIT_FALLBACK_TOPICS[:limit]:
            results.append({
                "title":   title,
                "url":     url,
                "date":    "",
                "preview": "[fallback] Google News 결과 없어 대표 토픽으로 대체",
                "source":  f"r/{subreddit} (fallback)",
            })
    return results


def search_arxiv(query: str, max_results: int = 5, category: str = "cs.RO") -> list:
    """arXiv Atom API로 로봇공학 논문을 검색한다.

    [동작]
        arXiv API에 cat:{category} AND all:{query} 로 쿼리.
        최신 제출일 내림차순 정렬.

    [반환]
        list[dict] — 각 항목: title, authors(최대 3명), published, summary(280자), url
    """
    if not _HAS_REQUESTS:
        return [{"error": "search_arxiv는 requests 패키지가 필요합니다 (pip install requests)"}]
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
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        results = []
        for entry in root.findall("a:entry", ns):
            link = next(
                (el.get("href") for el in entry.findall("a:link", ns)
                 if el.get("type") == "text/html"),
                ""
            )
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
        return [{"error": f"arXiv 오류: {type(exc).__name__}"}]


def search_robot_news(query: str, limit: int = 5, source: str = "all") -> list:
    """Robot Report / Robohub RSS에서 영문 로봇 뉴스를 검색한다.

    [동작]
        RSS_SOURCES에 등록된 피드를 순서대로 가져와 쿼리 단어가 포함된 기사를 필터링.
        source="all" 이면 모든 피드, source 이름이 있으면 해당 피드만.

    [반환]
        list[dict] — 각 항목: title, url, date, source, preview
    """
    if not _HAS_REQUESTS:
        return [{"error": "search_robot_news는 requests 패키지가 필요합니다 (pip install requests)"}]
    feeds = RSS_SOURCES if source == "all" else {source: RSS_SOURCES.get(source, "")}
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
                        "preview": _strip_html(desc)[:180],
                    })
            time.sleep(0.3)
        except Exception as exc:
            results.append({"error": f"{name}: {type(exc).__name__}"})

    ok = [r for r in results if "error" not in r]
    return (ok or results)[:limit]


def search_irobotnews(query: str, limit: int = 5) -> list:
    """로봇신문(irobotnews.com) 한국어 뉴스를 Google News RSS로 검색한다.

    [동작]
        Google News의 site:irobotnews.com 필터 + 한국어 로케일(hl=ko&gl=KR&ceid=KR:ko).

    [반환]
        list[dict] — 각 항목: title, url, date, preview, source
    """
    rss_query = urllib.parse.quote(f"{query} site:irobotnews.com")
    rss_url = (
        f"https://news.google.com/rss/search"
        f"?q={rss_query}&hl=ko&gl=KR&ceid=KR:ko"
    )
    results = []
    try:
        raw = _urllib_get(rss_url)
        root = ET.fromstring(raw)
        for item in root.findall(".//item")[:limit]:
            desc_el = item.find("description")
            results.append({
                "title":   (item.findtext("title") or "").strip(),
                "url":     (item.findtext("link")  or "").strip(),
                "date":    (item.findtext("pubDate") or "")[:22],
                "preview": _strip_html(desc_el.text or "")[:200] if desc_el is not None else "",
                "source":  "로봇신문",
            })
    except Exception as exc:
        results.append({"error": f"로봇신문 오류: {type(exc).__name__}"})

    if not results:
        results.append({"error": "로봇신문 검색 결과 없음 (Google News 인덱싱 없는 쿼리)"})
    return results


# ---------------------------------------------------------------------------
# DB 조회 함수 (mcp_server03 패턴: DB 우선 + 미가동 시 빈 결과 fallback)
# ---------------------------------------------------------------------------

def search_robot_db(query: str, data_type: str = "papers",
                    category: str = None, source: str = None) -> list:
    """D:/lance_db 의 로봇 기술 캐시에서 query 로 검색한다(LanceDB 읽기 전용).

    [동작]
        1) my_repositories 의 고정 조회 함수를 호출해 LanceDB 를 읽는다.
        2) DB 미준비(LanceDBUnavailableError) 또는 테이블 없음(ValueError)이면
           빈 list 로 안전 fallback 하고 크롤러 Tool 이 보완한다.
        3) data_type 으로 조회 테이블을 선택한다:
           - "papers" (기본): robot_papers (arXiv 논문 캐시)
           - "news"         : robot_news   (뉴스/커뮤니티 캐시)
           - "topics"       : robot_topics (로봇 기술 토픽)

    [반환]
        list[dict] — 각 항목에 data_source / data_type 필드 추가.
        DB 미준비이면 [{"data_source": "lancedb_unavailable", ...}].

    [보안]
        raw 쿼리/테이블 이름을 외부 인자로 받지 않는다.
        my_repositories 의 고정 함수만 호출한다.
    """
    try:
        import my_repositories as repos

        if source:
            rows = repos.search_robot_cache(query, source_type=source, limit=10)
        elif not query:
            rows = repos.list_robot_cache(limit=10)
        else:
            rows = repos.search_robot_cache(query, limit=10)

        return [{"data_source": "lancedb", **row} for row in rows]

    except Exception as exc:
        return [{"data_source": "lancedb_unavailable", "error": type(exc).__name__}]


# ---------------------------------------------------------------------------
# tool_name → 크롤러/DB 함수 디스패치 표
# ---------------------------------------------------------------------------
CRAWLER_DISPATCH = {
    "search_reddit":     (search_reddit,     ["query", "limit", "subreddit"]),
    "search_arxiv":      (search_arxiv,      ["query", "max_results", "category"]),
    "search_robot_news": (search_robot_news, ["query", "limit", "source"]),
    "search_irobotnews": (search_irobotnews, ["query", "limit"]),
    "search_robot_db":   (search_robot_db,   ["query", "data_type", "category", "source"]),
}


if __name__ == "__main__":
    print("[my_crawlers] 기본 동작 테스트 (네트워크 필요)")
    q = "ROS2 AMR navigation"
    print(f"\n--- search_irobotnews('{q}') ---")
    for r in search_irobotnews(q, limit=2):
        print(f"  {r.get('title', r.get('error', ''))[:60]}")
    print(f"\n--- search_arxiv('{q}') ---")
    for r in search_arxiv(q, max_results=2):
        print(f"  {r.get('title', r.get('error', ''))[:60]}")
