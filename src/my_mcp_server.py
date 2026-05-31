"""
로봇 신문 FastMCP Tool Server

[도구 구성]
  search_rfm_news            : RFM 논문 + Google News + 로봇진 뉴스 수집
  search_vla_news            : VLA 논문 + Google News + 로봇진 뉴스 수집
  search_imitation_learning_news : 모방학습 논문 + Google News + 로봇진 뉴스 수집
  search_reddit_robotics     : Reddit r/robotics 최신 게시물 수집

[실행]
  python src/my_mcp_server.py --http  → HTTP 서버 (포트 8766)
  python src/my_mcp_server.py         → stdio 모드 (FastMCP Client 서브프로세스)
"""
from __future__ import annotations

import sys
import urllib.parse

import requests
from bs4 import BeautifulSoup
from fastmcp import FastMCP

SERVER_NAME = "robot_news_mcp_server"
mcp = FastMCP(SERVER_NAME)

REGISTERED_TOOLS = [
    "search_rfm_news",
    "search_vla_news",
    "search_imitation_learning_news",
    "search_reddit_robotics",
]

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


# ─── 내부 크롤러 ───────────────────────────────────────────────────────────────

def _arxiv_search(query: str, limit: int = 5) -> list[dict]:
    """ArXiv Atom API로 최신순 논문을 검색합니다."""
    encoded = urllib.parse.quote(query)
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=all:{encoded}"
        f"&max_results={limit}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.content, "xml")
        results = []
        for entry in soup.find_all("entry"):
            link = entry.id.text.strip()
            if not link.startswith("http"):
                link = f"https://arxiv.org/abs/{link}"
            results.append({
                "type": "논문",
                "source": "ArXiv",
                "title": entry.title.text.strip().replace("\n", " "),
                "link": link,
                "date": entry.published.text.strip()[:10],
            })
        return results
    except Exception:
        return []


def _google_news_search(query: str, limit: int = 5) -> list[dict]:
    """Google News RSS로 최신 기사를 검색합니다."""
    encoded = urllib.parse.quote(query.encode("utf-8"))
    url = f"https://news.google.com/rss/search?q={encoded}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.content, "xml")
        results = []
        for item in soup.find_all("item")[:limit]:
            title = item.title.text.strip()
            source = "구글 뉴스"
            if " - " in title:
                parts = title.split(" - ")
                title = parts[0].strip()
                source = parts[-1].strip()
            try:
                link = item.link.next_sibling.strip()
            except Exception:
                link = item.link.text.strip() if item.link else ""
            results.append({
                "type": "뉴스",
                "source": source,
                "title": title,
                "link": link,
                "date": "최신",
            })
        return results
    except Exception:
        return []


def _robotzine_search(query: str, limit: int = 5) -> list[dict]:
    """robotzine.co.kr 검색으로 최신 로봇 전문지 기사를 수집합니다."""
    encoded = urllib.parse.quote(query)
    url = f"https://robotzine.co.kr/search?keyword={encoded}"
    try:
        resp = requests.get(url, timeout=10, headers=_HEADERS)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        seen: set[str] = set()
        results = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/entry/" not in href:
                continue
            title = a.get_text(strip=True)
            if not title or len(title) < 5:
                continue
            # 절대 URL 정규화
            if href.startswith("/"):
                href = "https://robotzine.co.kr" + href
            if href in seen:
                continue
            seen.add(href)
            results.append({
                "type": "뉴스",
                "source": "로봇진",
                "title": title,
                "link": href,
                "date": "최신",
            })
            if len(results) >= limit:
                break
        return results
    except Exception:
        return []


def _reddit_robotics_feed(limit: int = 10) -> list[dict]:
    """Reddit r/robotics Atom RSS 피드에서 최신 게시물을 수집합니다."""
    url = f"https://www.reddit.com/r/robotics.rss?limit={limit}"
    try:
        resp = requests.get(url, timeout=10, headers=_HEADERS)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.content, "xml")
        results = []
        for entry in soup.find_all("entry")[:limit]:
            title = entry.title.text.strip() if entry.title else "제목 없음"
            link_tag = entry.find("link")
            link = link_tag.get("href", "#") if link_tag else "#"
            date = entry.updated.text[:10] if entry.updated else "최신"
            author_tag = entry.find("author")
            author = ""
            if author_tag and author_tag.find("name"):
                author = author_tag.find("name").text.strip()
            results.append({
                "type": "Reddit",
                "source": f"r/robotics · u/{author}" if author else "r/robotics",
                "title": title,
                "link": link,
                "date": date,
            })
        return results
    except Exception:
        return []


# ─── MCP Tool 등록 ─────────────────────────────────────────────────────────────

@mcp.tool(name="search_rfm_news")
def search_rfm_news(top_k: int = 5) -> dict:
    """
    RFM(Robot Foundation Model) 최신 논문과 뉴스를 수집합니다.
    ArXiv 논문 + Google News + 로봇진(robotzine.co.kr) 기사를 반환합니다.
    """
    arxiv = _arxiv_search("robot foundation model embodied AI large language", top_k)
    gnews = _google_news_search("로봇 기반모델 foundation model RFM 엔보디드", top_k)
    rzine = _robotzine_search("파운데이션 foundation model 로봇", top_k)
    news = gnews + rzine
    return {
        "topic": "RFM (Robot Foundation Model)",
        "arxiv_count": len(arxiv),
        "news_count": len(news),
        "arxiv": arxiv,
        "news": news,
    }


@mcp.tool(name="search_vla_news")
def search_vla_news(top_k: int = 5) -> dict:
    """
    VLA(Vision-Language-Action) 최신 논문과 뉴스를 수집합니다.
    ArXiv 논문 + Google News + 로봇진(robotzine.co.kr) 기사를 반환합니다.
    """
    arxiv = _arxiv_search("vision language action VLA robot policy manipulation", top_k)
    gnews = _google_news_search("VLA 로봇 비전 언어 행동 모델 vision language action", top_k)
    rzine = _robotzine_search("VLA 비전 언어 로봇", top_k)
    news = gnews + rzine
    return {
        "topic": "VLA (Vision-Language-Action)",
        "arxiv_count": len(arxiv),
        "news_count": len(news),
        "arxiv": arxiv,
        "news": news,
    }


@mcp.tool(name="search_imitation_learning_news")
def search_imitation_learning_news(top_k: int = 5) -> dict:
    """
    모방학습(Imitation Learning) 최신 논문과 뉴스를 수집합니다.
    ArXiv 논문 + Google News + 로봇진(robotzine.co.kr) 기사를 반환합니다.
    """
    arxiv = _arxiv_search("imitation learning robot demonstration behavior cloning", top_k)
    gnews = _google_news_search("모방학습 로봇 시연학습 imitation learning robot", top_k)
    rzine = _robotzine_search("모방학습 로봇", top_k)
    news = gnews + rzine
    return {
        "topic": "모방학습 (Imitation Learning)",
        "arxiv_count": len(arxiv),
        "news_count": len(news),
        "arxiv": arxiv,
        "news": news,
    }


@mcp.tool(name="search_reddit_robotics")
def search_reddit_robotics(limit: int = 10) -> dict:
    """
    Reddit r/robotics 서브레딧의 최신 게시물을 수집합니다.
    Atom RSS 피드를 파싱해 제목·링크·날짜·작성자를 반환합니다.
    """
    posts = _reddit_robotics_feed(limit)
    return {
        "subreddit": "r/robotics",
        "post_count": len(posts),
        "posts": posts,
    }


# ─── 진입점 ───────────────────────────────────────────────────────────────────

def _print_startup() -> None:
    print("Robot News MCP Server 시작 (HTTP 모드)", file=sys.stderr)
    print("URL: http://127.0.0.1:8766/mcp", file=sys.stderr)
    print("등록 도구:", file=sys.stderr)
    for name in REGISTERED_TOOLS:
        print(f"  - {name}", file=sys.stderr)
    print("종료: Ctrl+C", file=sys.stderr)


if __name__ == "__main__":
    if "--http" in sys.argv:
        _print_startup()
        mcp.run(transport="http", host="127.0.0.1", port=8766, path="/mcp")
    else:
        mcp.run()  # stdio — FastMCP Client subprocess 모드
