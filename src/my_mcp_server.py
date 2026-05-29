"""
로봇 신문 FastMCP Tool Server

RFM(Robot Foundation Model) / VLA(Vision-Language-Action) / 모방학습 관련
최신 ArXiv 논문 + Google News 기사를 수집하는 MCP Tool 3종을 제공합니다.

[도구 구성]
  search_rfm_news            : RFM 논문·뉴스 수집
  search_vla_news            : VLA 논문·뉴스 수집
  search_imitation_learning_news : 모방학습 논문·뉴스 수집

[실행]
  python src/my_mcp_server.py
  → http://127.0.0.1:8766/mcp 에서 대기
  클라이언트: MY_MCP_URL=http://127.0.0.1:8766/mcp
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
]


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


# ─── MCP Tool 등록 ─────────────────────────────────────────────────────────────

@mcp.tool(name="search_rfm_news")
def search_rfm_news(top_k: int = 5) -> dict:
    """
    RFM(Robot Foundation Model) 최신 논문과 뉴스를 수집합니다.
    ArXiv 논문 top_k건 + Google News 기사 top_k건을 반환합니다.
    """
    arxiv = _arxiv_search("robot foundation model embodied AI large language", top_k)
    news = _google_news_search("로봇 기반모델 foundation model RFM 엔보디드", top_k)
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
    ArXiv 논문 top_k건 + Google News 기사 top_k건을 반환합니다.
    """
    arxiv = _arxiv_search("vision language action VLA robot policy manipulation", top_k)
    news = _google_news_search("VLA 로봇 비전 언어 행동 모델 vision language action", top_k)
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
    ArXiv 논문 top_k건 + Google News 기사 top_k건을 반환합니다.
    """
    arxiv = _arxiv_search("imitation learning robot demonstration behavior cloning", top_k)
    news = _google_news_search("모방학습 로봇 시연학습 imitation learning robot", top_k)
    return {
        "topic": "모방학습 (Imitation Learning)",
        "arxiv_count": len(arxiv),
        "news_count": len(news),
        "arxiv": arxiv,
        "news": news,
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
    # --http 플래그: 독립 HTTP 서버로 실행 (터미널에서 직접 기동)
    # 플래그 없음: stdio 모드로 실행 (FastMCP Client가 서브프로세스로 자동 기동)
    if "--http" in sys.argv:
        _print_startup()
        mcp.run(transport="http", host="127.0.0.1", port=8766, path="/mcp")
    else:
        mcp.run()  # stdio — FastMCP Client subprocess 모드
