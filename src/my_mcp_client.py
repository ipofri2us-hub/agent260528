"""
로봇 신문 MCP Client

my_mcp_server.py 의 3개 도구를 호출하는 클라이언트입니다.
FastMCP Client(asyncio 기반) + 동기 래퍼 구조입니다.

[핵심 개념]
  - call_tool(tool_name, tool_input) 이 이 파일의 단일 진입점입니다.
  - 상위 Agent(my_mcp_graph.py)는 tool_name 문자열과 입력 dict 만 알면 됩니다.
    (서버 연결, 응답 파싱 등 세부사항은 이 Client가 은닉)

[연결 우선순위]
  1) 환경변수 MY_MCP_URL 이 있으면 그 URL을 사용 (이미 실행 중인 서버)
  2) 없으면 my_mcp_server.py 스크립트 경로를 전달 → FastMCP가 서브프로세스로 자동 기동

[사용 예]
  # 서버가 실행 중일 때
  $env:MY_MCP_URL = "http://127.0.0.1:8766/mcp"
  python src/my_mcp_client.py

  # 서버 없이 (자동 기동)
  python src/my_mcp_client.py
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
from pathlib import Path

import yaml  # type: ignore[import-untyped]

# 이 Client가 허용하는 Tool 이름 목록 (화이트리스트)
SUPPORTED_TOOLS = [
    "search_rfm_news",
    "search_vla_news",
    "search_imitation_learning_news",
    "search_reddit_robotics",
]

TOOL_DESCRIPTIONS = {
    "search_rfm_news": "RFM(Robot Foundation Model) 최신 논문·뉴스 수집 (ArXiv + Google News + 로봇진)",
    "search_vla_news": "VLA(Vision-Language-Action) 최신 논문·뉴스 수집 (ArXiv + Google News + 로봇진)",
    "search_imitation_learning_news": "모방학습(Imitation Learning) 최신 논문·뉴스 수집 (ArXiv + Google News + 로봇진)",
    "search_reddit_robotics": "Reddit r/robotics 최신 게시물 수집",
}

def _load_url_from_config() -> str:
    """config.yaml 의 mcp.my_mcp_url 값을 읽어 반환합니다. 없으면 빈 문자열."""
    try:
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return str(cfg.get("mcp", {}).get("my_mcp_url", "")).strip()
    except Exception:
        pass
    return ""


def get_target() -> str:
    """
    FastMCP Client 연결 대상을 결정합니다.

    우선순위:
      1) 환경변수 MY_MCP_URL
      2) config.yaml  mcp.my_mcp_url
      3) my_mcp_server.py 스크립트 경로 (FastMCP 서브프로세스 자동 기동)
    """
    url = os.getenv("MY_MCP_URL", "").strip()
    if url:
        return url
    url = _load_url_from_config()
    if url:
        return url
    return str(Path(__file__).resolve().parent / "my_mcp_server.py")


def _extract_result(raw) -> dict:
    """
    FastMCP 응답 객체에서 실제 dict 결과를 추출합니다.

    FastMCP 버전/호출 경로마다 응답 형태가 다를 수 있어
    structured_content → content[].text (JSON) → data 순서로 시도합니다.
    """
    # 이미 dict 이면 그대로 반환
    if isinstance(raw, dict):
        return raw

    # list 형태: [TextContent(text='{"topic":...}'), ...]
    if isinstance(raw, list):
        for item in raw:
            text = getattr(item, "text", None)
            if text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
        return {"items": raw}

    # CallToolResult.structured_content (FastMCP ≥ 2.x dict 반환 시 자동 구조화)
    structured = getattr(raw, "structured_content", None)
    if structured is not None:
        return structured if isinstance(structured, dict) else {"data": structured}

    # CallToolResult.content → TextContent 리스트
    content = getattr(raw, "content", None)
    if content:
        items = content if isinstance(content, list) else [content]
        for item in items:
            text = getattr(item, "text", None)
            if text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

    # data 속성
    data = getattr(raw, "data", None)
    if data is not None:
        return data if isinstance(data, dict) else {"data": data}

    return {"raw": str(raw)}


async def _call_tool_async(tool_name: str, tool_input: dict) -> object:
    """FastMCP Client로 비동기 Tool 호출을 수행합니다."""
    from fastmcp import Client

    target = get_target()
    async with Client(target) as client:
        return await client.call_tool(tool_name, tool_input)


def call_tool(tool_name: str, tool_input: dict | None = None) -> dict:
    """
    이 Client의 단일 진입점입니다.
    Tool 이름과 입력 dict를 받아 MCP 서버를 호출하고 결과 dict를 반환합니다.

    이미 실행 중인 이벤트 루프(LangGraph 내부 등)에서도 안전하게 동작합니다.
    """
    if tool_input is None:
        tool_input = {}
    if tool_name not in SUPPORTED_TOOLS:
        raise ValueError(f"미지원 도구: {tool_name}")

    # 실행 중인 이벤트 루프가 있으면 별도 스레드에서 asyncio.run 실행
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop and running_loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _call_tool_async(tool_name, tool_input))
            raw = future.result()
    else:
        raw = asyncio.run(_call_tool_async(tool_name, tool_input))

    return _extract_result(raw)


def list_tools() -> list[dict]:
    """현재 Client가 지원하는 도구 목록을 반환합니다."""
    return [
        {"name": name, "description": TOOL_DESCRIPTIONS[name]}
        for name in SUPPORTED_TOOLS
    ]


# ─── 단독 실행 테스트 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Robot News MCP Client 테스트 ===")
    print(f"연결 대상: {get_target()}")
    print()

    for tool_name in SUPPORTED_TOOLS:
        print(f"[{tool_name}] 호출 중...")
        result = call_tool(tool_name, {"top_k": 3})
        topic = result.get("topic", tool_name)
        arxiv = result.get("arxiv", [])
        news = result.get("news", [])
        print(f"  토픽 : {topic}")
        print(f"  논문 : {len(arxiv)}건  뉴스 : {len(news)}건")
        for item in (arxiv + news)[:2]:
            title = item.get("title", "")
            print(f"    › {title[:70]}")
        print()
