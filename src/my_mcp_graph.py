"""
로봇 신문 Multi-Agent LangGraph

RFM / VLA / 모방학습 논문·뉴스를 수집해 마크다운 보고서를 생성합니다.

[에이전트 역할 분담]
  CoordinatorAgent      : 사용자 쿼리 분석, top_k 결정
  RFMFetcherAgent       : search_rfm_news 도구 호출
  VLAFetcherAgent       : search_vla_news 도구 호출
  ImitationFetcherAgent : search_imitation_learning_news 도구 호출
  ReportAgent           : 수집 결과 종합 → 마크다운 보고서

[State 흐름]
  user_query
    → [Coordinator] top_k 결정
    → [RFMFetcher]  rfm_results 채움
    → [VLAFetcher]  vla_results 채움
    → [ImitationFetcher] imitation_results 채움
    → [Report]      final_report 완성

[LangGraph 구조]
  coordinator → rfm_fetch → vla_fetch → imitation_fetch → report → END

[실행]
  # 서버 기동 후 (권장)
  터미널 1: python src/my_mcp_server.py
  터미널 2: $env:MY_MCP_URL="http://127.0.0.1:8766/mcp"
            python src/my_mcp_graph.py

  # 서버 자동 기동 모드
  python src/my_mcp_graph.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

# src/ 디렉터리를 sys.path 에 추가해 my_mcp_client 를 import 할 수 있게 합니다.
SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from my_mcp_client import call_tool as mcp_call_tool  # noqa: E402

DEFAULT_TOP_K = 5
SAMPLE_QUERY = "로봇 AI 최신 동향: RFM, VLA, 모방학습 논문과 뉴스를 알려주세요."


# ─── State 정의 ────────────────────────────────────────────────────────────────

class NewsState(TypedDict):
    """에이전트 간 공유 상태 (LangGraph Node 간 전달되는 딕셔너리)"""
    user_query: str
    top_k: int                          # 토픽당 수집 건수
    rfm_results: Dict[str, Any]         # RFM 수집 결과
    vla_results: Dict[str, Any]         # VLA 수집 결과
    imitation_results: Dict[str, Any]   # 모방학습 수집 결과
    agent_steps: List[str]              # 실행 흐름 기록
    final_report: str                   # 최종 마크다운 보고서


def _add_step(state: dict, agent: str, msg: str) -> None:
    state["agent_steps"].append(f"[{agent}] {msg}")


def _make_initial_state(user_query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    return {
        "user_query": user_query,
        "top_k": top_k,
        "rfm_results": {},
        "vla_results": {},
        "imitation_results": {},
        "agent_steps": [],
        "final_report": "",
    }


# ─── Agent 클래스 ──────────────────────────────────────────────────────────────

class CoordinatorAgent:
    """
    [조율 에이전트] 사용자 쿼리를 분석하고 검색 범위를 결정합니다.

    State 변경: top_k 조정, agent_steps 기록
    LangGraph 관점: 그래프의 첫 번째 입력 처리 노드
    """
    name = "CoordinatorAgent"

    def run(self, state: dict) -> dict:
        query = state.get("user_query", "")

        # 쿼리에서 숫자가 있으면 top_k 힌트로 해석 (최대 10)
        top_k = state.get("top_k", DEFAULT_TOP_K)
        for token in query.replace(",", " ").split():
            if token.isdigit():
                top_k = min(int(token), 10)
                break
        state["top_k"] = top_k

        _add_step(state, self.name,
                  f"쿼리 분석 완료. top_k={top_k} / 검색 토픽 = [RFM, VLA, 모방학습]")
        return state


class RFMFetcherAgent:
    """
    [RFM 수집 에이전트] search_rfm_news 도구를 호출합니다.

    State 변경: rfm_results 채움
    생성자에서 call_tool 함수를 주입받습니다 (의존성 주입).
    """
    name = "RFMFetcherAgent"

    def __init__(self, call_tool_func) -> None:
        self.call_tool = call_tool_func

    def run(self, state: dict) -> dict:
        top_k = state.get("top_k", DEFAULT_TOP_K)
        try:
            result = self.call_tool("search_rfm_news", {"top_k": top_k})
            state["rfm_results"] = result
            a = result.get("arxiv_count", len(result.get("arxiv", [])))
            n = result.get("news_count", len(result.get("news", [])))
            _add_step(state, self.name, f"RFM 수집 완료 — 논문 {a}건 / 뉴스 {n}건")
        except Exception as exc:
            state["rfm_results"] = {"arxiv": [], "news": [], "error": str(exc)}
            _add_step(state, self.name, f"[오류] RFM 수집 실패: {exc}")
        return state


class VLAFetcherAgent:
    """
    [VLA 수집 에이전트] search_vla_news 도구를 호출합니다.

    State 변경: vla_results 채움
    """
    name = "VLAFetcherAgent"

    def __init__(self, call_tool_func) -> None:
        self.call_tool = call_tool_func

    def run(self, state: dict) -> dict:
        top_k = state.get("top_k", DEFAULT_TOP_K)
        try:
            result = self.call_tool("search_vla_news", {"top_k": top_k})
            state["vla_results"] = result
            a = result.get("arxiv_count", len(result.get("arxiv", [])))
            n = result.get("news_count", len(result.get("news", [])))
            _add_step(state, self.name, f"VLA 수집 완료 — 논문 {a}건 / 뉴스 {n}건")
        except Exception as exc:
            state["vla_results"] = {"arxiv": [], "news": [], "error": str(exc)}
            _add_step(state, self.name, f"[오류] VLA 수집 실패: {exc}")
        return state


class ImitationFetcherAgent:
    """
    [모방학습 수집 에이전트] search_imitation_learning_news 도구를 호출합니다.

    State 변경: imitation_results 채움
    """
    name = "ImitationFetcherAgent"

    def __init__(self, call_tool_func) -> None:
        self.call_tool = call_tool_func

    def run(self, state: dict) -> dict:
        top_k = state.get("top_k", DEFAULT_TOP_K)
        try:
            result = self.call_tool("search_imitation_learning_news", {"top_k": top_k})
            state["imitation_results"] = result
            a = result.get("arxiv_count", len(result.get("arxiv", [])))
            n = result.get("news_count", len(result.get("news", [])))
            _add_step(state, self.name, f"모방학습 수집 완료 — 논문 {a}건 / 뉴스 {n}건")
        except Exception as exc:
            state["imitation_results"] = {"arxiv": [], "news": [], "error": str(exc)}
            _add_step(state, self.name, f"[오류] 모방학습 수집 실패: {exc}")
        return state


class ReportAgent:
    """
    [보고서 에이전트] 세 토픽의 수집 결과를 종합해 마크다운 보고서를 작성합니다.

    State 변경: final_report 채움
    LangGraph 관점: 그래프의 마지막 출력 생성 노드
    """
    name = "ReportAgent"

    def _format_topic_section(self, heading: str, data: dict) -> str:
        lines = [f"## {heading}"]
        arxiv_items = data.get("arxiv", [])
        news_items = data.get("news", [])
        error = data.get("error")

        if error:
            lines.append(f"> 수집 오류: {error}")
            return "\n".join(lines)

        if not arxiv_items and not news_items:
            lines.append("> 수집된 데이터가 없습니다.")
            return "\n".join(lines)

        if arxiv_items:
            lines.append("")
            lines.append("### 학술 논문 (ArXiv)")
            for i, item in enumerate(arxiv_items, 1):
                title = item.get("title", "제목 없음")
                link = item.get("link", "#")
                date = item.get("date", "")
                lines.append(f"{i}. [{title}]({link})  `{date}`")

        if news_items:
            lines.append("")
            lines.append("### 뉴스 기사")
            for i, item in enumerate(news_items, 1):
                title = item.get("title", "제목 없음")
                link = item.get("link", "#")
                source = item.get("source", "")
                lines.append(f"{i}. [{title}]({link})  — {source}")

        return "\n".join(lines)

    def run(self, state: dict) -> dict:
        rfm_sec = self._format_topic_section(
            "1. RFM (Robot Foundation Model)", state.get("rfm_results", {})
        )
        vla_sec = self._format_topic_section(
            "2. VLA (Vision-Language-Action)", state.get("vla_results", {})
        )
        imitation_sec = self._format_topic_section(
            "3. 모방학습 (Imitation Learning)", state.get("imitation_results", {})
        )

        steps_md = "\n".join(f"- {s}" for s in state.get("agent_steps", []))
        top_k = state.get("top_k", DEFAULT_TOP_K)

        state["final_report"] = f"""# 로봇 AI 기술 뉴스 수집 보고서

**검색 쿼리:** {state.get("user_query", "")}
**수집 설정:** 토픽당 논문·뉴스 최대 {top_k}건

---

{rfm_sec}

---

{vla_sec}

---

{imitation_sec}

---

## 에이전트 실행 기록

{steps_md}
"""
        _add_step(state, self.name, "최종 보고서 생성 완료")
        return state


# ─── LangGraph 워크플로우 ──────────────────────────────────────────────────────

def build_graph(call_tool_func) -> object:
    """
    Multi-Agent LangGraph를 조립합니다.

    [노드]  coordinator → rfm_fetch → vla_fetch → imitation_fetch → report → END
    [엣지]  순차 직렬 연결 (각 에이전트가 State를 채운 뒤 다음 에이전트로 전달)
    """
    coordinator = CoordinatorAgent()
    rfm_agent = RFMFetcherAgent(call_tool_func)
    vla_agent = VLAFetcherAgent(call_tool_func)
    imitation_agent = ImitationFetcherAgent(call_tool_func)
    report_agent = ReportAgent()

    workflow = StateGraph(dict)

    workflow.add_node("coordinator",       lambda s: coordinator.run(s))
    workflow.add_node("rfm_fetch",         lambda s: rfm_agent.run(s))
    workflow.add_node("vla_fetch",         lambda s: vla_agent.run(s))
    workflow.add_node("imitation_fetch",   lambda s: imitation_agent.run(s))
    workflow.add_node("report",            lambda s: report_agent.run(s))

    workflow.set_entry_point("coordinator")
    workflow.add_edge("coordinator",     "rfm_fetch")
    workflow.add_edge("rfm_fetch",       "vla_fetch")
    workflow.add_edge("vla_fetch",       "imitation_fetch")
    workflow.add_edge("imitation_fetch", "report")
    workflow.add_edge("report",          END)

    return workflow.compile()


def run_news_agent_flow(
    user_query: str,
    top_k: int = DEFAULT_TOP_K,
    call_tool_func=None,
) -> dict:
    """Multi-Agent LangGraph 워크플로우를 실행하고 최종 State를 반환합니다."""
    if call_tool_func is None:
        call_tool_func = mcp_call_tool

    state = _make_initial_state(user_query, top_k)
    app = build_graph(call_tool_func)
    return app.invoke(state)


def save_report(state: dict) -> Path:
    """최종 보고서를 outputs/ 에 저장합니다."""
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "robot_news_report.md"
    output_path.write_text(state["final_report"], encoding="utf-8")
    return output_path


# ─── 단독 실행 진입점 ──────────────────────────────────────────────────────────

def main() -> None:
    # Windows cp949 콘솔에서 한글·유니코드 특수문자를 안전하게 출력합니다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("[Robot News Multi-Agent Graph] 시작")
    print(f"쿼리: {SAMPLE_QUERY}")
    print()

    final_state = run_news_agent_flow(SAMPLE_QUERY, top_k=5)

    # 파일 저장을 먼저 해서 print 실패가 저장을 막지 않도록 합니다.
    output_path = save_report(final_state)

    print(final_state["final_report"])
    print(f"\n[저장 완료] {output_path}")


if __name__ == "__main__":
    main()
