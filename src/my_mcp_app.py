"""
로봇 신문 Multi-Agent Streamlit 앱
RFM / VLA / 모방학습 뉴스를 실시간 수집해 화면에 표시합니다.

실행:
  streamlit run src/my_mcp_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from my_mcp_graph import run_news_agent_flow, save_report  # noqa: E402

# ─── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="로봇 AI 뉴스 수집기",
    page_icon="🤖",
    layout="wide",
)


# ─── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _render_articles(items: list[dict], label: str) -> None:
    """논문·뉴스 목록을 카드 형태로 렌더링합니다."""
    if not items:
        st.caption(f"{label} 없음")
        return
    for item in items:
        title = item.get("title", "제목 없음")
        link = item.get("link", "#")
        source = item.get("source", "")
        date = item.get("date", "")
        meta = " · ".join(filter(None, [source, date]))
        st.markdown(f"**[{title}]({link})**")
        if meta:
            st.caption(meta)
        st.divider()


def _step_icon(step: str) -> str:
    if "[오류]" in step:
        return "🔴"
    if "완료" in step or "성공" in step:
        return "🟢"
    return "🔵"


# ─── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    top_k = st.slider("토픽당 수집 건수 (논문·뉴스 각각)", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.caption("**데이터 소스**")
    st.caption("• ArXiv Atom API (학술 논문)")
    st.caption("• Google News RSS (뉴스 기사)")
    st.markdown("---")
    st.caption("**MCP 연결**")
    import os
    mcp_url = os.getenv("MY_MCP_URL", "서브프로세스 자동 기동")
    st.caption(f"`{mcp_url}`")


# ─── 메인 UI ──────────────────────────────────────────────────────────────────
st.title("🤖 로봇 AI 뉴스 수집기")
st.markdown(
    "**RFM** (Robot Foundation Model) · **VLA** (Vision-Language-Action) · **모방학습** "
    "관련 최신 ArXiv 논문과 Google 뉴스를 Multi-Agent LangGraph로 수집합니다."
)

user_query = st.text_input(
    "검색 쿼리",
    value="로봇 AI 최신 동향: RFM, VLA, 모방학습 논문과 뉴스를 알려주세요.",
    help="쿼리는 에이전트 로그에 기록됩니다. top_k는 사이드바에서 설정합니다.",
)

run_btn = st.button("▶ 수집 시작", type="primary", use_container_width=True)

# ─── 실행 ──────────────────────────────────────────────────────────────────────
if run_btn:
    # 이전 결과 초기화
    for key in ("state", "report_path"):
        st.session_state.pop(key, None)

    progress = st.progress(0, text="에이전트 초기화 중...")
    status = st.empty()

    def _on_step(label: str, pct: int) -> None:
        progress.progress(pct, text=label)
        status.info(label)

    _on_step("CoordinatorAgent: 쿼리 분석 중...", 10)
    with st.spinner("Multi-Agent 수집 실행 중 (ArXiv · Google News 호출)..."):
        try:
            final_state = run_news_agent_flow(user_query=user_query, top_k=top_k)
            report_path = save_report(final_state)
            st.session_state["state"] = final_state
            st.session_state["report_path"] = report_path
            progress.progress(100, text="수집 완료!")
            status.empty()
        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(f"실행 오류: {exc}")

# ─── 결과 표시 ──────────────────────────────────────────────────────────────────
if "state" in st.session_state:
    state = st.session_state["state"]
    report_path: Path = st.session_state["report_path"]

    st.success(f"수집 완료 — 보고서 저장: `{report_path.name}`")

    # 다운로드 버튼
    with open(report_path, encoding="utf-8") as f:
        md_text = f.read()
    st.download_button(
        "⬇ 마크다운 보고서 다운로드",
        data=md_text,
        file_name="robot_news_report.md",
        mime="text/markdown",
    )

    st.markdown("---")

    # 에이전트 실행 기록 (접기)
    with st.expander("🔍 에이전트 실행 기록", expanded=False):
        for step in state.get("agent_steps", []):
            icon = _step_icon(step)
            st.markdown(f"{icon} {step}")

    st.markdown("---")

    # 3개 토픽 탭
    tab_rfm, tab_vla, tab_imitation = st.tabs([
        "🧠 RFM (Robot Foundation Model)",
        "👁 VLA (Vision-Language-Action)",
        "🎯 모방학습 (Imitation Learning)",
    ])

    with tab_rfm:
        data = state.get("rfm_results", {})
        if data.get("error"):
            st.error(data["error"])
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"📄 학술 논문 ({len(data.get('arxiv', []))}건)")
                _render_articles(data.get("arxiv", []), "논문")
            with c2:
                st.subheader(f"📰 뉴스 ({len(data.get('news', []))}건)")
                _render_articles(data.get("news", []), "뉴스")

    with tab_vla:
        data = state.get("vla_results", {})
        if data.get("error"):
            st.error(data["error"])
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"📄 학술 논문 ({len(data.get('arxiv', []))}건)")
                _render_articles(data.get("arxiv", []), "논문")
            with c2:
                st.subheader(f"📰 뉴스 ({len(data.get('news', []))}건)")
                _render_articles(data.get("news", []), "뉴스")

    with tab_imitation:
        data = state.get("imitation_results", {})
        if data.get("error"):
            st.error(data["error"])
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"📄 학술 논문 ({len(data.get('arxiv', []))}건)")
                _render_articles(data.get("arxiv", []), "논문")
            with c2:
                st.subheader(f"📰 뉴스 ({len(data.get('news', []))}건)")
                _render_articles(data.get("news", []), "뉴스")

else:
    st.info("위 **▶ 수집 시작** 버튼을 눌러 뉴스를 수집합니다.")

# $env:PYTHONIOENCODING = "utf-8"
# streamlit run src/my_mcp_app.py
# # → http://localhost:8501