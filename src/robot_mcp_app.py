# C:\work\agent260528\src\robot_mcp_app.py
from __future__ import annotations
import streamlit as st
from robot_mcp_graph import run_multi_agent_flow, save_result_markdown

def main():
    st.set_page_config(page_title="설정 동기화형 로봇 에이전트 인프라", layout="wide")
    st.title("🤖 Config 연동 프로덕션 로봇 AI 에이전트")
    st.info("본 인프라는 사내 형상 관리 규칙에 통합 바인딩되어 동작하는 엔터프라이즈 모듈입니다.")

    user_query = st.text_input(
        "에이전트 제어 명령을 입력하세요",
        value="휴머노이드 로봇 VLA 및 모방학습 관련 최신 논문과 로봇신문 기사 수집 요청"
    )

    if st.button("에이전트 인프라 가동 (Dynamic Registry)", type="primary"):
        with st.spinner("config.yaml의 정밀 키워드를 해석하여 하이브리드 데이터 피드 수혈 중..."):
            state = run_multi_agent_flow(user_query=user_query, mode="fastmcp")
            save_result_markdown(state)
            
        st.subheader("📊 1. 에이전트 지능적 추론 프로세스 (agent_steps)")
        for idx, step in enumerate(state.get("agent_steps", []), 1):
            if "grounded" in step or "10건" in step:
                st.success(f"{idx}단계: {step}")
            else:
                st.info(f"{idx}단계: {step}")

        st.subheader("📝 2. 실시간 컨텍스트 결합형 최종 기술 보고서")
        st.markdown(state.get("final_summary", ""))
        st.success("💾 산출물이 outputs/day3/day3_multi_agent_roles_result.md 경로에 업데이트되었습니다.")

if __name__ == "__main__":
    main()