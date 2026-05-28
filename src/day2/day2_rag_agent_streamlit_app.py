# -*- coding: utf-8 -*-
"""
2일차 RAG 실습 - Day2 RAG Agent v1 Streamlit 실행 화면 (보조 UI)

[위치와 목적 - 먼저 읽어둘 것]
이 Streamlit 화면은 기본 수업을 대체하는 도구가 아니라 보조 UI입니다.
2일차의 기본 기준은 CLI 실행 결과와 outputs/day2 산출물이며,
이 화면은 검색 결과(retrieved_docs), State, 최종 답변을 한 화면에서
눈으로 확인하기 위한 보조 수단입니다.

[운영 주석]
Streamlit/Chroma/Ollama 환경 오류가 나면 UI 디버깅에 오래 머무르지 말고,
CLI(day2_rag_agent_v1.py)와 outputs/day2 산출물 중심으로 수업을 진행하세요.

역할:
1. 사용자 질문 입력 화면을 제공합니다.
2. langgraph_rag_graph_runner.py의 run_langgraph_rag()를 호출합니다.
3. 최종 LangGraph State를 화면에 표시합니다.
4. Mustache 템플릿으로 최종 Markdown 리포트를 저장합니다.
5. 저장된 Markdown 리포트를 다운로드할 수 있게 합니다.

이 파일은 Chroma 검색 로직이나 LangGraph Node 로직을 직접 구현하지 않습니다.
실제 Agent 실행은 langgraph_rag_graph_runner.py의 run_langgraph_rag()에 위임합니다.
(즉 CLI 버전 day2_rag_agent_v1.py와 같은 실행 경로를 쓰고, 출력만 화면으로 보여줍니다.)

Streamlit 화면 실행 패키지:
pip install streamlit pandas pystache

기존 Day2 Agent 실행에 필요한 패키지:
pip install langgraph langchain-core typing-extensions chromadb google-genai python-dotenv

Ollama 사전 준비 명령어:
ollama list
ollama pull nomic-embed-text
ollama serve

실행 명령어:
streamlit run src/day2/day2_rag_agent_streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pystache
import streamlit as st


# Windows + VS Code + PowerShell에서 직접 실행할 때
# 같은 폴더의 langgraph_rag_graph_runner.py, graph_state.py를 import하기 위한 경로 설정입니다.
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent

for path in [CURRENT_DIR, SRC_DIR]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


from langgraph_rag_graph_runner import run_langgraph_rag
from graph_state import ManufacturingRAGState


DEFAULT_USER_QUERY = "EQP-EV-03에서 ALM-TEMP-402가 반복 발생했는데 원인 후보와 품질 영향 확인 항목을 알려줘"


def get_project_root() -> Path:
    """
    현재 파일 위치를 기준으로 프로젝트 루트 폴더를 찾습니다.

    프로젝트 위치가 C 드라이브에서 F 드라이브로 이동해도 동작하도록
    특정 경로를 하드코딩하지 않습니다.
    """
    current_file = Path(__file__).resolve()

    for parent in [current_file.parent, *current_file.parents]:
        has_src = (parent / "src").exists()
        has_docs_or_outputs = (parent / "docs").exists() or (parent / "outputs").exists()

        if has_src and has_docs_or_outputs:
            return parent

    return Path.cwd().resolve()


def run_day2_agent(user_query: str) -> ManufacturingRAGState:
    """
    Day2 RAG Agent v1을 실행합니다.

    실제 Chroma 검색과 LangGraph Node 실행은
    langgraph_rag_graph_runner.py의 run_langgraph_rag()가 담당합니다.
    """
    return run_langgraph_rag(user_query)


def prepare_report_data(state: ManufacturingRAGState) -> Dict[str, Any]:
    """
    LangGraph 최종 State를 Mustache 템플릿과 Streamlit 화면에 넘길 데이터로 변환합니다.
    """

    def escape_table_text(value: Any) -> str:
        return str(value or "").replace("\n", " ").replace("|", "/")

    retrieved_docs = state.get("retrieved_docs", [])
    trace_items = state.get("trace", [])
    errors = state.get("errors", [])

    final_answer = str(state.get("final_answer", "") or "").strip()
    if not final_answer:
        final_answer = (
            "검색된 근거 문서가 부족하여 원인 후보를 확정할 수 없습니다. "
            "설비 ID, 알람 코드, 공정 상태, 품질 지표, 정비 이력 정보를 추가로 확인해야 합니다."
        )

    formatted_docs: List[Dict[str, Any]] = []
    for doc in retrieved_docs:
        formatted_docs.append(
            {
                "rank": escape_table_text(doc.get("rank", "")),
                "score": escape_table_text(doc.get("score", "")),
                "distance": escape_table_text(doc.get("distance", "")),
                "doc_name": escape_table_text(doc.get("doc_name", "")),
                "section_title": escape_table_text(doc.get("section_title", "")),
                "chunk_id": escape_table_text(doc.get("chunk_id", "")),
                "keywords": escape_table_text(doc.get("keywords", "")),
                "preview": escape_table_text(doc.get("preview", "")),
            }
        )

    formatted_trace: List[Dict[str, str]] = []
    for item in trace_items:
        formatted_trace.append(
            {
                "node_name": escape_table_text(item.get("node_name", "")),
                "status": escape_table_text(item.get("status", "")),
                "message": escape_table_text(item.get("message", "")),
                "input_summary": escape_table_text(item.get("input_summary", "")),
                "output_summary": escape_table_text(item.get("output_summary", "")),
            }
        )

    return {
        "has_error": False,
        "error_message": "",
        "user_query": state.get("user_query", ""),
        "equipment_id": state.get("equipment_id", ""),
        "alarm_code": state.get("alarm_code", ""),
        "rewritten_query": state.get("rewritten_query", ""),
        "grounding_status": state.get("grounding_status", ""),
        "retry_count": state.get("retry_count", 0),
        "errors_text": ", ".join(str(error) for error in errors) if errors else "없음",
        "final_answer": final_answer,
        "has_retrieved_docs": bool(formatted_docs),
        "retrieved_docs": formatted_docs,
        "has_trace_items": bool(formatted_trace),
        "trace_items": formatted_trace,
    }


def prepare_error_data(error: Exception) -> Dict[str, Any]:
    """
    실행 중 오류가 발생했을 때 Mustache 템플릿과 Streamlit 화면에 넘길 오류 데이터를 만듭니다.
    """
    error_message = str(error)

    return {
        "has_error": True,
        "error_message": error_message,
        "user_query": DEFAULT_USER_QUERY,
        "equipment_id": "",
        "alarm_code": "",
        "rewritten_query": "",
        "grounding_status": "ERROR",
        "retry_count": 0,
        "errors_text": error_message,
        "final_answer": "",
        "has_retrieved_docs": False,
        "retrieved_docs": [],
        "has_trace_items": False,
        "trace_items": [],
    }


def save_results(
    template_data: Dict[str, Any],
    output_path: Path,
    template_path: Path,
) -> None:
    """
    Mustache 템플릿을 사용해 Day2 RAG Agent v1 최종 리포트를 저장합니다.
    """
    if not template_path.exists():
        return

    template_text = template_path.read_text(encoding="utf-8-sig")
    rendered_text = pystache.render(template_text, template_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered_text, encoding="utf-8-sig")


def state_items_to_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    retrieved_docs 또는 trace_items를 Streamlit 표로 보여주기 위한 DataFrame으로 변환합니다.
    """
    if not items:
        return pd.DataFrame()
    return pd.DataFrame(items)


def display_state(template_data: Dict[str, Any]) -> None:
    """
    LangGraph 최종 State를 Streamlit 화면에 표시합니다.

    최종 답변뿐 아니라 핵심 정보(설비 ID·알람 코드·grounding_status),
    검색된 근거 문서, Node 실행 Trace를 함께 보여줍니다.
    여기서 봐야 할 핵심은 답변 문장보다 "어떤 근거로 답했고 Trace가 어떻게 흘렀는가"입니다.
    """
    st.subheader("최종 답변")
    final_answer = template_data.get("final_answer", "")
    if final_answer:
        st.markdown(final_answer)
    elif template_data.get("has_error"):
        st.warning("오류로 인해 최종 답변이 생성되지 않았습니다.")
    else:
        st.info("최종 답변이 비어 있습니다.")

    st.subheader("핵심 정보")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("equipment_id", str(template_data.get("equipment_id", "") or "-"))
    col2.metric("alarm_code", str(template_data.get("alarm_code", "") or "-"))
    col3.metric("grounding_status", str(template_data.get("grounding_status", "") or "-"))
    col4.metric("retry_count", str(template_data.get("retry_count", 0)))

    errors_text = str(template_data.get("errors_text", "없음") or "없음")
    if errors_text != "없음":
        st.warning(errors_text)

    st.subheader("검색된 근거 문서")
    docs_df = state_items_to_dataframe(template_data.get("retrieved_docs", []))
    if docs_df.empty:
        st.info("검색된 근거 문서가 없습니다.")
    else:
        display_columns = [
            "rank",
            "score",
            "distance",
            "doc_name",
            "section_title",
            "chunk_id",
            "keywords",
            "preview",
        ]
        st.dataframe(docs_df[display_columns], use_container_width=True)

    st.subheader("Node 실행 Trace")
    trace_df = state_items_to_dataframe(template_data.get("trace_items", []))
    if trace_df.empty:
        st.info("Node 실행 Trace가 없습니다.")
    else:
        display_columns = [
            "node_name",
            "status",
            "message",
            "input_summary",
            "output_summary",
        ]
        st.dataframe(trace_df[display_columns], use_container_width=True)


def main() -> None:
    """
    Streamlit 화면 전체를 구성합니다.
    """
    st.set_page_config(
        page_title="Day2 RAG Agent v1",
        page_icon="🤖",
        layout="wide",
    )

    st.title("Day2 RAG Agent v1")
    st.caption("Chroma Vector DB + Ollama embedding + LangGraph 기반 RAG Agent 실행 화면")

    project_root = get_project_root()
    output_path = project_root / "outputs" / "day2" / "day2_rag_agent_v1_result.md"
    template_path = project_root / "templates" / "day2" / "day2_rag_agent_v1_result.mustache"

    with st.expander("실행 전 확인 사항", expanded=False):
        st.markdown(
            """
            아래 준비가 되어 있어야 RAG Agent가 정상 실행됩니다.

            1. Markdown 문서를 chunk로 나누는 `chunk_builder` 실행
            2. `chunk_preview.json`을 Chroma Vector DB에 저장하는 `chroma_index_builder` 실행
            3. Ollama 서버 실행
            4. `nomic-embed-text` 모델 설치
            5. `vector_db/chroma_db` 경로에 Chroma DB 생성

            예시 명령어:

            ```powershell
            python src/day2/chunk_builder_날짜_시간.py
            python src/day2/chroma_index_builder_날짜_시간.py
            ollama list
            ollama pull nomic-embed-text
            ollama serve
            ```
            """
        )

    if not template_path.exists():
        st.warning("Mustache 템플릿 파일을 찾지 못했습니다.")
        st.code(str(template_path))

    user_query = st.text_area(
        "사용자 질문",
        value=DEFAULT_USER_QUERY,
        height=120,
    )

    if st.button("RAG Agent 실행", type="primary"):
        if not user_query.strip():
            st.warning("사용자 질문을 입력해 주세요.")
            return

        try:
            with st.spinner("RAG Agent를 실행하는 중입니다..."):
                state = run_day2_agent(user_query)
                template_data = prepare_report_data(state)
                save_results(template_data, output_path, template_path)

        except Exception as error:
            template_data = prepare_error_data(error)
            save_results(template_data, output_path, template_path)

            st.error("RAG Agent 실행 중 오류가 발생했습니다.")
            st.exception(error)
            display_state(template_data)
            return

        st.success("RAG Agent 실행을 완료했습니다.")
        display_state(template_data)

        if template_path.exists():
            st.success("최종 리포트를 저장했습니다.")
            st.code(str(output_path))

        if output_path.exists():
            report_text = output_path.read_text(encoding="utf-8-sig")
            st.download_button(
                label="Markdown 리포트 다운로드",
                data=report_text,
                file_name="day2_rag_agent_v1_result.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
