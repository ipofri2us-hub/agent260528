# -*- coding: utf-8 -*-
"""
2일차 RAG 실습 - Day2 RAG Agent v1 최종 실행 파일 (통합 실행 파일)

[2일차 RAG Agent v1 흐름에서의 역할]
이 파일은 기능을 새로 구현하는 곳이 아니라, 앞 단계 결과를 하나로 묶는 통합 실행 파일입니다.
검색(rag_search)과 그래프 실행(langgraph_rag_graph_runner)은 모두 앞 단계 파일에 위임하고,
여기서는 그 최종 State를 받아 사람이 읽을 리포트로 정리합니다.

역할:
1. langgraph_rag_graph_runner.py의 run_langgraph_rag()를 호출합니다.
2. 최종 LangGraph State를 받습니다.
3. Mustache 템플릿으로 최종 리포트를 생성합니다.
4. outputs/day2/day2_rag_agent_v1_result.md 파일로 저장합니다.

[결과를 보는 관점]
이 리포트는 답변 문장이 매끄러운지보다, "어떤 문서 근거(retrieved_docs)를 썼는지,
grounding_status는 무엇인지, Trace에 실행 흐름이 제대로 남았는지"를 확인하는 것이 핵심입니다.

[3일차 연결]
day2_rag_agent_v1_result.md는 RAG 검색이 still 한 파일 안에 묶여 있던 시점의 기준 결과,
즉 search_manual Tool 분리 전 기준 결과입니다. 3일차에는 이 RAG 검색 기능이
search_manual MCP Tool로 분리되어 외부에서 호출 가능한 도구가 됩니다.

필요 패키지:
pip install langgraph langchain-core typing-extensions chromadb google-genai python-dotenv pystache

실행 명령어:
python src/day2/day2_rag_agent_v1_20260524_044306.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pystache


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


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
    LangGraph 최종 State를 Mustache 템플릿에 넘길 데이터로 변환합니다.

    State에서 리포트에 필요한 값(근거 출처 retrieved_docs, grounding_status,
    실행 이력 trace)만 골라 표로 그릴 수 있게 가공합니다.
    표 깨짐을 막기 위해 줄바꿈/파이프 문자를 치환하며, 답변이 비면 안내 문구로 대체합니다.
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
    실행 중 오류가 발생했을 때도 같은 Mustache 템플릿으로 오류 리포트를 만들기 위한 데이터입니다.

    Chroma/Ollama/LangGraph 환경 문제로 실행이 끊겨도, 수업 흐름을 잇기 위해
    grounding_status="ERROR"와 오류 메시지를 담은 리포트를 동일한 형식으로 남깁니다.
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
        logger.warning("Mustache 템플릿 파일을 찾지 못했습니다: %s", template_path)
        logger.warning("templates/day2/day2_rag_agent_v1_result.mustache 파일을 먼저 생성해 주세요.")
        return

    template_text = template_path.read_text(encoding="utf-8-sig")
    rendered_text = pystache.render(template_text, template_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered_text, encoding="utf-8-sig")


def main() -> None:
    """
    Day2 RAG Agent v1 최종 실행 함수입니다.
    """
    project_root = get_project_root()
    output_path = project_root / "outputs" / "day2" / "day2_rag_agent_v1_result.md"
    template_path = project_root / "templates" / "day2" / "day2_rag_agent_v1_result.mustache"

    logger.info("Day2 RAG Agent v1 실행을 시작합니다.")
    logger.info("사용자 질문: %s", DEFAULT_USER_QUERY)

    try:
        state = run_day2_agent(DEFAULT_USER_QUERY)
        template_data = prepare_report_data(state)
        save_results(template_data, output_path, template_path)

        logger.info("LangGraph 실행을 완료했습니다.")
        logger.info("grounding_status: %s", state.get("grounding_status", ""))
        logger.info("retrieved_docs 개수: %s", len(state.get("retrieved_docs", [])))
        logger.info("최종 리포트 저장 경로: %s", output_path)

    except Exception as error:
        template_data = prepare_error_data(error)
        save_results(template_data, output_path, template_path)

        logger.error("Day2 RAG Agent v1 실행 중 오류가 발생했습니다.")
        logger.error("오류 리포트를 저장했습니다.")
        logger.error("최종 리포트 저장 경로: %s", output_path)


if __name__ == "__main__":
    main()
