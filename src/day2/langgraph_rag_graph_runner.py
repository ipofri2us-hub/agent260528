# -*- coding: utf-8 -*-
"""
2일차 RAG 실습 - LangGraph StateGraph 기반 RAG 실행기 (Node·분기 정의)

[2일차 RAG Agent v1 흐름에서의 역할]
이 파일은 RAG Agent의 "두뇌"에 해당합니다. graph_state.py가 정의한 State를
입력받아 갱신하는 Node들과, Node 사이를 잇는 분기 규칙을 정의합니다.
Node는 단순 함수가 아니라 "State를 입력받아 갱신하는 Agent 실행 단위"입니다.

[Node 구성과 실행 순서]
parse_query → retrieve_docs → generate_answer → verify_grounding
                                                      │
                              (근거 없음) ─ 조건부 분기 → query_rewrite → retrieve_docs (재시도)
                              (근거 있음/재시도 한도) → 종료

[조건부 분기의 의미 - 품질 관리]
verify_grounding 이후의 분기는 단순 if가 아니라, "검색 근거 없이 답하는 것"을 막기 위한
품질 관리 구조입니다. 근거가 없으면 답을 확정하지 않고 질의를 재작성해 한 번 더 검색합니다.

[산출물과 4일차 연결]
각 Node가 add_trace로 남기는 실행 이력은 State.trace에 쌓이고,
day2_rag_agent_v1.py가 이를 Markdown(예: langgraph_rag_trace.md / 최종 리포트)으로 저장합니다.
이 Trace는 4일차 Trace 기반 실행 품질 평가의 기반 자료가 됩니다.
(최종 Markdown 리포트 저장 자체는 day2_rag_agent_v1.py가 담당)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent

for path in [CURRENT_DIR, SRC_DIR]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from langgraph.graph import StateGraph, END
import pystache

from graph_state import (
    ManufacturingRAGState,
    create_initial_state,
    add_trace,
    summarize_retrieved_docs,
)
from rag_search import search_top_k
from llm_client import generate_response


MAX_RETRY_COUNT = 1


def get_project_root() -> Path:
    """현재 파일 위치를 기준으로 프로젝트 루트 폴더를 찾습니다."""
    current_path = Path(__file__).resolve()

    for parent in [current_path.parent, *current_path.parents]:
        if (parent / "src").exists() and ((parent / "docs").exists() or (parent / "outputs").exists()):
            return parent

    return Path.cwd().resolve()


def extract_alarm_code(text: str) -> str:
    """ALM-TEMP-402 같은 알람 코드를 추출합니다."""
    if not text:
        return ""
    match = re.search(r"ALM-[A-Z]+-[0-9]+", text)
    return match.group(0) if match else ""


def extract_equipment_id(text: str) -> str:
    """EQP-EV-03 같은 설비 ID를 추출합니다."""
    if not text:
        return ""
    match = re.search(r"EQP-[A-Z]+-[0-9]+", text)
    return match.group(0) if match else ""


def render_prompt_template(template_name: str, data: dict[str, Any]) -> str:
    """templates/day2 폴더의 Mustache 템플릿을 렌더링합니다."""
    template_path = get_project_root() / "templates" / "day2" / template_name
    template_text = template_path.read_text(encoding="utf-8-sig")
    return pystache.render(template_text, data)


def build_llm_prompt(state: ManufacturingRAGState) -> str:
    """LLM에 전달할 프롬프트를 Mustache 템플릿으로 생성합니다."""
    retrieved_docs = state.get("retrieved_docs", [])

    data = {
        "user_query": state.get("user_query", ""),
        "equipment_id": state.get("equipment_id", ""),
        "alarm_code": state.get("alarm_code", ""),
        "equipment_id_display": state.get("equipment_id", "") or "질문에서 명확히 확인되지 않음",
        "alarm_code_display": state.get("alarm_code", "") or "질문에서 명확히 확인되지 않음",
        "retrieved_docs": retrieved_docs,
    }

    return render_prompt_template("rag_answer_prompt.mustache", data)


def build_mock_answer(state: ManufacturingRAGState) -> str:
    """LLM 호출 실패 또는 검색 근거 부족 시 사용할 짧은 교육용 mock 답변을 만듭니다.

    실제 LLM 호출이 실패했을 때 수업 흐름을 유지하기 위한 교육용 대체 답변입니다.
    """
    user_query = state.get("user_query", "")
    equipment_id = state.get("equipment_id", "") or "질문에서 명확히 확인되지 않음"
    alarm_code = state.get("alarm_code", "") or "질문에서 명확히 확인되지 않음"
    retrieved_docs = state.get("retrieved_docs", [])

    if retrieved_docs:
        doc_lines = []
        for doc in retrieved_docs:
            doc_lines.append(
                f"- Rank {doc.get('rank', '')} / "
                f"{doc.get('doc_name', '')} / "
                f"{doc.get('section_title', '')} / "
                f"score={doc.get('score', '')} / "
                f"{doc.get('preview', '')}"
            )
        doc_summary = "\n".join(doc_lines)
    else:
        doc_summary = "- 검색된 근거 문서가 없습니다."

    return f"""
## 1. 질의 요약
- 사용자 질문: {user_query}

## 2. 확인된 설비 ID와 알람 코드
- 설비 ID: {equipment_id}
- 알람 코드: {alarm_code}

## 3. 검색 근거 요약
{doc_summary}

## 4. 원인 후보
- 온도 상승 추세가 있었는지 확인이 필요합니다.
- 냉각 상태 또는 센서 값 변동 가능성을 검토할 수 있습니다.
- 최근 정비 이력 또는 설정 변경 여부를 함께 확인할 수 있습니다.

## 5. 품질 영향 확인 항목
- 알람 발생 전후의 불량률 변화를 확인합니다.
- 알람 발생 전후의 수율 변화를 확인합니다.
- 검사 결과와 반복 알람 시점이 겹치는지 확인합니다.

## 6. 추가 확인 필요 사항
- Chroma Vector DB 검색 결과와 실제 교육용 로그의 일치 여부를 확인합니다.
- 검색 근거가 부족하면 질문을 설비 ID, 알람 코드, 증상 중심으로 다시 작성합니다.
- 원인은 확정하지 않고 담당자 검토가 필요한 후보로만 정리합니다.

## 7. 주의 문구
본 답변은 교육용 문서 검색 기반 초안입니다. 실제 설비 판단이나 조치는 담당자 검토가 필요합니다.
""".strip()


def parse_query_node(state: ManufacturingRAGState) -> ManufacturingRAGState:
    """[Node 1] 사용자 질문에서 검색·Tool 호출에 필요한 조건을 추출합니다.

    질문 문장에서 설비 ID(EQP-*)와 알람 코드(ALM-*)를 뽑아 State에 채웁니다.
    이 값들은 이후 검색 정확도를 높이고, 3일차 search_manual Tool 호출 시
    입력 인자로 그대로 이어지는 핵심 조건입니다.
    """
    print("- parse_query_node 실행")

    query = state.get("rewritten_query") or state.get("user_query", "")
    equipment_id = extract_equipment_id(query)
    alarm_code = extract_alarm_code(query)

    state["equipment_id"] = equipment_id
    state["alarm_code"] = alarm_code

    add_trace(
        state,
        node_name="parse_query_node",
        status="success",
        message="질문에서 설비 ID와 알람 코드를 추출했습니다.",
        input_summary=query,
        output_summary=f"equipment_id={equipment_id or '없음'}, alarm_code={alarm_code or '없음'}",
    )
    return state


def retrieve_docs_node(state: ManufacturingRAGState) -> ManufacturingRAGState:
    """[Node 2] RAG 근거 후보를 확보합니다.

    rag_search.search_top_k()를 호출해 Top-3 chunk를 State.retrieved_docs에 담습니다.
    이 결과는 정답이 아니라 답변 전 검토할 근거 후보 집합이며,
    검색이 0건이면 status=warning과 함께 grounding 단계에서 재작성 분기의 근거가 됩니다.
    """
    print("- retrieve_docs_node 실행")

    query = state.get("rewritten_query") or state.get("user_query", "")

    try:
        retrieved_docs = search_top_k(query, top_k=3)
    except Exception as error:
        retrieved_docs = []
        state["errors"].append(f"RAG 검색 실패: {type(error).__name__}: {error}")

    state["retrieved_docs"] = retrieved_docs  # type: ignore[assignment]

    if retrieved_docs:
        status = "success"
        output_summary = summarize_retrieved_docs(retrieved_docs)  # type: ignore[arg-type]
    else:
        status = "warning"
        output_summary = "retrieved_docs=0건"
        state["errors"].append("검색 결과가 없습니다.")

    add_trace(
        state,
        node_name="retrieve_docs_node",
        status=status,
        message="RAG 검색을 수행했습니다.",
        input_summary=query,
        output_summary=output_summary,
    )
    return state


def generate_answer_node(state: ManufacturingRAGState) -> ManufacturingRAGState:
    """[Node 3] 검색 근거를 바탕으로 답변 초안을 생성합니다.

    retrieved_docs를 프롬프트에 넣어 llm_client.generate_response로 답변을 만듭니다.
    근거가 없거나 LLM 호출이 실패하면 수업 흐름이 끊기지 않도록 교육용 mock 답변으로 대체하고,
    answer_source(llm_client / mock)를 Trace에 남겨 "무엇으로 답했는지"를 추적할 수 있게 합니다.
    """
    print("- generate_answer_node 실행")

    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        answer = build_mock_answer(state)
        state["draft_answer"] = answer
        state["final_answer"] = answer
        add_trace(
            state,
            node_name="generate_answer_node",
            status="warning",
            message="검색 근거가 없어 교육용 mock 답변을 생성했습니다.",
            input_summary="retrieved_docs=0건",
            output_summary="answer_source=mock",
        )
        return state

    try:
        prompt = build_llm_prompt(state)
        answer = generate_response(prompt)

        if not isinstance(answer, str) or not answer.strip():
            answer = build_mock_answer(state)
            answer_source = "mock"
            status = "warning"
            message = "LLM 응답이 비어 있어 교육용 mock 답변을 생성했습니다."
        else:
            answer_source = "llm_client"
            status = "success"
            message = "llm_client.py를 통해 답변을 생성했습니다."

    except Exception as error:
        answer = build_mock_answer(state)
        state["errors"].append(f"LLM 호출 실패: {type(error).__name__}: {error}")
        answer_source = "mock"
        status = "warning"
        message = "LLM 호출 실패로 교육용 mock 답변을 생성했습니다."

    state["draft_answer"] = answer
    state["final_answer"] = answer

    add_trace(
        state,
        node_name="generate_answer_node",
        status=status,
        message=message,
        input_summary=f"retrieved_docs={len(retrieved_docs)}건",
        output_summary=f"answer_source={answer_source}",
    )
    return state


def verify_grounding_node(state: ManufacturingRAGState) -> ManufacturingRAGState:
    """[Node 4] 답변이 검색 근거와 연결되어 있는지 검토합니다.

    검색 근거 유무로 grounding_status(PASS / NEEDS_REWRITE)와 needs_rewrite를 정합니다.
    이 Node가 "근거 없는 답변을 그대로 내보내지 않게" 막는 품질 관문이며,
    재시도 한도(MAX_RETRY_COUNT)에 도달하면 재작성을 멈추고 종료해 무한 루프를 방지합니다.
    """
    print("- verify_grounding_node 실행")

    retrieved_docs = state.get("retrieved_docs", [])

    if retrieved_docs:
        state["grounding_status"] = "PASS"
        state["needs_rewrite"] = False
        message = "검색 근거가 있어 grounding_status를 PASS로 설정했습니다."
    else:
        state["grounding_status"] = "NEEDS_REWRITE"
        if state.get("retry_count", 0) >= MAX_RETRY_COUNT:
            state["needs_rewrite"] = False
            message = "재시도 횟수 제한에 도달하여 재작성 없이 종료합니다."
        else:
            state["needs_rewrite"] = True
            message = "검색 근거가 없어 질의를 한 번 재작성합니다."

    add_trace(
        state,
        node_name="verify_grounding_node",
        status="success" if retrieved_docs else "warning",
        message=message,
        input_summary=f"retrieved_docs={len(retrieved_docs)}건, retry_count={state.get('retry_count', 0)}",
        output_summary=(
            f"grounding_status={state['grounding_status']}, "
            f"needs_rewrite={state['needs_rewrite']}, "
            f"retry_count={state['retry_count']}"
        ),
    )
    return state


def query_rewrite_node(state: ManufacturingRAGState) -> ManufacturingRAGState:
    """[Node 5] 근거 부족 시 재검색 가능한 질의로 변환합니다.

    추출된 설비 ID·알람 코드와 핵심 증상 키워드를 합쳐 rewritten_query를 만들고,
    retry_count를 올린 뒤 다시 retrieve_docs로 보냅니다.
    "검색이 비면 질문을 더 검색 친화적으로 바꿔 한 번 더 시도"하는 자가 보정 루프입니다.
    """
    print("- query_rewrite_node 실행")

    equipment_id = state.get("equipment_id", "")
    alarm_code = state.get("alarm_code", "")

    keyword_parts = [equipment_id, alarm_code, "온도 상승", "반복 알람", "품질 영향", "원인 후보"]
    rewritten_query = " ".join(part for part in keyword_parts if part).strip()

    if not rewritten_query:
        rewritten_query = "온도 상승 반복 알람 품질 영향 원인 후보 확인 항목"

    state["rewritten_query"] = rewritten_query
    state["retry_count"] = state.get("retry_count", 0) + 1
    state["needs_rewrite"] = False

    add_trace(
        state,
        node_name="query_rewrite_node",
        status="success",
        message="검색 실패 후 핵심 키워드 중심으로 질의를 재작성했습니다.",
        input_summary=state.get("user_query", ""),
        output_summary=rewritten_query,
    )
    return state


def should_rewrite_or_end(state: ManufacturingRAGState) -> str:
    """[조건부 분기] verify_grounding 이후 재작성/종료를 결정하는 라우팅 함수입니다.

    근거가 확보되면(PASS) 종료, 재시도 한도에 닿으면 종료, 그 외 근거 부족이면 재작성으로 보냅니다.
    이 라우팅이 "근거 없는 답변 방지"와 "무한 재시도 방지"를 동시에 거는 품질 관리 지점입니다.
    """
    if state.get("grounding_status") == "PASS":
        return "end"
    if state.get("retry_count", 0) >= MAX_RETRY_COUNT:
        return "end"
    if state.get("needs_rewrite") is True:
        return "rewrite"
    return "end"


def build_graph():
    """LangGraph StateGraph를 구성합니다.

    Node 5개를 등록하고 실행 순서(엣지)를 연결합니다. 핵심은 verify_grounding 뒤의
    add_conditional_edges로, should_rewrite_or_end 결과에 따라 query_rewrite(재시도) 또는
    END(종료)로 갈라지는 부분입니다. query_rewrite는 다시 retrieve_docs로 이어져 재검색 루프를 만듭니다.
    """
    graph_builder = StateGraph(ManufacturingRAGState)
    graph_builder.add_node("parse_query", parse_query_node)
    graph_builder.add_node("retrieve_docs", retrieve_docs_node)
    graph_builder.add_node("generate_answer", generate_answer_node)
    graph_builder.add_node("verify_grounding", verify_grounding_node)
    graph_builder.add_node("query_rewrite", query_rewrite_node)
    graph_builder.set_entry_point("parse_query")
    graph_builder.add_edge("parse_query", "retrieve_docs")
    graph_builder.add_edge("retrieve_docs", "generate_answer")
    graph_builder.add_edge("generate_answer", "verify_grounding")
    graph_builder.add_conditional_edges(
        "verify_grounding",
        should_rewrite_or_end,
        {"rewrite": "query_rewrite", "end": END},
    )
    graph_builder.add_edge("query_rewrite", "retrieve_docs")
    return graph_builder.compile()


def run_langgraph_rag(user_query: str) -> ManufacturingRAGState:
    """day2_rag_agent_v1.py가 호출하는 외부 인터페이스입니다.

    질문으로 초기 State를 만들고 그래프를 끝까지 실행한 뒤, 최종 State를 돌려줍니다.
    호출 측(CLI / Streamlit)은 이 최종 State의 final_answer, retrieved_docs, trace만 보고
    리포트를 렌더링하면 되므로, Node·검색 로직을 알 필요가 없습니다.
    """
    initial_state = create_initial_state(user_query)
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    return final_state


def main() -> None:
    """단독 실행 시 샘플 질문으로 LangGraph RAG 흐름을 실행합니다."""
    sample_query = "EQP-EV-03에서 ALM-TEMP-402가 반복 발생했는데 원인 후보와 품질 영향 확인 항목을 알려줘"
    final_state = run_langgraph_rag(sample_query)
    print("[완료] LangGraph RAG 실행")
    print(f"grounding_status: {final_state.get('grounding_status', '')}")
    print()
    print(final_state.get("final_answer", ""))


if __name__ == "__main__":
    main()
