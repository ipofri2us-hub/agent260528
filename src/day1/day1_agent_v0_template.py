"""
삼성디스플레이 재직자 대상 AI Agent Architecture 1일차 7교시 단순 Agent v0 실습 파일입니다.

이 파일은 초보자가 위에서 아래로 실행 흐름을 이해할 수 있도록
핵심 코드를 run_day1_agent_v0() 함수 안에 모아 둔 교육용 단순 버전입니다.

실습 주제:
- Prompt, Chain, LangGraph, LLM 호출부 분리 구조 통합
- State, Node, Edge, Conditional Edge 이해
- Prompt / Result / Trace Mustache 템플릿 분리

중요:
- 이 파일은 실제 LangGraph를 사용합니다.
- 이 파일은 Gemini/OpenAI/Anthropic/Ollama/NVIDIA SDK를 직접 import하지 않습니다.
- LLM 호출은 llm_client.py의 generate_response(prompt)를 통해서만 수행합니다.
- API Key, Authorization, token, password, secret 값은 출력하거나 저장하지 않습니다.
- 실제 사내 데이터가 아니라 DisplayEdu Fab 교육용 가상 데이터만 사용합니다.
"""

# ============================================================
# 파일명: day1_agent_v0_template.py
# 목적:
#   앞에서 따로 배운 Prompt, Chain, LangGraph, LLM 호출을 하나로 합쳐
#   "가장 단순한 형태의 Agent(에이전트)"를 만들어 보는 1일차 마무리 실습입니다.
#
# Agent란?(초보자 설명):
#   스스로 상황을 보고 "다음에 무엇을 할지" 정하면서 일을 처리하는 프로그램 흐름입니다.
#   여기서는 입력을 확인하고 → 정보가 충분한지 판단하고 →
#   충분하면 로그와 매뉴얼을 조사해 보고서를 만들고,
#   부족하면 "정보를 더 달라"고 되묻습니다.
#   비유: 알람 내용을 보고 조치 방향을 정리해 주는 공장 보조 직원과 같습니다.
#
# 이 파일에서 배우는 것:
#   1. State / Node / Edge / Conditional Edge로 Agent의 판단 흐름을 표현하는 방법
#   2. "조건부 분기"로 정상 처리와 추가 정보 요청을 나누는 방법
#   3. 최종 결과(Result)와 실행 기록(Trace)을 각각 Markdown으로 남기는 방법
#
# 전체 실행 흐름:
#   1. 입력 데이터(JSON, CSV, 매뉴얼)를 읽습니다.
#   2. 초기 State(작업 기록지)를 만듭니다.
#   3. 각 처리 단계를 Node(내부 함수)로 정의합니다.
#   4. Node들을 연결해 Graph를 구성하고 실행합니다.
#      (입력 확인 → 필수정보 확인 → [로그조회 → 매뉴얼검색 → Prompt생성 → LLM호출 → 보고서] / [추가정보 요청])
#   5. 최종 보고서(Result)와 실행 기록(Trace)을 파일로 저장합니다.
#
# 초보자를 위한 비유:
#   이 파일은 "알람 접수 → 판단 → 조사 → 보고서 작성"까지를
#   혼자 처리하는 신입 담당자(Agent v0)의 첫 업무 매뉴얼과 같습니다.
#   (매뉴얼 검색을 본격적인 RAG로 바꾸는 일은 2일차에서 진행합니다.)
# ============================================================

from pathlib import Path
import json
import sys

import pandas as pd
import pystache
# LangGraph 도구: StateGraph(그래프 본체), START/END(시작·끝 표시)
from langgraph.graph import END, START, StateGraph


# ------------------------------------------------------------
# 함수명: run_day1_agent_v0
# 역할:
#   Day1 Agent v0의 전체 판단·처리 흐름을 한 번 실행합니다.
#
# 입력값:
#   없음 (필요한 데이터는 함수 안에서 파일로 읽습니다.)
#
# 출력값:
#   반환값은 없습니다.
#   진행 상황을 화면에 출력하고, 결과(Result)와 실행 기록(Trace)을
#   각각 Markdown 파일로 저장합니다.
#
# 초보자 설명:
#   이 함수 하나가 곧 "Agent v0의 한 번의 업무 처리"입니다.
#   Node(처리 단계)들이 State(기록지)를 주고받으며 일이 진행됩니다.
# ------------------------------------------------------------
def run_day1_agent_v0():
    """
    Day1 Agent v0 단순 버전을 실행합니다.

    이 함수는 초보자가 Agent v0 흐름을 이해할 수 있도록
    핵심 코드를 한 곳에 모아 둔 교육용 단순 버전입니다.

    실행 흐름:
    1. 프로젝트 경로를 찾습니다.
    2. sample_query.json, sample_alarm_logs.csv, alarm_manual.md를 읽습니다.
    3. 초기 state를 만듭니다.
    4. LangGraph Node들을 내부 함수로 정의합니다.
    5. Conditional Edge를 포함한 Graph를 구성합니다.
    6. Agent v0를 실행합니다.
    7. Prompt Mustache 템플릿으로 LLM Prompt를 만듭니다.
    8. llm_client.py를 통해 LLM을 호출합니다.
    9. Result Mustache 템플릿으로 최종 Markdown 결과를 만듭니다.
    10. Trace Mustache 템플릿으로 실행 Trace Markdown을 만듭니다.
    11. outputs/day1/day1_agent_v0_result.md와 day1_agent_v0_trace.md 파일로 저장합니다.
    """
    print("[1일차 Agent v0] 제조 장애 대응 Agent v0를 실행합니다.")

    # 1. 현재 파일 위치를 기준으로 프로젝트 루트 경로를 찾습니다.
    # 이 파일은 src/day1 폴더 안에 있다고 가정합니다.
    # 따라서 parents[2]가 프로젝트 루트입니다.
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]

    # 2. src 폴더를 import 경로에 추가합니다.
    # 이렇게 해야 src/llm_client.py를 불러올 수 있습니다.
    src_dir = project_root / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # 3. 입력 파일, 템플릿 파일, 출력 파일 경로를 준비합니다.
    query_path = project_root / "data" / "sample_query.json"
    csv_path = project_root / "data" / "sample_alarm_logs.csv"
    manual_path = project_root / "docs" / "alarm_manual.md"

    prompt_template_path = project_root / "templates" / "day1" / "day1_agent_v0_prompt.mustache"
    result_template_path = project_root / "templates" / "day1" / "day1_agent_v0_result.mustache"
    trace_template_path = project_root / "templates" / "day1" / "day1_agent_v0_trace.mustache"

    result_path = project_root / "outputs" / "day1" / "day1_agent_v0_result.md"
    trace_path = project_root / "outputs" / "day1" / "day1_agent_v0_trace.md"

    # 4. 입력 데이터를 읽습니다.
    # 단순 버전에서는 인코딩 fallback을 넣지 않고 utf-8-sig 기준으로 읽습니다.
    query_data = json.loads(query_path.read_text(encoding="utf-8-sig"))
    logs = pd.read_csv(csv_path, encoding="utf-8-sig")
    manual_text = manual_path.read_text(encoding="utf-8-sig")

    # 5. 초기 State를 만듭니다.
    # State는 Node 사이를 이동하는 작업 상태입니다.
    state = {
        "user_query": str(query_data.get("user_query", "")),
        "line_id": str(query_data.get("line_id", "")),
        "process_name": str(query_data.get("process_name", "")),
        "equipment_id": str(query_data.get("equipment_id", "")),
        "alarm_code": str(query_data.get("alarm_code", "")),
        "has_required_info": False,
        "next_action": "",
        "log_results": [],
        "log_summary": {},
        "manual_section": "",
        "llm_prompt": "",
        "llm_response": "",
        "final_report": "",
        "messages": [],
        "trace": [],
        "errors": [],
    }

    # 6. Trace를 추가하는 내부 함수입니다.
    # API Key나 민감정보는 trace에 저장하지 않습니다.
    # 현재 예제에서는 교육용 입력값과 node 실행 요약만 저장합니다.
    def add_trace(state, node_name, input_summary, output_summary, next_node):
        state["trace"].append(
            {
                "node_name": node_name,
                "input_summary": input_summary,
                "output_summary": output_summary,
                "next_node": next_node,
            }
        )
        return state

    # 아래 7-1 ~ 7-9 함수들이 각각 하나의 Node(처리 단계)입니다.
    # 공통 규칙: 모든 Node는 state(기록지)를 받아 내용을 채운 뒤 다시 state를 돌려줍니다.

    # 7-1. 입력값을 확인하는 Node입니다.
    def load_input_node(state):
        print("load_input_node 실행")

        for key in ["user_query", "line_id", "process_name", "equipment_id", "alarm_code"]:
            state[key] = str(state.get(key) or "")

        message = f"입력 확인: equipment_id={state['equipment_id'] or '없음'}, alarm_code={state['alarm_code'] or '없음'}"
        print(message)
        state["messages"].append(message)

        return add_trace(
            state,
            "load_input_node",
            "sample_query 기반 입력 State 확인",
            message,
            "check_required_info_node",
        )

    # 7-2. 필수 정보가 있는지 확인하는 Node입니다.
    # 이 Node가 만든 next_action 값이 Conditional Edge에서 사용됩니다.
    def check_required_info_node(state):
        print("check_required_info_node 실행")

        has_equipment = bool(state.get("equipment_id", "").strip())
        has_alarm = bool(state.get("alarm_code", "").strip())

        if has_equipment and has_alarm:
            state["has_required_info"] = True
            state["next_action"] = "investigate"
            message = "LangGraph 분기: 필수 정보 있음 → 로그 조회"
            next_node = "search_log_node"
        else:
            state["has_required_info"] = False
            state["next_action"] = "ask_more_info"
            message = "LangGraph 분기: 필수 정보 부족 → 추가 정보 요청"
            next_node = "ask_more_info_node"

        print(message)
        state["messages"].append(message)

        return add_trace(
            state,
            "check_required_info_node",
            "equipment_id와 alarm_code 존재 여부 확인",
            message,
            next_node,
        )

    # 7-3. Conditional Edge에서 사용할 분기 함수입니다.
    def route_after_required_info_check(state):
        if state.get("next_action") == "investigate":
            return "search_log_node"
        return "ask_more_info_node"

    # 7-4. 로그를 조회하는 Node입니다.
    def search_log_node(state):
        print("search_log_node 실행")

        equipment_id = state.get("equipment_id", "")
        alarm_code = state.get("alarm_code", "")

        filtered = logs[
            (logs["equipment_id"].astype(str) == str(equipment_id))
            & (logs["alarm_code"].astype(str) == str(alarm_code))
        ].copy()

        if "timestamp" in filtered.columns:
            filtered = filtered.sort_values("timestamp")

        records = filtered.to_dict(orient="records")
        state["log_results"] = records

        if not records:
            summary = {
                "total_count": 0,
                "summary_text": "해당 조건의 로그가 없습니다.",
                "severity_counts": {},
            }
        else:
            severity_counts = {}
            timestamps = []
            repeat_values = []

            for row in records:
                severity = str(row.get("severity", ""))
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

                if row.get("timestamp"):
                    timestamps.append(str(row.get("timestamp")))

                try:
                    repeat_values.append(int(float(row.get("repeat_count", 0))))
                except (TypeError, ValueError):
                    pass

            summary = {
                "total_count": len(records),
                "first_timestamp": timestamps[0] if timestamps else "",
                "last_timestamp": timestamps[-1] if timestamps else "",
                "severity_counts": severity_counts,
                "repeat_count_max": max(repeat_values) if repeat_values else None,
                "summary_text": f"관련 로그 {len(records)}건이 확인되었습니다.",
            }

        state["log_summary"] = summary

        message = f"로그 조회: 관련 로그 {len(records)}건 발견"
        print(message)
        state["messages"].append(message)

        return add_trace(
            state,
            "search_log_node",
            f"equipment_id={equipment_id}, alarm_code={alarm_code}",
            message,
            "search_manual_node",
        )

    # 7-5. 매뉴얼을 검색하는 Node입니다.
    def search_manual_node(state):
        print("search_manual_node 실행")

        alarm_code = state.get("alarm_code", "")

        # 실제 RAG 검색은 2일차에서 다룹니다.
        # 여기서는 초보자가 흐름을 이해하기 쉽도록 단순 문자열 검색만 사용합니다.
        manual_lines = manual_text.splitlines()
        found_lines = []

        for line in manual_lines:
            if alarm_code in line or "온도" in line or "확인" in line or "조치" in line:
                found_lines.append(line)

        manual_section = "\n".join(found_lines[:40])

        if not manual_section.strip():
            manual_section = manual_text[:2000]

        state["manual_section"] = manual_section

        message = f"매뉴얼 검색: {alarm_code} 관련 내용 확인"
        print(message)
        state["messages"].append(message)

        return add_trace(
            state,
            "search_manual_node",
            f"alarm_code={alarm_code}",
            message,
            "build_prompt_node",
        )

    # 7-6. Prompt Mustache 템플릿으로 LLM Prompt를 만드는 Node입니다.
    def build_prompt_node(state):
        print("build_prompt_node 실행")

        log_summary_text = json.dumps(
            state.get("log_summary", {}),
            ensure_ascii=False,
            indent=2,
        )

        prompt_template = prompt_template_path.read_text(encoding="utf-8-sig")

        prompt_data = {
            "user_query": state.get("user_query", ""),
            "line_id": state.get("line_id", ""),
            "process_name": state.get("process_name", ""),
            "equipment_id": state.get("equipment_id", ""),
            "alarm_code": state.get("alarm_code", ""),
            "log_summary_text": log_summary_text,
            "manual_section": state.get("manual_section", ""),
        }

        state["llm_prompt"] = pystache.render(prompt_template, prompt_data)

        message = "LLM 프롬프트 생성 완료"
        state["messages"].append(message)

        return add_trace(
            state,
            "build_prompt_node",
            "log_summary와 manual_section 기반 프롬프트 템플릿 렌더링",
            message,
            "generate_response_node",
        )

    # 7-7. llm_client.py를 통해 LLM 응답을 생성하는 Node입니다.
    # 단순 버전에서는 복잡한 try-except를 넣지 않습니다.
    # LLM 오류 처리는 Claude Code 실습에서 추가할 수 있습니다.
    def generate_response_node(state):
        print("generate_response_node 실행")
        print("LLM 응답 생성: llm_client.py를 통해 호출")

        # 이 한 줄이 실제로 LLM에게 묻고 답을 받는 부분입니다.
        # 앞 단계에서 만들어 둔 llm_prompt를 보내고, 돌아온 답변 글을 state에 저장합니다.
        from llm_client import generate_response

        state["llm_response"] = generate_response(state["llm_prompt"])

        message = "llm_client.py를 통한 LLM 응답 생성 완료"
        state["messages"].append(message)

        return add_trace(
            state,
            "generate_response_node",
            "llm_prompt를 llm_client.generate_response에 전달",
            message,
            "build_final_report_node",
        )

    # 7-8. Result Mustache 템플릿으로 최종 Markdown 결과를 만드는 Node입니다.
    def build_final_report_node(state):
        print("build_final_report_node 실행")

        log_summary_text = json.dumps(
            state.get("log_summary", {}),
            ensure_ascii=False,
            indent=2,
        )

        result_template = result_template_path.read_text(encoding="utf-8-sig")

        result_data = {
            "user_query": state.get("user_query", ""),
            "line_id": state.get("line_id", ""),
            "process_name": state.get("process_name", ""),
            "equipment_id": state.get("equipment_id", ""),
            "alarm_code": state.get("alarm_code", ""),
            "log_summary_text": log_summary_text,
            "manual_section": state.get("manual_section", ""),
            "llm_response": state.get("llm_response", ""),
            "additional_message": "",
        }

        state["final_report"] = pystache.render(result_template, result_data)
        state["messages"].append("최종 리포트 생성 완료")

        return add_trace(
            state,
            "build_final_report_node",
            "log_summary, manual_section, llm_response 템플릿 렌더링",
            "final_report 생성 완료",
            "END",
        )

    # 7-9. 정보가 부족할 때 실행되는 Node입니다.
    # 이 경우 LLM을 호출하지 않습니다.
    def ask_more_info_node(state):
        print("ask_more_info_node 실행")

        missing_items = []

        if not state.get("equipment_id", "").strip():
            missing_items.append("equipment_id")

        if not state.get("alarm_code", "").strip():
            missing_items.append("alarm_code")

        missing_text = ", ".join(missing_items) if missing_items else "필수 정보"

        message = (
            f"알람 원인을 확인하려면 {missing_text} 값이 필요합니다. "
            "예: equipment_id=EQP-EV-03, alarm_code=ALM-TEMP-402"
        )

        state["messages"].append(message)
        state["llm_response"] = "정보 부족 케이스이므로 LLM을 호출하지 않았습니다."

        result_template = result_template_path.read_text(encoding="utf-8-sig")

        result_data = {
            "user_query": state.get("user_query", ""),
            "line_id": state.get("line_id", ""),
            "process_name": state.get("process_name", ""),
            "equipment_id": state.get("equipment_id", "") or "없음",
            "alarm_code": state.get("alarm_code", "") or "없음",
            "log_summary_text": "{}",
            "manual_section": "",
            "llm_response": state["llm_response"],
            "additional_message": message,
        }

        state["final_report"] = pystache.render(result_template, result_data)

        return add_trace(
            state,
            "ask_more_info_node",
            f"누락 정보: {missing_text}",
            "추가 정보 요청 결과 생성, LLM 호출 없음",
            "END",
        )

    # 8. LangGraph를 구성합니다.
    # dict 기반 State를 사용해서 초보자가 State 구조를 쉽게 볼 수 있게 했습니다.
    graph = StateGraph(dict)

    graph.add_node("load_input_node", load_input_node)
    graph.add_node("check_required_info_node", check_required_info_node)
    graph.add_node("search_log_node", search_log_node)
    graph.add_node("search_manual_node", search_manual_node)
    graph.add_node("build_prompt_node", build_prompt_node)
    graph.add_node("generate_response_node", generate_response_node)
    graph.add_node("build_final_report_node", build_final_report_node)
    graph.add_node("ask_more_info_node", ask_more_info_node)

    graph.add_edge(START, "load_input_node")
    graph.add_edge("load_input_node", "check_required_info_node")

    # 핵심 갈림길(Conditional Edge):
    # check_required_info_node 다음에 route_after_required_info_check가 갈 곳을 정하고,
    # 아래 딕셔너리가 "그 반환값 → 실제 이동할 Node"를 이어 줍니다.
    # 필수 정보가 있으면 조사 흐름으로, 없으면 추가 정보 요청으로 갈라집니다.
    graph.add_conditional_edges(
        "check_required_info_node",
        route_after_required_info_check,
        {
            "search_log_node": "search_log_node",
            "ask_more_info_node": "ask_more_info_node",
        },
    )

    graph.add_edge("search_log_node", "search_manual_node")
    graph.add_edge("search_manual_node", "build_prompt_node")
    graph.add_edge("build_prompt_node", "generate_response_node")
    graph.add_edge("generate_response_node", "build_final_report_node")
    graph.add_edge("build_final_report_node", END)
    graph.add_edge("ask_more_info_node", END)

    app = graph.compile()

    # 9. Agent v0를 실행합니다.
    final_state = app.invoke(state)

    # 10. Trace Mustache 템플릿으로 Trace Markdown을 만듭니다.
    trace_rows = []

    for idx, item in enumerate(final_state.get("trace", []), start=1):
        trace_rows.append(
            "| {idx} | {node} | {inp} | {out} | {next_node} |".format(
                idx=idx,
                node=str(item.get("node_name", "")).replace("|", "\\|"),
                inp=str(item.get("input_summary", "")).replace("|", "\\|"),
                out=str(item.get("output_summary", "")).replace("|", "\\|"),
                next_node=str(item.get("next_node", "")).replace("|", "\\|"),
            )
        )

    trace_table = "\n".join(trace_rows) if trace_rows else "| - | - | - | - | - |"
    messages = "\n".join(f"- {msg}" for msg in final_state.get("messages", [])) or "- 메시지 없음"
    errors = "\n".join(f"- {err}" for err in final_state.get("errors", [])) or "- 오류 없음"

    if final_state.get("next_action") == "investigate":
        route_result = "search_log_node로 이동"
    else:
        route_result = "ask_more_info_node로 이동"

    trace_template = trace_template_path.read_text(encoding="utf-8-sig")

    trace_data = {
        "user_query": final_state.get("user_query", ""),
        "line_id": final_state.get("line_id", ""),
        "process_name": final_state.get("process_name", ""),
        "equipment_id": final_state.get("equipment_id", "") or "없음",
        "alarm_code": final_state.get("alarm_code", "") or "없음",
        "next_action": final_state.get("next_action", ""),
        "route_result": route_result,
        "trace_table": trace_table,
        "messages": messages,
        "errors": errors,
    }

    trace_report = pystache.render(trace_template, trace_data)

    # 11. 결과 파일을 저장합니다.
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(final_state.get("final_report", ""), encoding="utf-8-sig")
    trace_path.write_text(trace_report, encoding="utf-8-sig")

    print(f"결과 저장: {result_path.relative_to(project_root)}")
    print(f"Trace 저장: {trace_path.relative_to(project_root)}")
    print("다음 단계: 2일차에는 매뉴얼 단순 검색을 RAG 검색으로 확장합니다.")


def main():
    """
    프로그램 시작점입니다.
    """
    run_day1_agent_v0()


# 이 아래 부분은 이 파일을 직접 실행했을 때만 동작합니다.
# (예: 터미널에서 `python src/day1/day1_agent_v0_template.py` 를 입력했을 때)
# 다른 파일에서 import해서 사용할 때는 실행되지 않습니다.
# 초보자 관점에서는 "이 파일의 시작 버튼"이라고 이해하면 됩니다.
if __name__ == "__main__":
    main()
