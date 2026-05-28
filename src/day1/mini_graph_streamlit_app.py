"""
삼성디스플레이 재직자 대상 AI Agent Architecture 1일차 6교시 Mini Graph Streamlit 실습 화면입니다.

역할:
- data/sample_query.json에서 실습 입력 조건을 읽습니다.
- data/sample_alarm_logs.csv에서 설비 ID와 알람 코드 기준으로 로그를 필터링합니다.
- templates/day1/mini_graph_llm_prompt.mustache로 LLM Prompt를 만듭니다.
- llm_client.py의 generate_response(prompt)를 통해 LLM을 호출합니다.
- templates/day1/mini_graph_trace_report.mustache로 Trace Markdown 결과를 만듭니다.
- outputs/day1/mini_graph_trace.md로 저장합니다.
- LangGraph의 State, Node, Edge, Conditional Edge 흐름을 Streamlit 화면에서 확인합니다.

중요:
- 이 파일은 실제 LangGraph를 사용합니다.
- 이 파일은 Gemini/OpenAI/Anthropic/Ollama/NVIDIA SDK를 직접 import하지 않습니다.
- LLM 호출은 llm_client.py의 generate_response(prompt)를 통해서만 수행합니다.
- API Key, Authorization, Bearer token, password, secret, .env 전체 내용은 화면에 표시하지 않습니다.
- os.environ 전체를 출력하지 않습니다.
- 실제 사내 데이터가 아니라 DisplayEdu Fab 교육용 가상 데이터만 사용합니다.
- 이 화면은 Prompt 수정 실습이 아니라 LangGraph 실행 흐름과 Conditional Edge 이해를 위한 화면입니다.
"""

# ============================================================
# 파일명: mini_graph_streamlit_app.py
# 목적:
#   mini_graph_runner.py의 LangGraph 흐름을 웹 화면(Streamlit)에서 확인하는 실습 파일입니다.
#   특히 "조건에 따라 거쳐 가는 Node가 달라지는" Conditional Edge를 눈으로 보는 데 집중합니다.
#
# 다시 보는 핵심 용어(초보자 설명):
#   - State: 작업하며 채워 나가는 "작업 기록지"(dict)
#   - Node: 흐름 중 하나의 처리 단계(예: 로그 조회 단계)
#   - Edge: 한 단계 다음에 갈 단계를 잇는 연결선
#   - Conditional Edge: 상황(정보 유무)에 따라 다음 단계가 갈라지는 갈림길
#
# 이 파일에서 배우는 것:
#   1. "정상 케이스 / 정보 부족 케이스"를 골라 실행하며 분기를 비교하는 방법
#   2. 실제로 실행된 Node 순서와 Trace(실행 기록)를 화면에서 확인하는 방법
#   3. 같은 그래프라도 입력 State에 따라 결과 경로가 달라진다는 점
#
# 전체 실행 흐름:
#   1. 프로젝트 루트를 찾고 사용 파일/실행 흐름을 화면에 보여 줍니다.
#   2. 실행 케이스(정상/정보 부족)를 선택합니다.
#   3. "Mini Graph 실행하기" 버튼을 누르면 run_mini_graph_case()가 그래프를 실행합니다.
#   4. 분기 결과, 실행된 Node 순서, Trace, LLM 응답을 화면에 표시하고 저장합니다.
#
# 초보자를 위한 비유:
#   공장의 "작업 순서도"를 화면에 띄워 놓고,
#   입력 상황에 따라 어느 길로 제품이 흘러가는지 직접 눌러 보며 확인하는 화면입니다.
#
# 실행 방법:
#   streamlit run src/day1/mini_graph_streamlit_app.py
# ============================================================

from pathlib import Path
import json
import sys

import pandas as pd
import pystache
import streamlit as st
# LangGraph 도구: StateGraph(그래프 본체), START/END(시작·끝 표시)
from langgraph.graph import END, START, StateGraph


# 현재 파일 위치를 기준으로 data/src가 함께 있는 프로젝트 루트 폴더를 찾는 함수입니다.
# (실습 폴더를 다른 위치로 옮겨도 파일을 잘 찾도록 돕습니다.)
def find_project_root():
    """
    현재 파일 위치를 기준으로 프로젝트 루트 폴더를 찾습니다.

    프로젝트 폴더가 C 드라이브가 아닌 다른 위치로 이동해도 동작하도록
    data, src 폴더가 함께 있는 위치를 찾습니다.
    """
    current_file = Path(__file__).resolve()

    for parent in [current_file.parent, *current_file.parents]:
        if (parent / "data").exists() and (parent / "src").exists():
            return parent

    # 이 파일이 src/day1 폴더 안에 있다는 전제에서 사용하는 예비 경로입니다.
    return current_file.parents[2]


# ------------------------------------------------------------
# 함수명: prepare_mini_graph_inputs
# 역할:
#   그래프를 실행하기 전에 필요한 파일 경로를 준비하고,
#   필수 파일이 있는지 확인한 뒤, 입력 데이터(JSON, CSV)를 읽어 둡니다.
#
# 입력값:
#   project_root: 프로젝트 루트 폴더 경로(Path)
#
# 출력값:
#   경로와 읽어 둔 데이터가 담긴 dict
#
# 초보자 설명:
#   이 함수는 "준비 단계"만 담당하고 그래프를 실행하지는 않습니다.
#   실제 실행은 버튼을 누른 뒤 run_mini_graph_case()에서 이루어집니다.
# ------------------------------------------------------------
def prepare_mini_graph_inputs(project_root):
    """
    Mini Graph 실행에 필요한 입력 파일과 템플릿 경로를 준비합니다.

    이 함수에서는 Graph를 실행하지 않습니다.
    파일 경로 준비, 필수 파일 확인, query/log 데이터 읽기만 수행합니다.
    """
    # src/llm_client.py를 import하기 위해 src 폴더를 import 경로에 추가합니다.
    src_dir = project_root / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    query_path = project_root / "data" / "sample_query.json"
    csv_path = project_root / "data" / "sample_alarm_logs.csv"
    prompt_template_path = project_root / "templates" / "day1" / "mini_graph_llm_prompt.mustache"
    trace_template_path = project_root / "templates" / "day1" / "mini_graph_trace_report.mustache"
    output_path = project_root / "outputs" / "day1" / "mini_graph_trace.md"

    required_files = [
        query_path,
        csv_path,
        prompt_template_path,
        trace_template_path,
    ]

    for file_path in required_files:
        if not file_path.exists():
            relative_path = file_path.relative_to(project_root)
            raise FileNotFoundError(f"필수 파일을 찾지 못했습니다: {relative_path}")

    query_data = json.loads(query_path.read_text(encoding="utf-8-sig"))
    logs = pd.read_csv(csv_path, encoding="utf-8-sig")

    return {
        "project_root": project_root,
        "query_path": query_path,
        "csv_path": csv_path,
        "prompt_template_path": prompt_template_path,
        "trace_template_path": trace_template_path,
        "output_path": output_path,
        "query_data": query_data,
        "logs": logs,
    }


# ------------------------------------------------------------
# 함수명: run_mini_graph_case
# 역할:
#   선택한 케이스(정상 / 정보 부족) 하나에 대해 LangGraph를 구성하고 실행한 뒤,
#   실행 기록(Trace)을 Markdown으로 만들어 저장합니다.
#
# 입력값:
#   graph_data: prepare_mini_graph_inputs()가 준비한 값 모음(dict)
#   case_type: "정상 케이스" 또는 "정보 부족 케이스"
#
# 출력값:
#   최종 State, Trace 보고서 등 화면 표시에 쓸 값들이 담긴 dict
#
# 초보자 설명:
#   이 함수 안에서 Node(처리 단계)들을 정의하고, 그것들을 연결해 그래프를 만든 뒤 실행합니다.
#   case_type이 "정보 부족 케이스"이면 일부러 equipment_id를 비워서,
#   조건부 분기가 "추가 정보 요청" 쪽으로 갈라지는 모습을 보여 줍니다.
# ------------------------------------------------------------
def run_mini_graph_case(graph_data, case_type):
    """
    선택한 케이스 하나를 LangGraph로 실행합니다.

    case_type:
    - 정상 케이스
    - 정보 부족 케이스
    """
    project_root = graph_data["project_root"]
    query_data = graph_data["query_data"]
    logs = graph_data["logs"]
    prompt_template_path = graph_data["prompt_template_path"]
    trace_template_path = graph_data["trace_template_path"]
    output_path = graph_data["output_path"]

    state = {
        "user_query": str(query_data.get("user_query", "")),
        "line_id": str(query_data.get("line_id", "")),
        "process_name": str(query_data.get("process_name", "")),
        "equipment_id": str(query_data.get("equipment_id", "")),
        "alarm_code": str(query_data.get("alarm_code", "")),
        "has_required_info": False,
        "log_results": [],
        "log_summary": {},
        "llm_prompt": "",
        "llm_response": "",
        "next_action": "",
        "messages": [],
        "trace": [],
        "errors": [],
    }

    if case_type == "정보 부족 케이스":
        state["equipment_id"] = ""
        state["user_query"] = "ALM-TEMP-402 교육용 알람이 반복 발생한 것 같습니다. 어떤 정보를 더 확인해야 하나요?"

    # 아래 내부 함수들이 각각 하나의 Node(처리 단계)입니다.
    # 모든 Node는 state(기록지)를 받아 내용을 채운 뒤 다시 state를 돌려줍니다.
    # add_trace는 "이 단계에서 무슨 일을 했는지"를 기록지에 남기는 도우미 함수입니다.
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

    def start_node(state):
        print("start_node 실행")
        state["messages"].append("LangGraph mini graph 실행을 시작합니다.")
        return add_trace(state, "start_node", "초기 State 입력", "실행 시작 메시지 추가", "parse_query_node")

    def parse_query_node(state):
        print("parse_query_node 실행")
        for key in ["user_query", "line_id", "process_name", "equipment_id", "alarm_code"]:
            state[key] = str(state.get(key) or "")
        output_summary = f"equipment_id={state['equipment_id'] or '없음'}, alarm_code={state['alarm_code'] or '없음'} 확인"
        state["messages"].append(output_summary)
        return add_trace(state, "parse_query_node", "user_query와 query 필드 확인", output_summary, "check_required_info_node")

    def check_required_info_node(state):
        print("check_required_info_node 실행")
        has_equipment_id = bool(state.get("equipment_id", "").strip())
        has_alarm_code = bool(state.get("alarm_code", "").strip())
        if has_equipment_id and has_alarm_code:
            state["has_required_info"] = True
            state["next_action"] = "search_log"
            message = "필수 정보 있음 → search_log_node로 이동"
            next_node = "search_log_node"
        else:
            state["has_required_info"] = False
            state["next_action"] = "ask_more_info"
            message = "필수 정보 부족 → ask_more_info_node로 이동"
            next_node = "ask_more_info_node"
        print(f"check_required_info_node: {message}")
        state["messages"].append(message)
        return add_trace(state, "check_required_info_node", "equipment_id와 alarm_code 존재 여부 확인", message, next_node)

    def route_after_required_info_check(state):
        if state.get("next_action") == "search_log":
            return "search_log_node"
        return "ask_more_info_node"

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
        state["log_results"] = filtered.to_dict(orient="records")
        message = f"관련 로그 {len(state['log_results'])}건 발견"
        print(f"search_log_node: {message}")
        state["messages"].append(message)
        return add_trace(state, "search_log_node", f"equipment_id={equipment_id}, alarm_code={alarm_code}", message, "summarize_result_node")

    def ask_more_info_node(state):
        print("ask_more_info_node 실행")
        missing = []
        if not state.get("equipment_id", "").strip():
            missing.append("equipment_id")
        if not state.get("alarm_code", "").strip():
            missing.append("alarm_code")
        missing_text = ", ".join(missing) if missing else "필수 정보"
        message = (
            f"알람 원인을 확인하려면 {missing_text} 값이 필요합니다. "
            "교육용 예시로는 equipment_id=EQP-EV-03, alarm_code=ALM-TEMP-402를 사용할 수 있습니다."
        )
        state["messages"].append(message)
        state["llm_response"] = "정보 부족 케이스이므로 LLM을 호출하지 않았습니다."
        return add_trace(state, "ask_more_info_node", f"누락 정보: {missing_text}", "추가 정보 요청 메시지 생성, LLM 호출 없음", "END")

    def summarize_result_node(state):
        print("summarize_result_node 실행")
        records = state.get("log_results", [])
        if not records:
            summary = {
                "total_count": 0,
                "message": "해당 조건의 로그가 없습니다.",
                "severity_counts": {},
                "repeat_count_max": None,
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
            }
        state["log_summary"] = summary
        if summary.get("total_count", 0) == 0:
            message = "해당 조건의 로그가 없습니다."
        else:
            message = (
                f"관련 로그 {summary['total_count']}건 요약 완료 "
                f"(최초={summary.get('first_timestamp', 'N/A')}, "
                f"마지막={summary.get('last_timestamp', 'N/A')})"
            )
        state["messages"].append(message)
        return add_trace(state, "summarize_result_node", "log_results 요약", message, "build_llm_prompt_node")

    def build_llm_prompt_node(state):
        print("build_llm_prompt_node 실행")
        summary_text = json.dumps(state.get("log_summary", {}), ensure_ascii=False, indent=2)
        prompt_template = prompt_template_path.read_text(encoding="utf-8-sig")
        prompt_data = {
            "user_query": state.get("user_query", ""),
            "line_id": state.get("line_id", ""),
            "process_name": state.get("process_name", ""),
            "equipment_id": state.get("equipment_id", ""),
            "alarm_code": state.get("alarm_code", ""),
            "summary_text": summary_text,
        }
        state["llm_prompt"] = pystache.render(prompt_template, prompt_data)
        state["messages"].append("LLM 요약 응답 생성을 위한 프롬프트를 생성했습니다.")
        return add_trace(state, "build_llm_prompt_node", "log_summary 기반 프롬프트 템플릿 렌더링", "llm_prompt 생성 완료", "generate_llm_response_node")

    def generate_llm_response_node(state):
        print("generate_llm_response_node 실행")
        print("generate_llm_response_node: llm_client.py를 통해 LLM 응답 생성")
        from llm_client import generate_response
        state["llm_response"] = generate_response(state["llm_prompt"])
        message = "llm_client.py를 통한 LLM 응답 생성 완료"
        state["messages"].append(message)
        return add_trace(state, "generate_llm_response_node", "llm_prompt를 llm_client.generate_response에 전달", message, "END")

    # Node들을 연결해 Graph(작업 순서도)를 만듭니다.
    #   - add_node("이름", 함수): 처리 단계 하나를 등록
    #   - add_edge(A, B): A 다음에 항상 B로 이동
    #   - add_conditional_edges(...): 조건에 따라 갈 곳이 달라지는 갈림길
    #   - compile(): 순서도를 실제 실행 가능한 형태로 완성
    graph = StateGraph(dict)
    graph.add_node("start_node", start_node)
    graph.add_node("parse_query_node", parse_query_node)
    graph.add_node("check_required_info_node", check_required_info_node)
    graph.add_node("search_log_node", search_log_node)
    graph.add_node("ask_more_info_node", ask_more_info_node)
    graph.add_node("summarize_result_node", summarize_result_node)
    graph.add_node("build_llm_prompt_node", build_llm_prompt_node)
    graph.add_node("generate_llm_response_node", generate_llm_response_node)
    graph.add_edge(START, "start_node")
    graph.add_edge("start_node", "parse_query_node")
    graph.add_edge("parse_query_node", "check_required_info_node")
    graph.add_conditional_edges(
        "check_required_info_node",
        route_after_required_info_check,
        {
            "search_log_node": "search_log_node",
            "ask_more_info_node": "ask_more_info_node",
        },
    )
    graph.add_edge("search_log_node", "summarize_result_node")
    graph.add_edge("summarize_result_node", "build_llm_prompt_node")
    graph.add_edge("build_llm_prompt_node", "generate_llm_response_node")
    graph.add_edge("generate_llm_response_node", END)
    graph.add_edge("ask_more_info_node", END)

    app = graph.compile()
    final_state = app.invoke(state)

    trace_rows = []
    for index, item in enumerate(final_state.get("trace", []), start=1):
        trace_rows.append(
            "| {idx} | {node} | {inp} | {out} | {next_node} |".format(
                idx=index,
                node=str(item.get("node_name", "")).replace("|", "\\|"),
                inp=str(item.get("input_summary", "")).replace("|", "\\|"),
                out=str(item.get("output_summary", "")).replace("|", "\\|"),
                next_node=str(item.get("next_node", "")).replace("|", "\\|"),
            )
        )

    trace_table = "\n".join(trace_rows) if trace_rows else "| - | - | - | - | - |"
    message_text = "\n".join(f"- {message}" for message in final_state.get("messages", [])) or "- 메시지 없음"
    error_text = "\n".join(f"- {error}" for error in final_state.get("errors", [])) or "- 오류 없음"
    summary_text = json.dumps(final_state.get("log_summary", {}), ensure_ascii=False, indent=2)
    route_result = "search_log_node로 이동" if final_state.get("next_action") == "search_log" else "ask_more_info_node로 이동"
    llm_prompt = final_state.get("llm_prompt", "") or "정보 부족 케이스이므로 LLM 프롬프트를 생성하지 않았습니다."
    llm_response = final_state.get("llm_response", "") or "LLM 응답이 없습니다."

    template_data = {
        "case_name": case_type,
        "user_query": final_state.get("user_query", ""),
        "line_id": final_state.get("line_id", ""),
        "process_name": final_state.get("process_name", ""),
        "equipment_id_display": final_state.get("equipment_id", "") or "없음",
        "alarm_code_display": final_state.get("alarm_code", "") or "없음",
        "trace_table": trace_table,
        "next_action": final_state.get("next_action", ""),
        "route_result": route_result,
        "summary_text": summary_text,
        "llm_prompt": llm_prompt,
        "llm_response": llm_response,
        "error_text": error_text,
        "message_text": message_text,
    }

    template_text = trace_template_path.read_text(encoding="utf-8-sig")
    trace_report = pystache.render(template_text, template_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(trace_report, encoding="utf-8-sig")

    return {
        "case_type": case_type,
        "final_state": final_state,
        "trace_report": trace_report,
        "output_path": output_path,
        "project_root": project_root,
    }


def main():
    """
    Streamlit 화면 시작점입니다.
    """
    st.set_page_config(
        page_title="Day1 Mini Graph 실습",
        layout="wide",
    )

    st.title("1일차 Mini Graph Streamlit 실습 화면")
    st.caption("LangGraph의 State, Node, Edge, Conditional Edge 흐름을 화면에서 확인하는 교육용 예제입니다.")

    st.markdown(
        """
이 화면은 `mini_graph_runner_simple` 예제의 LangGraph 흐름을 Streamlit으로 보여줍니다.

- **State**는 Agent가 들고 다니는 작업 상태입니다.
- **Node**는 하나의 처리 단계입니다.
- **Edge**는 다음 단계로 이동하는 연결입니다.
- **Conditional Edge**는 조건에 따라 다음 단계가 달라지는 연결입니다.
- 이 실습에서는 Prompt 수정이 아니라, Graph의 분기와 Trace를 확인하는 데 집중합니다.
        """
    )

    st.subheader("실행 안내")

    col_cli, col_ui = st.columns(2)

    with col_cli:
        st.markdown("**CLI 버전 실행**")
        st.code(
            "python src/day1/mini_graph_runner_simple_20260523_122444.py",
            language="powershell",
        )

    with col_ui:
        st.markdown("**Streamlit 버전 실행**")
        st.code(
            "streamlit run src/day1/mini_graph_streamlit_app.py",
            language="powershell",
        )

    project_root = find_project_root()

    try:
        graph_data = prepare_mini_graph_inputs(project_root)
    except Exception as exc:
        st.error("Mini Graph 입력 데이터를 준비하는 중 오류가 발생했습니다.")
        st.code(str(exc)[:1000])
        return

    st.info(f"프로젝트 루트: {project_root}")

    st.sidebar.header("실행 안내")
    st.sidebar.code(
        "streamlit run src/day1/mini_graph_streamlit_app.py",
        language="powershell",
    )

    st.sidebar.header("사용 파일")

    file_paths = [
        "data/sample_query.json",
        "data/sample_alarm_logs.csv",
        "templates/day1/mini_graph_llm_prompt.mustache",
        "templates/day1/mini_graph_trace_report.mustache",
        "outputs/day1/mini_graph_trace.md",
    ]

    for relative_path in file_paths:
        st.sidebar.markdown(f"- `{relative_path}`")

    st.sidebar.header("주의 사항")
    st.sidebar.warning(
        "API Key, .env 내용, Authorization token, password, secret 값은 화면에 표시하지 않습니다."
    )

    st.subheader("사용 파일 내용 확인")

    for relative_path in file_paths:
        file_path = project_root / relative_path

        with st.expander(relative_path):
            if file_path.exists():
                if file_path.suffix.lower() == ".csv":
                    df = pd.read_csv(file_path, encoding="utf-8-sig")
                    st.dataframe(df, use_container_width=True)
                else:
                    text = file_path.read_text(encoding="utf-8-sig")
                    st.code(text[:5000], language="markdown")
                    if len(text) > 5000:
                        st.caption("파일 내용이 길어 앞 5,000자만 표시했습니다.")
            else:
                st.warning(f"파일을 찾지 못했습니다: {relative_path}")

    case_type = st.radio(
        "실행 케이스 선택",
        ["정상 케이스", "정보 부족 케이스"],
        horizontal=True,
    )

    graph_flow_text = """
START
 → start_node
 → parse_query_node
 → check_required_info_node
 → 조건부 분기
    - 필수 정보 있음:
      search_log_node
      → summarize_result_node
      → build_llm_prompt_node
      → generate_llm_response_node
      → END

    - 필수 정보 부족:
      ask_more_info_node
      → END
"""

    st.subheader("LangGraph 실행 흐름")
    st.code(graph_flow_text, language="text")

    if "mini_graph_result" not in st.session_state:
        st.session_state["mini_graph_result"] = None

    if st.button("Mini Graph 실행하기", type="primary"):
        try:
            with st.spinner("LangGraph mini graph를 실행하는 중입니다..."):
                st.session_state["mini_graph_result"] = run_mini_graph_case(
                    graph_data=graph_data,
                    case_type=case_type,
                )
        except Exception as exc:
            st.error("Mini Graph 실행 중 오류가 발생했습니다.")
            st.code(str(exc)[:1000])

    result = st.session_state["mini_graph_result"]

    if result is None:
        st.info("아직 Mini Graph를 실행하지 않았습니다. 실행 케이스를 선택한 뒤 버튼을 눌러 주세요.")
        return

    final_state = result["final_state"]
    trace_report = result["trace_report"]

    st.subheader("Conditional Edge 분기 결과")

    if final_state.get("next_action") == "search_log":
        st.success("Conditional Edge 결과: 필수 정보 있음 → search_log_node로 이동")
    else:
        st.warning("Conditional Edge 결과: 필수 정보 부족 → ask_more_info_node로 이동")

    executed_nodes = [
        item.get("node_name", "")
        for item in final_state.get("trace", [])
    ]

    st.subheader("실제 실행된 Node 순서")
    st.code(" → ".join(executed_nodes), language="text")

    st.subheader("Node 실행 Trace")
    trace_df = pd.DataFrame(final_state.get("trace", []))
    st.dataframe(trace_df, use_container_width=True)

    st.subheader("로그 요약")
    st.json(final_state.get("log_summary", {}))

    st.subheader("생성된 LLM Prompt")

    if final_state.get("llm_prompt"):
        st.code(final_state["llm_prompt"], language="markdown")
    else:
        st.info("정보 부족 케이스이므로 LLM Prompt를 생성하지 않았습니다.")

    st.subheader("LLM 응답")
    st.markdown(final_state.get("llm_response", ""))

    st.subheader("Trace Markdown 결과")
    st.markdown(trace_report)

    relative_output = result["output_path"].relative_to(result["project_root"])
    st.success(f"Trace 파일 저장 완료: {relative_output}")

    st.download_button(
        label="Trace Markdown 다운로드",
        data=trace_report,
        file_name="mini_graph_trace.md",
        mime="text/markdown",
    )


# 이 아래 부분은 이 파일을 직접 실행했을 때만 동작합니다.
# Streamlit 앱은 보통 터미널에서 다음 명령으로 실행합니다.
#   streamlit run src/day1/mini_graph_streamlit_app.py
# 초보자 관점에서는 "이 화면의 시작 버튼"이라고 이해하면 됩니다.
if __name__ == "__main__":
    main()
