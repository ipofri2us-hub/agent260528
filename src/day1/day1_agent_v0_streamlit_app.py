"""
삼성디스플레이 재직자 대상 AI Agent Architecture 1일차 7교시 Agent v0 Streamlit 실습 화면입니다.

역할:
- Day1 Agent v0 CLI 단순 버전과 최대한 비슷한 LangGraph 실행 구조를 유지합니다.
- Streamlit 화면에서 State, Node, Edge, Conditional Edge 흐름을 확인합니다.
- Prompt Mustache 템플릿으로 생성된 LLM Prompt를 보여줍니다.
- 수강생이 LLM 서버로 보내기 전에 Prompt를 직접 수정할 수 있게 합니다.
- 수정된 Prompt로 LLM 응답 변화를 확인할 수 있게 합니다.
- Result Markdown과 Trace Markdown을 Mustache 템플릿으로 생성하고 저장합니다.

중요:
- 이 파일은 Gemini/OpenAI/Anthropic/Ollama/NVIDIA SDK를 직접 import하지 않습니다.
- LLM 호출은 llm_client.py의 generate_response(prompt)를 통해서만 수행합니다.
- API Key, Authorization, Bearer token, password, secret, .env 전체 내용은 화면에 표시하지 않습니다.
- os.environ 전체를 출력하지 않습니다.
- 실제 사내 데이터가 아니라 DisplayEdu Fab 교육용 가상 데이터만 사용합니다.
"""

# ============================================================
# 파일명: day1_agent_v0_streamlit_app.py
# 목적:
#   day1_agent_v0_template.py(Agent v0)의 흐름을 웹 화면(Streamlit)에서 보여 주는 실습 파일입니다.
#   이 화면의 핵심은 "LLM에 보내기 전에 Prompt를 사람이 직접 확인·수정"할 수 있다는 점입니다.
#
# Agent란?(초보자 설명):
#   상황을 보고 다음 행동을 스스로 정하며 일을 처리하는 프로그램 흐름입니다.
#   여기서는 입력을 확인하고, 정보가 충분한지 판단해
#   충분하면 로그·매뉴얼을 조사하고, 부족하면 추가 정보를 요청합니다.
#
# 이 화면이 CLI 버전과 다른 점:
#   그래프를 끝까지 한 번에 돌리지 않고 "Prompt 생성 단계"에서 잠깐 멈춥니다.
#   (1단계) 버튼으로 Prompt까지 만들고 →
#   (2단계) 수강생이 Prompt를 고친 뒤 버튼으로 LLM을 호출합니다.
#
# 이 파일에서 배우는 것:
#   1. State / Node / Conditional Edge로 Agent 판단 흐름을 화면에서 따라가기
#   2. LLM 호출 직전 Prompt를 사람이 검토·수정하는 "사람 개입(Human-in-the-loop)" 구조
#   3. 실행된 Node 순서와 Trace, 최종 보고서를 화면에서 확인하고 저장하기
#
# 전체 실행 흐름:
#   1. 화면 안내와 사용 파일을 보여 주고, 실행 케이스(정상/정보 부족)를 고릅니다.
#   2. (1단계 버튼) 그래프를 Prompt 생성 단계까지 실행합니다.
#   3. 정보가 충분하면 Prompt를 화면에서 수정한 뒤 (2단계 버튼)으로 LLM을 호출합니다.
#      정보가 부족하면 LLM 없이 "추가 정보 요청" 결과를 보여 줍니다.
#   4. 최종 결과(Result)와 실행 기록(Trace)을 화면에 표시하고 파일로 저장합니다.
#
# 초보자를 위한 비유:
#   신입 담당자(Agent)가 보고서를 쓰기 직전,
#   "이대로 LLM에 물어봐도 될까요?" 하고 초안(Prompt)을 먼저 보여 주면
#   사람이 검토·수정한 뒤 진행시키는 업무 방식과 같습니다.
#
# 실행 방법:
#   streamlit run src/day1/day1_agent_v0_streamlit_app.py
# ============================================================

from pathlib import Path
import json
import sys

import pandas as pd
import pystache
import streamlit as st
import streamlit.components.v1 as components
# LangGraph 도구: StateGraph(그래프 본체), START/END(시작·끝 표시)
from langgraph.graph import END, START, StateGraph


# ------------------------------------------------------------
# 함수명: run_day1_agent_v0_streamlit
# 역할:
#   Day1 Agent v0의 LangGraph 흐름을 화면에 그리고,
#   버튼 클릭에 따라 (1단계) Prompt 생성과 (2단계) LLM 호출을 나눠 실행합니다.
#
# 입력값:
#   없음 (필요한 데이터는 함수 안에서 파일로 읽습니다.)
#
# 출력값:
#   반환값은 없습니다.
#   화면에 흐름/Prompt/응답/보고서를 표시하고, 결과를 파일로 저장합니다.
#
# 초보자 설명:
#   CLI 버전과 Agent 실행 구조는 같지만,
#   LLM에 보내기 전 Prompt를 사람이 확인·수정할 수 있도록 중간에 멈추는 점이 다릅니다.
# ------------------------------------------------------------
def run_day1_agent_v0_streamlit():
    """
    Day1 Agent v0 Streamlit 화면을 실행합니다.

    CLI 버전 day1_agent_v0_simple_20260523_124305.py와 최대한 비슷한 구조를 유지합니다.
    차이점은 print와 파일 확인 중심이 아니라 Streamlit 화면에서
    Prompt, Trace, Result를 보여준다는 점입니다.

    Streamlit 버전에서는 수강생이 LLM 서버로 보내기 전에 Prompt를 수정할 수 있도록
    1단계는 Prompt 생성까지만 실행하고, 2단계에서 수정된 Prompt로 LLM을 호출합니다.
    """
    st.set_page_config(
        page_title="Day1 Agent v0 실습",
        layout="wide",
    )

    st.title("1일차 Agent v0 Streamlit 실습 화면")
    st.caption("LangGraph 실행 흐름, Prompt, LLM 응답, Result/Trace Markdown을 화면에서 확인하는 교육용 예제입니다.")

    st.subheader("실행 방법")

    col_cli, col_ui = st.columns(2)

    with col_cli:
        st.markdown("**CLI 버전 실행**")
        st.code(
            "python src/day1/day1_agent_v0_simple_20260523_124305.py",
            language="powershell",
        )
        st.caption("터미널에서 Agent v0 흐름을 실행하는 기본 실습입니다.")

    with col_ui:
        st.markdown("**Streamlit 버전 실행**")
        st.code(
            "streamlit run src/day1/day1_agent_v0_streamlit_app.py",
            language="powershell",
        )
        st.caption("화면에서 Graph 흐름, Prompt, Trace, 결과를 확인하는 실습입니다.")

    st.markdown(
        """
이 화면은 Day1 Agent v0의 LangGraph 실행 흐름을 보여줍니다.

- Agent는 로그를 조회하고, 매뉴얼을 검색하고, Prompt를 만든 뒤 LLM을 호출합니다.
- 이 Streamlit 버전에서는 **LLM 서버로 보내기 전에 Prompt를 직접 수정**할 수 있습니다.
- Node 실행 Trace를 통해 State가 어떤 Node를 거쳐 이동했는지 확인할 수 있습니다.
- CLI 버전과 Agent 실행 구조는 최대한 동일하게 유지했습니다.
        """
    )

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

    st.sidebar.header("실행 안내")
    st.sidebar.code(
        "streamlit run src/day1/day1_agent_v0_streamlit_app.py",
        language="powershell",
    )

    st.sidebar.header("사용 파일")

    file_paths = [
        "data/sample_query.json",
        "data/sample_alarm_logs.csv",
        "docs/alarm_manual.md",
        "templates/day1/day1_agent_v0_prompt.mustache",
        "templates/day1/day1_agent_v0_result.mustache",
        "templates/day1/day1_agent_v0_trace.mustache",
        "outputs/day1/day1_agent_v0_result.md",
        "outputs/day1/day1_agent_v0_trace.md",
    ]

    for relative_path in file_paths:
        st.sidebar.markdown(f"- `{relative_path}`")

    st.sidebar.header("주의 사항")
    st.sidebar.warning(
        "API Key, .env 내용, Authorization token, password, secret 값은 화면에 표시하지 않습니다."
    )

    st.info(f"프로젝트 루트: {project_root}")

    st.warning(
        "Prompt에는 API Key, 비밀번호, 실제 사내 데이터, 실제 설비명, 실제 고객정보를 입력하지 마세요. "
        "이 실습은 교육용 가상 데이터만 사용합니다."
    )

    st.subheader("사용 파일 내용 확인")

    for relative_path in file_paths:
        file_path = project_root / relative_path

        with st.expander(relative_path):
            if file_path.exists():
                if file_path.suffix == ".csv":
                    df = pd.read_csv(file_path, encoding="utf-8-sig")
                    st.dataframe(df, use_container_width=True)
                else:
                    text = file_path.read_text(encoding="utf-8-sig")
                    st.code(text[:5000])
            else:
                st.warning(f"파일을 찾지 못했습니다: {relative_path}")

    try:
        query_data = json.loads(query_path.read_text(encoding="utf-8-sig"))
        logs = pd.read_csv(csv_path, encoding="utf-8-sig")
        manual_text = manual_path.read_text(encoding="utf-8-sig")
    except Exception as exc:
        st.error("입력 파일을 읽는 중 오류가 발생했습니다.")
        st.code(str(exc)[:1000])
        return

    case_type = st.radio(
        "실행 케이스 선택",
        ["정상 케이스", "정보 부족 케이스"],
        horizontal=True,
    )

    # 실행 케이스가 바뀌면 이전 결과를 지워서 화면 혼동을 줄입니다.
    if st.session_state.get("last_case_type") != case_type:
        st.session_state["agent_result"] = None
        st.session_state["llm_result"] = None
        st.session_state["last_case_type"] = case_type

    # 4. 초기 State를 만듭니다.
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

    if case_type == "정보 부족 케이스":
        state["equipment_id"] = ""
        state["user_query"] = (
            "ALM-TEMP-402 교육용 알람이 반복 발생한 것 같습니다. "
            "원인 확인을 위해 어떤 정보가 더 필요한가요?"
        )

    st.subheader("실습 조건")

    col1, col2, col3 = st.columns(3)
    col1.metric("설비 ID", state["equipment_id"] or "없음")
    col2.metric("알람 코드", state["alarm_code"] or "없음")
    col3.metric("실행 케이스", case_type)

    st.write("사용자 요청")
    st.info(state["user_query"])

    graph_flow_text = """
START
 → load_input_node
 → check_required_info_node
 → 조건부 분기
    - 필수 정보 있음:
      search_log_node
      → search_manual_node
      → build_prompt_node
      → LLM 호출 전 사용자 Prompt 수정
      → generate_response_node 역할을 Streamlit 버튼에서 수행
      → build_final_report_node 역할을 Streamlit 버튼에서 수행
      → END

    - 필수 정보 부족:
      ask_more_info_node
      → END
"""

    st.subheader("LangGraph 실행 흐름")
    st.code(graph_flow_text, language="text")

    

    # 5. Trace를 추가하는 내부 함수입니다.
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

    # 아래 6-1 ~ 6-7 함수들이 각각 하나의 Node(처리 단계)입니다.
    # 공통 규칙: 모든 Node는 state(기록지)를 받아 내용을 채운 뒤 다시 state를 돌려줍니다.

    # 6-1. 입력값을 확인하는 Node입니다.
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

    # 6-2. 필수 정보가 있는지 확인하는 Node입니다.
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

    # 6-3. Conditional Edge에서 사용할 분기 함수입니다.
    def route_after_required_info_check(state):
        if state.get("next_action") == "investigate":
            return "search_log_node"
        return "ask_more_info_node"

    # 6-4. 로그를 조회하는 Node입니다.
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

    # 6-5. 매뉴얼을 검색하는 Node입니다.
    def search_manual_node(state):
        print("search_manual_node 실행")

        alarm_code = state.get("alarm_code", "")
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

    # 6-6. Prompt Mustache 템플릿으로 LLM Prompt를 만드는 Node입니다.
    # Streamlit 버전에서는 여기까지만 Graph를 실행하고, 실제 LLM 호출은 사용자가 Prompt를 수정한 뒤 버튼으로 수행합니다.
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
            "사용자 Prompt 수정 후 LLM 호출",
        )

    # 6-7. 정보가 부족할 때 실행되는 Node입니다.
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

    # 7. LangGraph를 구성합니다.
    # Streamlit 버전에서는 LLM 호출 전 Prompt 수정이 가능하도록 build_prompt_node에서 종료합니다.
    #   - add_node("이름", 함수): 처리 단계 등록 / add_edge(A, B): A 다음 항상 B
    #   - add_conditional_edges(...): 조건에 따라 갈리는 갈림길 / compile(): 실행 형태로 완성
    # 그래서 CLI 버전과 달리 generate_response_node가 그래프 안에 없습니다.
    # (LLM 호출은 아래 2단계 버튼에서 사람이 Prompt를 확인한 뒤에 직접 수행합니다.)
    graph = StateGraph(dict)

    graph.add_node("load_input_node", load_input_node)
    graph.add_node("check_required_info_node", check_required_info_node)
    graph.add_node("search_log_node", search_log_node)
    graph.add_node("search_manual_node", search_manual_node)
    graph.add_node("build_prompt_node", build_prompt_node)
    graph.add_node("ask_more_info_node", ask_more_info_node)

    graph.add_edge(START, "load_input_node")
    graph.add_edge("load_input_node", "check_required_info_node")

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
    graph.add_edge("build_prompt_node", END)
    graph.add_edge("ask_more_info_node", END)

    app = graph.compile()

    if "agent_result" not in st.session_state:
        st.session_state["agent_result"] = None

    if "llm_result" not in st.session_state:
        st.session_state["llm_result"] = None

    # [1단계] 버튼: 그래프를 실행해 Prompt 생성까지만 진행합니다. (아직 LLM은 부르지 않음)
    # app.invoke(state)는 START부터 Node들을 순서대로 실행하고 최종 state를 돌려줍니다.
    if st.button("1단계: Agent v0로 Prompt 생성하기", type="primary"):
        try:
            with st.spinner("Agent v0가 로그와 매뉴얼을 확인하고 Prompt를 생성하는 중입니다..."):
                final_state = app.invoke(state)

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

                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text(trace_report, encoding="utf-8-sig")

                # 정보 부족 케이스는 LLM을 호출하지 않으므로 결과 파일도 바로 저장합니다.
                if final_state.get("final_report"):
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(final_state.get("final_report", ""), encoding="utf-8-sig")

                st.session_state["agent_result"] = {
                    "final_state": final_state,
                    "trace_report": trace_report,
                    "result_path": result_path,
                    "trace_path": trace_path,
                    "project_root": project_root,
                }

                st.session_state["llm_result"] = None

        except Exception as exc:
            st.error("Agent v0 Prompt 생성 중 오류가 발생했습니다.")
            st.code(str(exc)[:1000])

    result = st.session_state["agent_result"]

    if result is not None:
        final_state = result["final_state"]
        trace_report = result["trace_report"]

        st.subheader("Conditional Edge 분기 결과")

        if final_state.get("next_action") == "investigate":
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

        if final_state.get("next_action") != "investigate":
            st.subheader("정보 부족 케이스 결과")
            st.markdown(final_state.get("final_report", ""))

            final_report_for_download = final_state.get("final_report", "")
            llm_result = None

        else:
            st.subheader("LLM 서버로 보내기 전 Prompt")
            st.code(final_state.get("llm_prompt", ""), language="markdown")

            st.subheader("Prompt 수정")
            edited_prompt = st.text_area(
                "LLM에 전달할 Prompt를 수정해 보세요.",
                value=final_state.get("llm_prompt", ""),
                height=550,
            )

            st.caption(
                "수정한 Prompt로 LLM을 실행합니다. 이 단계에서만 LLM 서버로 Prompt가 전송됩니다."
            )

            # [2단계] 버튼: 사람이 확인·수정한 edited_prompt로 실제 LLM을 호출합니다.
            # 이 단계에서만 LLM 서버로 Prompt가 전송됩니다.
            if st.button("2단계: 수정한 Prompt로 LLM 실행하기"):
                try:
                    from llm_client import generate_response

                    with st.spinner("수정한 Prompt로 LLM 응답을 생성하는 중입니다..."):
                        new_response = generate_response(edited_prompt)

                    log_summary_text = json.dumps(
                        final_state.get("log_summary", {}),
                        ensure_ascii=False,
                        indent=2,
                    )

                    result_template = result_template_path.read_text(encoding="utf-8-sig")

                    result_data = {
                        "user_query": final_state.get("user_query", ""),
                        "line_id": final_state.get("line_id", ""),
                        "process_name": final_state.get("process_name", ""),
                        "equipment_id": final_state.get("equipment_id", ""),
                        "alarm_code": final_state.get("alarm_code", ""),
                        "log_summary_text": log_summary_text,
                        "manual_section": final_state.get("manual_section", ""),
                        "llm_response": new_response,
                        "additional_message": "수강생이 수정한 Prompt로 LLM 응답을 생성했습니다.",
                    }

                    new_final_report = pystache.render(result_template, result_data)
                    result["result_path"].parent.mkdir(parents=True, exist_ok=True)
                    result["result_path"].write_text(new_final_report, encoding="utf-8-sig")

                    st.session_state["llm_result"] = {
                        "edited_prompt": edited_prompt,
                        "new_response": new_response,
                        "new_final_report": new_final_report,
                    }

                except Exception as exc:
                    st.error("수정한 Prompt로 LLM을 실행하는 중 오류가 발생했습니다.")
                    st.code(str(exc)[:1000])

            llm_result = st.session_state["llm_result"]

            if llm_result is not None:
                st.subheader("실제 LLM 서버로 보낸 Prompt")
                st.code(llm_result["edited_prompt"], language="markdown")

                st.subheader("LLM 응답")
                st.markdown(llm_result["new_response"])

                st.subheader("최종 Markdown 결과")
                st.markdown(llm_result["new_final_report"])

                final_report_for_download = llm_result["new_final_report"]
            else:
                st.info("아직 LLM을 실행하지 않았습니다. Prompt를 확인하거나 수정한 뒤 2단계 버튼을 눌러 주세요.")
                final_report_for_download = ""

        st.subheader("Trace Markdown 결과")
        st.markdown(trace_report)

        relative_trace = result["trace_path"].relative_to(project_root)
        st.success(f"Trace 파일 저장 완료: {relative_trace}")

        if final_report_for_download:
            relative_result = result["result_path"].relative_to(project_root)
            st.success(f"결과 파일 저장 완료: {relative_result}")

            st.download_button(
                label="결과 Markdown 다운로드",
                data=final_report_for_download,
                file_name="day1_agent_v0_result.md",
                mime="text/markdown",
            )

        st.download_button(
            label="Trace Markdown 다운로드",
            data=trace_report,
            file_name="day1_agent_v0_trace.md",
            mime="text/markdown",
        )

    else:
        st.info("아직 Agent v0를 실행하지 않았습니다. 실행 케이스를 선택한 뒤 버튼을 눌러 주세요.")


def main():
    """
    프로그램 시작점입니다.
    """
    run_day1_agent_v0_streamlit()


# 이 아래 부분은 이 파일을 직접 실행했을 때만 동작합니다.
# Streamlit 앱은 보통 터미널에서 다음 명령으로 실행합니다.
#   streamlit run src/day1/day1_agent_v0_streamlit_app.py
# 초보자 관점에서는 "이 화면의 시작 버튼"이라고 이해하면 됩니다.
if __name__ == "__main__":
    main()
