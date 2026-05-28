"""
삼성디스플레이 재직자 대상 AI Agent Architecture 1일차 5교시 Chain Streamlit 실습 화면입니다.

역할:
- data/sample_query.json에서 사용자 요청을 읽습니다.
- docs/alarm_manual.md에서 알람 코드 관련 내용을 단순 검색합니다.
- templates/day1/simple_chain_prompt.mustache로 기본 LLM Prompt를 생성합니다.
- 수강생이 Streamlit text_area에서 Prompt를 직접 수정할 수 있게 합니다.
- 수정된 Prompt를 llm_client.py의 generate_response(prompt)에 전달합니다.
- templates/day1/simple_chain_result.mustache로 최종 Markdown 결과를 생성합니다.
- outputs/day1/simple_chain_result.md로 저장합니다.

중요:
- 이 파일은 Gemini/OpenAI/Anthropic/Ollama/NVIDIA SDK를 직접 import하지 않습니다.
- LLM 호출은 llm_client.py의 generate_response(prompt)를 통해서만 수행합니다.
- API Key, Authorization, Bearer token, password, secret, .env 전체 내용은 화면에 표시하지 않습니다.
- os.environ 전체를 출력하지 않습니다.
- 실제 사내 데이터가 아니라 DisplayEdu Fab 교육용 가상 데이터만 사용합니다.
"""

# ============================================================
# 파일명: simple_chain_streamlit_app.py
# 목적:
#   simple_chain_starter.py의 Chain 흐름을 웹 화면(Streamlit)으로 보여 주는 실습 파일입니다.
#   이 화면의 특징은 "수강생이 LLM에게 보낼 Prompt를 직접 고쳐 볼 수 있다"는 점입니다.
#
# 이 파일에서 배우는 것:
#   1. 기본 Prompt를 화면 입력칸(text_area)에 띄워 직접 수정하는 방법
#   2. 수정한 Prompt로 LLM을 호출해 답변이 어떻게 달라지는지 비교하는 방법
#   3. "데이터 준비"와 "LLM 호출"을 함수로 나누어, 버튼을 누를 때만 LLM을 호출하는 구조
#
# 전체 실행 흐름:
#   1. 프로젝트 루트를 찾고 사용 파일 내용을 화면에 보여 줍니다.
#   2. prepare_chain_inputs()로 기본 Prompt까지 미리 만들어 둡니다.
#   3. 수강생이 화면에서 Prompt를 수정합니다.
#   4. 버튼을 누르면 run_llm_with_prompt()가 수정된 Prompt로 LLM을 호출합니다.
#   5. LLM 응답과 최종 Markdown 결과를 화면에 보여 주고 저장합니다.
#
# 초보자를 위한 비유:
#   LLM에게 보내는 Prompt(지시문)를 직접 손보면 답이 어떻게 달라지는지
#   실험해 보는 "지시문 연습장"과 같은 화면입니다.
#
# 실행 방법:
#   streamlit run src/day1/simple_chain_streamlit_app.py
# ============================================================

from pathlib import Path
import json
import sys

import pandas as pd
import pystache
import streamlit as st


# 현재 파일 위치를 기준으로 data/docs/src가 함께 있는 프로젝트 루트 폴더를 찾는 함수입니다.
# (실습 폴더를 다른 위치로 옮겨도 파일을 잘 찾도록 돕습니다.)
def find_project_root():
    """
    현재 파일 위치를 기준으로 프로젝트 루트 폴더를 찾습니다.

    프로젝트 폴더가 C 드라이브가 아닌 다른 위치로 이동해도 동작하도록
    data, docs, src 폴더가 함께 있는 위치를 찾습니다.
    """
    current_file = Path(__file__).resolve()

    for parent in [current_file.parent, *current_file.parents]:
        if (parent / "data").exists() and (parent / "docs").exists() and (parent / "src").exists():
            return parent

    # 이 파일이 src/day1 폴더 안에 있다는 전제에서 사용하는 예비 경로입니다.
    return current_file.parents[2]


# ------------------------------------------------------------
# 함수명: prepare_chain_inputs
# 역할:
#   LLM을 호출하기 "직전"까지의 준비 작업을 모아서 합니다.
#   (파일 읽기 → 매뉴얼 단순 검색 → 기본 Prompt 만들기)
#
# 입력값:
#   project_root: 프로젝트 루트 폴더 경로(Path)
#
# 출력값:
#   화면에 표시하거나 다음 단계에서 쓸 값들이 담긴 dict
#   (입력 조건, 매뉴얼 검색 결과, 기본 Prompt 등)
#
# 초보자 설명:
#   LLM 호출은 여기서 하지 않습니다.
#   "수강생이 Prompt를 고친 뒤 버튼을 눌렀을 때"만 LLM을 호출하기 위해
#   준비 단계와 호출 단계를 일부러 나눠 둔 것입니다.
# ------------------------------------------------------------
def prepare_chain_inputs(project_root):
    """
    LLM 호출 전까지의 Chain 입력 데이터를 준비합니다.

    이 함수에서는 다음 작업만 수행합니다.
    - sample_query.json 읽기
    - alarm_manual.md 읽기
    - 알람 코드 관련 매뉴얼 일부 찾기
    - Prompt Mustache 템플릿으로 기본 Prompt 만들기

    LLM 호출은 main()의 버튼 클릭 이후에 수행합니다.
    """
    # src/llm_client.py를 import하기 위해 src 폴더를 import 경로에 추가합니다.
    src_dir = project_root / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # 이번 Chain 실습에서 사용할 파일 경로입니다.
    query_path = project_root / "data" / "sample_query.json"
    manual_path = project_root / "docs" / "alarm_manual.md"

    prompt_template_path = project_root / "templates" / "day1" / "simple_chain_prompt.mustache"
    result_template_path = project_root / "templates" / "day1" / "simple_chain_result.mustache"

    output_path = project_root / "outputs" / "day1" / "simple_chain_result.md"

    # 필수 파일이 없으면 수강생이 바로 알 수 있게 쉬운 메시지로 알려줍니다.
    required_files = [
        query_path,
        manual_path,
        prompt_template_path,
        result_template_path,
    ]

    for file_path in required_files:
        if not file_path.exists():
            relative_path = file_path.relative_to(project_root)
            raise FileNotFoundError(f"필수 파일을 찾지 못했습니다: {relative_path}")

    # 사용자 요청을 읽습니다.
    query_data = json.loads(query_path.read_text(encoding="utf-8-sig"))

    scenario_name = str(
        query_data.get(
            "scenario_name",
            "DisplayEdu Fab 교육용 챔버 온도 편차 반복 알람 분석 시나리오",
        )
    )
    user_query = str(query_data["user_query"])
    line_id = str(query_data["line_id"])
    process_name = str(query_data["process_name"])
    equipment_id = str(query_data["equipment_id"])
    alarm_code = str(query_data["alarm_code"])

    # 알람 매뉴얼을 읽습니다.
    manual_text = manual_path.read_text(encoding="utf-8-sig")

    # 실제 RAG 검색은 2일차에서 다룹니다.
    # 여기서는 Chain 흐름 이해가 목적이므로 알람 코드와 주요 키워드가 포함된 줄만 단순 검색합니다.
    manual_lines = manual_text.splitlines()
    found_lines = []

    for line in manual_lines:
        if alarm_code in line or "온도" in line or "확인" in line or "조치" in line:
            found_lines.append(line)

    manual_section = "\n".join(found_lines[:40])

    if not manual_section.strip():
        manual_section = manual_text[:2000]

    # Prompt Mustache 템플릿을 읽고 기본 Prompt를 만듭니다.
    prompt_template = prompt_template_path.read_text(encoding="utf-8-sig")

    prompt_data = {
        "user_query": user_query,
        "line_id": line_id,
        "process_name": process_name,
        "equipment_id": equipment_id,
        "alarm_code": alarm_code,
        "manual_section": manual_section,
    }

    default_prompt = pystache.render(prompt_template, prompt_data)

    return {
        "project_root": project_root,
        "query_path": query_path,
        "manual_path": manual_path,
        "prompt_template_path": prompt_template_path,
        "result_template_path": result_template_path,
        "output_path": output_path,
        "query_data": query_data,
        "scenario_name": scenario_name,
        "user_query": user_query,
        "line_id": line_id,
        "process_name": process_name,
        "equipment_id": equipment_id,
        "alarm_code": alarm_code,
        "manual_section": manual_section,
        "default_prompt": default_prompt,
    }


# ------------------------------------------------------------
# 함수명: run_llm_with_prompt
# 역할:
#   수강생이 수정한 Prompt로 실제 LLM을 호출하고,
#   결과를 Markdown 보고서로 만들어 파일로 저장합니다.
#
# 입력값:
#   chain_data: prepare_chain_inputs()가 준비해 둔 값 모음(dict)
#   edited_prompt: 수강생이 화면에서 수정한 최종 Prompt 문자열
#
# 출력값:
#   화면 표시에 쓸 값들이 담긴 dict (실제 보낸 Prompt, LLM 응답, 보고서 등)
#
# 초보자 설명:
#   여기서 edited_prompt(수정한 지시문)가 그대로 LLM에게 전달됩니다.
#   그래서 지시문을 바꾸면 답변도 달라지는 것을 직접 확인할 수 있습니다.
# ------------------------------------------------------------
def run_llm_with_prompt(chain_data, edited_prompt):
    """
    수강생이 수정한 Prompt로 LLM을 호출하고,
    Result Mustache 템플릿으로 Markdown 결과를 생성한 뒤 저장합니다.

    중요:
    - edited_prompt를 실제 LLM 호출에 사용합니다.
    - Result Markdown에도 edited_prompt를 저장합니다.
    - provider별 SDK를 직접 import하지 않고 llm_client.py만 호출합니다.
    """
    from llm_client import generate_response

    response = generate_response(edited_prompt)

    result_template = chain_data["result_template_path"].read_text(encoding="utf-8-sig")

    result_data = {
        "scenario_name": chain_data["scenario_name"],
        "user_query": chain_data["user_query"],
        "line_id": chain_data["line_id"],
        "process_name": chain_data["process_name"],
        "equipment_id": chain_data["equipment_id"],
        "alarm_code": chain_data["alarm_code"],
        "manual_section": chain_data["manual_section"],
        "prompt": edited_prompt,
        "response": response,
    }

    result_markdown = pystache.render(result_template, result_data)

    output_path = chain_data["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result_markdown, encoding="utf-8-sig")

    return {
        "edited_prompt": edited_prompt,
        "response": response,
        "result_markdown": result_markdown,
        "output_path": output_path,
        "project_root": chain_data["project_root"],
    }


def main():
    """
    Streamlit 화면 시작점입니다.
    """
    st.set_page_config(
        page_title="Day1 Chain Prompt 실습",
        layout="wide",
    )

    st.title("1일차 Chain Prompt 실습 화면")
    st.caption("Prompt를 직접 수정하고 LLM 응답 변화를 확인하는 Streamlit 예제입니다.")

    st.markdown(
        """
이 화면은 `simple_chain_starter_simple` 예제의 Chain 흐름을 Streamlit으로 보여줍니다.

- **Python 코드**는 사용자 요청 읽기, 매뉴얼 검색, LLM 호출, 결과 저장을 담당합니다.
- **Prompt Mustache**는 LLM에게 전달할 기본 지시문을 만듭니다.
- 수강생은 **text_area**에서 Prompt를 직접 수정할 수 있습니다.
- 수정된 Prompt로 LLM을 호출하면 응답이 어떻게 달라지는지 확인할 수 있습니다.
        """
    )

    st.sidebar.header("실행 안내")
    st.sidebar.code(
        "streamlit run src/day1/simple_chain_streamlit_app_20260523_205151.py",
        language="powershell",
    )

    st.sidebar.header("사용 파일")

    file_paths = [
        "data/sample_query.json",
        "docs/alarm_manual.md",
        "templates/day1/simple_chain_prompt.mustache",
        "templates/day1/simple_chain_result.mustache",
        "outputs/day1/simple_chain_result.md",
    ]

    for relative_path in file_paths:
        st.sidebar.markdown(f"- `{relative_path}`")

    st.sidebar.header("주의 사항")
    st.sidebar.warning(
        "API Key, .env 내용, Authorization token, password, secret 값은 화면에 표시하지 않습니다."
    )

    project_root = find_project_root()
    st.info(f"프로젝트 루트: {project_root}")

    try:
        chain_data = prepare_chain_inputs(project_root)
    except Exception as exc:
        st.error("Chain 입력 데이터를 준비하는 중 오류가 발생했습니다.")
        st.code(str(exc)[:1000])
        return

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

    st.subheader("1. 실습 조건")

    col1, col2, col3 = st.columns(3)
    col1.metric("설비 ID", chain_data["equipment_id"])
    col2.metric("알람 코드", chain_data["alarm_code"])
    col3.metric("공정", chain_data["process_name"])

    st.write("사용자 요청")
    st.info(chain_data["user_query"])

    st.subheader("2. 참고 매뉴얼 내용")
    st.code(chain_data["manual_section"], language="markdown")

    st.subheader("3. LLM에 전달할 Prompt 수정")

    # st.text_area는 여러 줄을 입력/수정할 수 있는 큰 입력칸입니다.
    # value에 기본 Prompt를 넣어 두면, 수강생이 그 내용을 자유롭게 고칠 수 있습니다.
    # 사용자가 고친 최종 내용이 edited_prompt 변수에 담깁니다.
    edited_prompt = st.text_area(
        "아래 Prompt를 수정한 뒤 LLM을 실행해 보세요.",
        value=chain_data["default_prompt"],
        height=550,
    )

    st.caption(
        "기본 Prompt는 simple_chain_prompt.mustache와 입력 데이터를 결합해 생성되었습니다. "
        "수정된 Prompt가 실제 LLM 호출에 사용됩니다."
    )

    if "chain_result" not in st.session_state:
        st.session_state["chain_result"] = None

    if st.button("수정한 Prompt로 LLM 실행하기", type="primary"):
        try:
            with st.spinner("수정한 Prompt로 LLM 응답을 생성하는 중입니다..."):
                st.session_state["chain_result"] = run_llm_with_prompt(
                    chain_data=chain_data,
                    edited_prompt=edited_prompt,
                )

        except Exception as exc:
            st.error("LLM 실행 또는 결과 저장 중 오류가 발생했습니다.")
            st.code(str(exc)[:1000])

    result = st.session_state["chain_result"]

    if result is not None:
        st.subheader("4. LLM에 실제 전달한 Prompt")
        st.code(result["edited_prompt"], language="markdown")

        st.subheader("5. LLM 응답")
        st.markdown(result["response"])

        st.subheader("6. 최종 Markdown 결과")
        st.markdown(result["result_markdown"])

        relative_output = result["output_path"].relative_to(result["project_root"])
        st.success(f"결과 파일 저장 완료: {relative_output}")

        st.download_button(
            label="Markdown 결과 다운로드",
            data=result["result_markdown"],
            file_name="simple_chain_result.md",
            mime="text/markdown",
        )
    else:
        st.info("아직 LLM을 실행하지 않았습니다. Prompt를 확인하거나 수정한 뒤 버튼을 눌러 주세요.")


# 이 아래 부분은 이 파일을 직접 실행했을 때만 동작합니다.
# Streamlit 앱은 보통 터미널에서 다음 명령으로 실행합니다.
#   streamlit run src/day1/simple_chain_streamlit_app.py
# 초보자 관점에서는 "이 화면의 시작 버튼"이라고 이해하면 됩니다.
if __name__ == "__main__":
    main()
