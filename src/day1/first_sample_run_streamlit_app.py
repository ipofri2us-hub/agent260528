"""
삼성디스플레이 재직자 대상 AI Agent Architecture 1일차 Streamlit 실습 화면입니다.

역할:
- data/sample_query.json에서 실습 입력 조건을 읽습니다.
- data/sample_alarm_logs.csv에서 설비 ID와 알람 코드 기준으로 로그를 필터링합니다.
- docs/alarm_manual.md의 앞부분 일부를 참고 자료로 사용합니다.
- templates/day1/first_sample_run_prompt.mustache로 LLM Prompt를 만듭니다.
- llm_client.py의 generate_response(prompt)를 통해 LLM을 호출합니다.
- templates/day1/first_sample_run_result.mustache로 Markdown 결과를 만듭니다.
- outputs/day1/first_sample_run_result.md로 저장합니다.
- 위 과정을 Streamlit 화면에서 확인할 수 있게 보여줍니다.

중요:
- 이 파일은 Gemini/OpenAI/Anthropic/Ollama/NVIDIA SDK를 직접 import하지 않습니다.
- LLM 호출은 llm_client.py의 generate_response(prompt)를 통해서만 수행합니다.
- API Key, Authorization, Bearer token, password, secret, .env 전체 내용은 화면에 표시하지 않습니다.
- os.environ 전체를 출력하지 않습니다.
- 실제 사내 데이터가 아니라 DisplayEdu Fab 교육용 가상 데이터만 사용합니다.
"""

# ============================================================
# 파일명: first_sample_run_streamlit_app.py
# 목적:
#   first_sample_run.py와 같은 "첫 Agent 실행" 흐름을,
#   터미널 출력 대신 웹 화면(Streamlit)에서 눈으로 보며 확인하는 실습 파일입니다.
#
# Streamlit이란?(초보자 설명):
#   파이썬 코드만으로 간단한 웹 화면을 만들어 주는 도구입니다.
#   st.title, st.button, st.dataframe 처럼 st.~ 로 시작하는 명령이 화면 요소가 됩니다.
#   화면에서 버튼을 누르면 연결된 파이썬 코드가 실행됩니다.
#
# 이 파일에서 배우는 것:
#   1. 버튼 클릭으로 Agent 흐름을 실행하고 결과를 화면에 표시하는 방법
#   2. 입력 파일 / Prompt / LLM 응답 / 최종 보고서를 화면 단계별로 보여 주는 방법
#   3. CLI 버전(first_sample_run.py)과 같은 로직을 화면용으로 감싸는 구조
#
# 전체 실행 흐름:
#   1. 화면 제목과 안내를 그립니다.
#   2. 프로젝트 루트를 찾고, 사용 파일 내용을 화면에 보여 줍니다.
#   3. "Agent 실행하기" 버튼을 누르면 run_agent_once()가 한 번 실행됩니다.
#   4. 실행 결과(로그 요약 / Prompt / LLM 응답 / 보고서)를 단계별로 화면에 표시합니다.
#
# 초보자를 위한 비유:
#   CLI 버전이 "혼자 조용히 일하고 결과 파일만 남기는 직원"이라면,
#   이 화면은 "일하는 과정을 화면에 중계해서 보여 주는 직원"과 같습니다.
#
# 실행 방법:
#   streamlit run src/day1/first_sample_run_streamlit_app.py
# ============================================================

from pathlib import Path
import json
import sys

import pandas as pd
import pystache
import streamlit as st


# ------------------------------------------------------------
# 함수명: find_project_root
# 역할:
#   data / docs / src 폴더가 함께 들어 있는 위치를 위로 거슬러 올라가며 찾아
#   "프로젝트 루트 폴더"로 정합니다.
#
# 입력값:
#   없음
#
# 출력값:
#   프로젝트 루트 폴더 경로(Path)
#
# 초보자 설명:
#   실습 폴더를 다른 위치로 옮겨도 파일을 잘 찾도록 도와주는 함수입니다.
#   "공통으로 들어 있는 폴더 묶음을 기준으로 집을 찾는다"고 생각하면 됩니다.
# ------------------------------------------------------------
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

    return current_file.parents[2]


# ------------------------------------------------------------
# 함수명: run_agent_once
# 역할:
#   버튼을 눌렀을 때 Agent 흐름(데이터 읽기 → 필터/요약 → Prompt 생성 →
#   LLM 호출 → 결과 저장)을 한 번 실행하고, 화면에 보여 줄 값들을 모아 돌려줍니다.
#
# 입력값:
#   project_root: 프로젝트 루트 폴더 경로(Path)
#
# 출력값:
#   화면 표시에 필요한 값들이 담긴 dict
#   (설비 ID, 알람 코드, 로그 요약, Prompt, LLM 응답, 보고서 등)
#
# 초보자 설명:
#   CLI 버전의 run_first_sample()과 같은 순서로 동작합니다.
#   다른 점은 결과를 print가 아니라 dict로 돌려주어 화면에 표시한다는 것입니다.
# ------------------------------------------------------------
def run_agent_once(project_root):
    """
    Streamlit 버튼을 눌렀을 때 Agent 흐름을 한 번 실행합니다.

    first_sample_run_simple의 run_first_sample()과 같은 순서로 실행합니다.
    """
    # 1. src 폴더를 import 경로에 추가합니다.
    src_dir = project_root / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # 2. 이번 예제에서 사용할 파일 경로를 준비합니다.
    query_path = project_root / "data" / "sample_query.json"
    log_path = project_root / "data" / "sample_alarm_logs.csv"
    manual_path = project_root / "docs" / "alarm_manual.md"

    prompt_template_path = project_root / "templates" / "day1" / "first_sample_run_prompt.mustache"
    result_template_path = project_root / "templates" / "day1" / "first_sample_run_result.mustache"

    output_path = project_root / "outputs" / "day1" / "first_sample_run_result.md"

    # 3. 필수 파일이 없으면 수강생이 바로 알 수 있게 쉬운 메시지로 알려줍니다.
    required_files = [
        query_path,
        log_path,
        manual_path,
        prompt_template_path,
        result_template_path,
    ]

    for file_path in required_files:
        if not file_path.exists():
            relative_path = file_path.relative_to(project_root)
            raise FileNotFoundError(f"필수 파일을 찾지 못했습니다: {relative_path}")

    # 4. 사용자 요청 조건을 읽습니다.
    query_data = json.loads(query_path.read_text(encoding="utf-8-sig"))

    equipment_id = str(query_data["equipment_id"])
    alarm_code = str(query_data["alarm_code"])
    user_query = str(query_data["user_query"])

    # 5. 제조 알람 로그 CSV를 읽습니다.
    logs = pd.read_csv(log_path, encoding="utf-8-sig")

    # 6. 설비 ID와 알람 코드가 일치하는 로그만 찾습니다.
    filtered_logs = logs[
        (logs["equipment_id"].astype(str) == equipment_id)
        & (logs["alarm_code"].astype(str) == alarm_code)
    ].copy()

    if "timestamp" in filtered_logs.columns:
        filtered_logs = filtered_logs.sort_values("timestamp")

    # 7. 필터링된 로그를 간단히 요약합니다.
    total_count = len(filtered_logs)

    if total_count > 0:
        first_time = filtered_logs["timestamp"].iloc[0]
        last_time = filtered_logs["timestamp"].iloc[-1]
        severity_counts = filtered_logs["severity"].value_counts().to_dict()
    else:
        first_time = "해당 없음"
        last_time = "해당 없음"
        severity_counts = {}

    # 8. 알람 매뉴얼을 읽습니다.
    # 실제 RAG 검색은 2일차에서 더 자세히 다룹니다.
    # 여기서는 첫 실행 흐름을 단순하게 보기 위해 매뉴얼 앞부분 일부만 사용합니다.
    manual_text = manual_path.read_text(encoding="utf-8-sig")
    manual_preview = manual_text[:2000]

    # 9. 관련 로그 일부를 LLM이 읽기 쉬운 JSON 문자열로 바꿉니다.
    logs_json = filtered_logs.head(10).to_json(
        orient="records",
        force_ascii=False,
        indent=2,
    )

    # 10. Prompt Mustache 템플릿을 읽습니다.
    prompt_template = prompt_template_path.read_text(encoding="utf-8-sig")

    # 11. Prompt 템플릿에 넣을 데이터를 준비합니다.
    prompt_data = {
        "user_query": user_query,
        "equipment_id": equipment_id,
        "alarm_code": alarm_code,
        "total_count": total_count,
        "first_time": str(first_time),
        "last_time": str(last_time),
        "severity_counts": severity_counts,
        "logs_json": logs_json,
        "manual_preview": manual_preview,
    }

    # 12. Mustache 템플릿으로 LLM에게 보낼 Prompt를 만듭니다.
    prompt = pystache.render(prompt_template, prompt_data)

    # 13. llm_client.py를 통해 LLM을 호출합니다.
    # 이 파일에서는 provider별 SDK를 직접 import하지 않습니다.
    # API Key나 .env 내용도 화면에 표시하지 않습니다.
    from llm_client import generate_response

    llm_response = generate_response(prompt)

    # 14. Result Mustache 템플릿을 읽습니다.
    result_template = result_template_path.read_text(encoding="utf-8-sig")

    # 15. Result 템플릿에 넣을 데이터를 준비합니다.
    result_data = {
        "equipment_id": equipment_id,
        "alarm_code": alarm_code,
        "total_count": total_count,
        "first_time": str(first_time),
        "last_time": str(last_time),
        "severity_counts": severity_counts,
        "prompt": prompt,
        "llm_response": llm_response,
    }

    # 16. Mustache 템플릿으로 Markdown 결과를 만듭니다.
    report = pystache.render(result_template, result_data)

    # 17. 결과 Markdown 파일을 저장합니다.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8-sig")

    return {
        "project_root": project_root,
        "query_data": query_data,
        "equipment_id": equipment_id,
        "alarm_code": alarm_code,
        "user_query": user_query,
        "total_count": total_count,
        "first_time": str(first_time),
        "last_time": str(last_time),
        "severity_counts": severity_counts,
        "filtered_logs": filtered_logs,
        "manual_preview": manual_preview,
        "prompt": prompt,
        "llm_response": llm_response,
        "report": report,
        "output_path": output_path,
    }


def main():
    """
    Streamlit 화면 시작점입니다.
    """

    def show_used_file_expanders(project_root, file_paths):
        """
        사이드바에서 링크처럼 표시한 파일들의 내용을 본문 expander에서 보여줍니다.
        """
        st.subheader("사용 파일 내용 확인")
        st.caption("사이드바의 파일 경로를 클릭하면 이 영역의 해당 파일 위치로 이동합니다.")

        for index, relative_path in enumerate(file_paths, start=1):
            anchor_id = f"file-{index}"
            file_path = project_root / relative_path

            st.markdown(f'<a id="{anchor_id}"></a>', unsafe_allow_html=True)

            with st.expander(f"{index}. {relative_path}", expanded=False):
                st.code(str(file_path), language="text")

                if not file_path.exists():
                    st.warning(f"파일을 찾지 못했습니다: {relative_path}")
                    continue

                if file_path.suffix.lower() == ".csv":
                    df = pd.read_csv(file_path, encoding="utf-8-sig")
                    st.dataframe(df, use_container_width=True)
                else:
                    file_text = file_path.read_text(encoding="utf-8-sig")
                    st.code(file_text[:5000], language="markdown")

                    if len(file_text) > 5000:
                        st.caption("파일 내용이 길어 앞 5,000자만 표시했습니다.")

                st.download_button(
                    label=f"다운로드: {file_path.name}",
                    data=file_path.read_bytes(),
                    file_name=file_path.name,
                    mime="text/plain",
                    key=f"download_{index}_{file_path.name}",
                )

    st.set_page_config(
        page_title="Day1 첫 Agent 실행",
        layout="wide",
    )

    st.title("1일차 첫 Agent 실행 화면")
    st.caption("Prompt와 생성된 Markdown 결과를 화면에서 확인하는 교육용 Streamlit 예제입니다.")

    current_file = Path(__file__).resolve()
    streamlit_command = f"streamlit run src/day1/{current_file.name}"

    st.markdown(
        """
이 화면은 `first_sample_run_simple` 예제와 같은 흐름을 Streamlit으로 보여줍니다.

- **Python 코드**: 데이터 읽기, 로그 필터링, LLM 호출, 결과 저장을 담당합니다.
- **Prompt Mustache**: LLM에게 전달할 지시문을 담당합니다.
- **Result Mustache**: 최종 Markdown 결과 문서 구조를 담당합니다.
- **Streamlit**: Agent 내부 실행 과정을 화면에 보여주는 관찰 화면입니다.
        """
    )

    st.info(
        "이 화면은 Prompt를 직접 수정하는 화면이 아닙니다.\n\n"
        "Agent가 입력 파일을 읽고, 정해진 Prompt Mustache 템플릿으로 Prompt를 생성한 뒤, "
        "LLM에 전달하는 전체 흐름을 확인하는 첫 실행 화면입니다.\n\n"
        "Prompt를 직접 수정하는 실습은 이후 Chain 또는 Agent v0 Streamlit 예제에서 진행합니다."
    )

    st.sidebar.header("실행 안내")
    st.sidebar.code(
        streamlit_command,
        language="powershell",
    )

    st.sidebar.markdown("### 사용 파일")

    file_paths = [
        "data/sample_query.json",
        "data/sample_alarm_logs.csv",
        "docs/alarm_manual.md",
        "templates/day1/first_sample_run_prompt.mustache",
        "templates/day1/first_sample_run_result.mustache",
        "outputs/day1/first_sample_run_result.md",
    ]

    for index, relative_path in enumerate(file_paths, start=1):
        st.sidebar.markdown(f"- [{relative_path}](#file-{index})")

    st.sidebar.header("주의 사항")
    st.sidebar.warning("API Key, .env 내용, Authorization token, password, secret 값은 화면에 표시하지 않습니다.")

    project_root = find_project_root()
    st.info(f"프로젝트 루트: {project_root}")

    show_used_file_expanders(project_root, file_paths)

    # st.session_state는 화면이 다시 그려져도 값을 기억해 두는 "화면용 메모장"입니다.
    # 버튼을 누르기 전에는 결과가 없으므로 None으로 초기화합니다.
    if "agent_result" not in st.session_state:
        st.session_state["agent_result"] = None

    # st.button(...)은 버튼을 누른 그 순간에만 True가 됩니다.
    # 즉, 아래 코드는 "Agent 실행하기" 버튼을 눌렀을 때만 동작합니다.
    if st.button("Agent 실행하기", type="primary"):
        try:
            with st.spinner("Agent가 Prompt를 만들고 LLM 응답을 생성하는 중입니다..."):
                st.session_state["agent_result"] = run_agent_once(project_root)
        except Exception as exc:
            st.error("Agent 실행 중 오류가 발생했습니다.")
            st.code(str(exc)[:1000])
            st.session_state["agent_result"] = None

    result = st.session_state["agent_result"]

    if result is None:
        st.markdown("---")
        st.write("아직 Agent를 실행하지 않았습니다. 위의 **Agent 실행하기** 버튼을 누르면 실행 과정이 표시됩니다.")
        return

    st.markdown("---")

    st.subheader("1. 실습 조건")
    col1, col2, col3 = st.columns(3)
    col1.metric("설비 ID", result["equipment_id"])
    col2.metric("알람 코드", result["alarm_code"])
    col3.metric("관련 로그 수", result["total_count"])

    st.write("사용자 요청")
    st.info(result["user_query"])

    st.write("로그 요약")
    st.json(
        {
            "first_time": result["first_time"],
            "last_time": result["last_time"],
            "severity_counts": result["severity_counts"],
        }
    )

    st.subheader("2. 필터링된 로그")
    st.dataframe(result["filtered_logs"], use_container_width=True)

    st.subheader("3. 참고 매뉴얼 일부")
    st.code(result["manual_preview"], language="markdown")

    st.subheader("4. LLM에 전달된 Prompt")
    st.code(result["prompt"], language="markdown")

    st.subheader("5. LLM 응답")
    st.markdown(result["llm_response"])

    st.subheader("6. 저장된 Markdown 결과")
    st.markdown(result["report"])

    relative_output = result["output_path"].relative_to(result["project_root"])
    st.success(f"결과 파일 저장 완료: {relative_output}")

    st.download_button(
        label="Markdown 결과 다운로드",
        data=result["report"],
        file_name="first_sample_run_result.md",
        mime="text/markdown",
    )


# 이 아래 부분은 이 파일을 직접 실행했을 때만 동작합니다.
# Streamlit 앱은 보통 터미널에서 다음 명령으로 실행합니다.
#   streamlit run src/day1/first_sample_run_streamlit_app.py
# 초보자 관점에서는 "이 화면의 시작 버튼"이라고 이해하면 됩니다.
if __name__ == "__main__":
    main()
