"""
삼성디스플레이 재직자 대상 AI Agent Architecture 1일차 실습용 단순 실행 파일입니다.

이 파일은 초보자가 위에서 아래로 읽기 쉽도록 단순하게 구성했습니다.

역할:
- data/sample_query.json에서 실습 입력 조건을 읽습니다.
- data/sample_alarm_logs.csv에서 설비 ID와 알람 코드 기준으로 로그를 필터링합니다.
- docs/alarm_manual.md의 앞부분 일부를 참고 자료로 사용합니다.
- templates/day1/first_sample_run_prompt.mustache로 LLM Prompt를 만듭니다.
- llm_client.py의 generate_response(prompt)를 통해 LLM을 호출합니다.
- templates/day1/first_sample_run_result.mustache로 Markdown 결과를 만듭니다.
- outputs/day1/first_sample_run_result.md로 저장합니다.

중요:
- 이 파일은 Gemini/OpenAI/Anthropic/Ollama/NVIDIA SDK를 직접 import하지 않습니다.
- LLM 호출은 llm_client.py의 generate_response(prompt)를 통해서만 수행합니다.
- 실제 사내 데이터가 아니라 DisplayEdu Fab 교육용 가상 데이터만 사용합니다.
"""

# ============================================================
# 파일명: first_sample_run.py
# 목적:
#   제조 현장의 알람 로그와 매뉴얼을 읽어,
#   LLM에게 "어떤 알람이 몇 번 발생했는지"를 정리하게 만드는
#   1일차 첫 번째 실행 예제입니다.
#
# 이 파일에서 배우는 것:
#   1. Python 코드가 위에서 아래로 한 줄씩 실행되는 흐름
#   2. JSON, CSV, 텍스트 파일을 읽어 입력 데이터를 준비하는 방법
#   3. 템플릿(Mustache)에 데이터를 끼워 넣어 LLM Prompt를 만드는 방법
#   4. llm_client.py 한 곳을 통해서만 LLM을 호출하는 구조
#
# 전체 실행 흐름:
#   1. 입력 파일 / 템플릿 / 출력 파일의 경로를 준비합니다.
#   2. 사용자 요청(JSON)과 알람 로그(CSV)를 읽습니다.
#   3. 설비 ID와 알람 코드가 일치하는 로그만 골라 간단히 요약합니다.
#   4. 매뉴얼 앞부분과 로그 요약을 합쳐 LLM Prompt를 만듭니다.
#   5. LLM을 호출해 응답을 받습니다.
#   6. 응답을 Markdown 보고서 파일로 저장합니다.
#
# 초보자를 위한 비유:
#   이 파일은 공장에 들어온 알람 기록을 모아
#   "무슨 일이 몇 번 있었는지"를 한 장으로 정리해 주는
#   자동 보고서 작성기와 같습니다.
#   (문서를 직접 '검색'해서 답하는 RAG는 2일차에서 배웁니다.
#    이 파일은 그 전 단계의 가장 단순한 출발점입니다.)
# ============================================================

# 아래 import들은 이 파일이 사용할 도구를 미리 불러오는 부분입니다.
# - Path: 파일 경로를 다루는 도구
# - json: JSON 파일을 Python 자료로 바꾸는 도구
# - sys: 파이썬 실행 환경(특히 import 경로)을 다루는 도구
# - pandas(pd): 표(CSV) 데이터를 다루는 도구
# - pystache: Mustache 템플릿에 값을 채워 넣는 도구
from pathlib import Path
import json
import sys

import pandas as pd
import pystache


# ------------------------------------------------------------
# 함수명: run_first_sample
# 역할:
#   1일차 첫 실행 예제의 전체 흐름을 한 번에 실행합니다.
#   (데이터 읽기 → 로그 필터/요약 → Prompt 생성 → LLM 호출 → 결과 저장)
#
# 입력값:
#   없음 (필요한 데이터는 함수 안에서 파일을 직접 읽습니다.)
#
# 출력값:
#   반환값은 없습니다.
#   대신 화면에 진행 상황을 출력하고, 결과를 Markdown 파일로 저장합니다.
#
# 초보자 설명:
#   이 함수 하나만 따라 읽으면 "Agent가 알람을 정리하는 한 번의 흐름"을
#   처음부터 끝까지 이해할 수 있도록 만든 교육용 함수입니다.
# ------------------------------------------------------------
def run_first_sample():
    """
    1일차 첫 실행 예제입니다.

    이 함수는 교육용 제조 Agent가 다음 순서로 동작하는 것을 보여줍니다.

    1. 사용자 요청을 읽습니다.
    2. 제조 알람 로그 CSV를 읽습니다.
    3. 설비 ID와 알람 코드로 관련 로그를 찾습니다.
    4. 알람 매뉴얼을 읽습니다.
    5. Prompt 템플릿으로 LLM에게 보낼 프롬프트를 만듭니다.
    6. llm_client.py를 통해 LLM을 호출합니다.
    7. Result 템플릿으로 Markdown 결과를 만듭니다.
    8. 결과를 Markdown 파일로 저장합니다.
    """
    print("[1일차 첫 실행] 교육용 제조 Agent 예제를 실행합니다.")

    # ------------------------------------------------------------
    # 단계 1: 파일 경로 준비
    #   - 어떤 위치에서 실행해도 프로젝트 폴더를 찾을 수 있게 경로를 정리합니다.
    # ------------------------------------------------------------

    # 1. 현재 파일 위치를 기준으로 프로젝트 루트 경로를 찾습니다.
    # 이 파일은 src/day1 폴더 안에 있다고 가정합니다.
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]

    # 2. src 폴더를 import 경로에 추가합니다.
    # 이렇게 해야 src/llm_client.py를 불러올 수 있습니다.
    src_dir = project_root / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # 3. 이번 예제에서 사용할 파일 경로를 준비합니다.
    query_path = project_root / "data" / "sample_query.json"
    log_path = project_root / "data" / "sample_alarm_logs.csv"
    manual_path = project_root / "docs" / "alarm_manual.md"

    prompt_template_path = project_root / "templates" / "day1" / "first_sample_run_prompt.mustache"
    result_template_path = project_root / "templates" / "day1" / "first_sample_run_result.mustache"

    output_path = project_root / "outputs" / "day1" / "first_sample_run_result.md"

    # ------------------------------------------------------------
    # 단계 2: 입력 데이터 읽기 + 관련 로그만 골라 요약
    #   - 사용자 요청(JSON)과 알람 로그(CSV)를 읽습니다.
    #   - 설비 ID와 알람 코드가 일치하는 로그만 추려서 건수/시간/심각도를 정리합니다.
    # ------------------------------------------------------------

    # 4. 사용자 요청 조건을 읽습니다.
    # read_text(encoding="utf-8-sig")는 한글이 깨지지 않게 파일을 읽는 방법입니다.
    # (utf-8-sig는 Windows 메모장으로 저장한 한글 파일도 안전하게 읽습니다.)
    query_data = json.loads(query_path.read_text(encoding="utf-8-sig"))

    equipment_id = str(query_data["equipment_id"])
    alarm_code = str(query_data["alarm_code"])
    user_query = str(query_data["user_query"])

    print(f"필터 조건: equipment_id={equipment_id}, alarm_code={alarm_code}")

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

    # ------------------------------------------------------------
    # 단계 3: LLM에게 보낼 Prompt(지시문) 만들기
    #   - 템플릿 파일에 위에서 정리한 값들을 끼워 넣어 완성된 Prompt를 만듭니다.
    # ------------------------------------------------------------

    # 10. Prompt Mustache 템플릿을 읽습니다.
    # Mustache 템플릿은 {{이름}} 같은 빈칸이 들어 있는 "양식 문서"입니다.
    # 코드에서 그 빈칸에 실제 값을 채워 넣어 완성된 Prompt를 만듭니다.
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
    # pystache.render(양식, 값들)은 양식의 빈칸을 값으로 채워 완성된 글을 돌려줍니다.
    prompt = pystache.render(prompt_template, prompt_data)

    # ------------------------------------------------------------
    # 단계 4: LLM 호출 (실제 답변 생성)
    # ------------------------------------------------------------

    # 13. llm_client.py를 통해 LLM을 호출합니다.
    # 이 파일에서는 provider별 SDK를 직접 import하지 않습니다.
    # generate_response(prompt)는 "Prompt를 보내고 답변 글을 돌려받는" 단 하나의 창구입니다.
    # mock / local / cloud 중 무엇을 쓸지는 .env 설정이 정하므로, 이 코드는 바꿀 필요가 없습니다.
    from llm_client import generate_response

    print("선택된 LLM으로 응답 생성 중...")
    llm_response = generate_response(prompt)

    # ------------------------------------------------------------
    # 단계 5: 결과를 Markdown 보고서로 만들어 파일로 저장
    # ------------------------------------------------------------

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

    print(f"관련 로그 수: {total_count}건")
    print(f"결과 저장: {output_path.relative_to(project_root)}")


def main():
    """
    프로그램 시작점입니다.
    """
    run_first_sample()


# 이 아래 부분은 이 파일을 직접 실행했을 때만 동작합니다.
# (예: 터미널에서 `python src/day1/first_sample_run.py` 를 입력했을 때)
# 다른 파일에서 import해서 함수만 가져다 쓸 때는 실행되지 않습니다.
# 초보자 관점에서는 "이 파일의 시작 버튼"이라고 이해하면 됩니다.
if __name__ == "__main__":
    main()
