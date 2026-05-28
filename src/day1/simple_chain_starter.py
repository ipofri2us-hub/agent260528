"""
삼성디스플레이 재직자 대상 AI Agent Architecture 1일차 5교시 단순 Chain 실습 파일입니다.

이 파일은 초보자가 위에서 아래로 실행 흐름을 이해할 수 있도록
핵심 코드를 run_simple_chain() 함수 안에 모아 둔 교육용 단순 버전입니다.

실습 주제:
- Agentic Workflow: Chain 기반 업무 처리 실습
- Prompt Mustache 템플릿 분리
- Result Markdown Mustache 템플릿 분리
- llm_client.py를 통한 LLM 호출부 교체 가능 구조 이해

중요:
- 이 파일은 Gemini/OpenAI/Anthropic/Ollama/NVIDIA SDK를 직접 import하지 않습니다.
- LLM 호출은 llm_client.py의 generate_response(prompt)를 통해서만 수행합니다.
- API Key, Authorization, token, password, secret 값은 출력하거나 저장하지 않습니다.
- 실제 사내 데이터가 아니라 DisplayEdu Fab 교육용 가상 데이터만 사용합니다.
"""

# ============================================================
# 파일명: simple_chain_starter.py
# 목적:
#   여러 작업을 정해진 순서대로 이어서 처리하는 "Chain(체인)" 흐름을
#   이해하기 위한 1일차 실습 파일입니다.
#
# 이 파일에서 배우는 것:
#   1. Chain = "입력 → 가공 → LLM 호출 → 결과 정리"처럼 단계를 한 줄로 잇는 방식
#   2. 매뉴얼 문서에서 알람 코드와 관련된 줄만 골라내는 단순 검색
#   3. Prompt 양식(Mustache)과 결과 양식을 코드와 분리해 두는 이유
#   4. llm_client.py를 통해서만 LLM을 호출하는 교체 가능한 구조
#
# 전체 실행 흐름:
#   1. 프로젝트 경로와 사용할 파일들을 준비합니다.
#   2. 사용자 요청(JSON)을 읽습니다.
#   3. 알람 매뉴얼(Markdown)을 읽습니다.
#   4. 알람 코드와 관련된 매뉴얼 일부를 단순 검색으로 찾습니다.
#   5. Prompt 템플릿에 값을 채워 LLM Prompt를 만듭니다.
#   6. LLM을 호출해 답변을 받습니다.
#   7. 결과 템플릿으로 Markdown 보고서를 만들어 저장합니다.
#
# 초보자를 위한 비유:
#   Chain은 공장의 컨베이어 벨트와 같습니다.
#   재료(사용자 요청)가 벨트를 따라 정해진 작업 순서를 차례로 지나
#   완성품(보고서)이 되어 나옵니다.
#   (매뉴얼에서 관련 내용을 찾아 답하는 본격적인 RAG는 2일차에서 배웁니다.)
# ============================================================

from pathlib import Path
import json
import sys

import pystache


# ------------------------------------------------------------
# 함수명: run_simple_chain
# 역할:
#   1일차 Chain 실습의 전체 단계를 순서대로 한 번 실행합니다.
#
# 입력값:
#   없음 (필요한 데이터는 함수 안에서 파일로 읽습니다.)
#
# 출력값:
#   반환값은 없습니다.
#   진행 상황을 화면에 출력하고, 결과를 Markdown 파일로 저장합니다.
#
# 초보자 설명:
#   "사용자 요청 → 매뉴얼 검색 → Prompt 만들기 → LLM 호출 → 결과 저장"이라는
#   체인(연속 작업)이 위에서 아래로 어떻게 이어지는지 보여 주는 함수입니다.
# ------------------------------------------------------------
def run_simple_chain():
    """
    1일차 Chain 실습을 실행합니다.

    이 함수는 초보자가 Chain 기반 Agent 흐름을 이해할 수 있도록
    핵심 코드를 한 곳에 모아 둔 교육용 단순 버전입니다.

    실행 흐름:
    1. 프로젝트 경로를 찾습니다.
    2. sample_query.json을 읽습니다.
    3. alarm_manual.md를 읽습니다.
    4. 알람 코드와 관련된 매뉴얼 일부를 찾습니다.
    5. Prompt Mustache 템플릿으로 LLM Prompt를 만듭니다.
    6. llm_client.py를 통해 LLM을 호출합니다.
    7. Result Mustache 템플릿으로 Markdown 결과를 만듭니다.
    8. outputs/day1/simple_chain_result.md 파일로 저장합니다.
    """
    print("[1일차 Chain 실습] 사용자 요청을 읽습니다.")

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

    # 3. 이번 Chain 실습에서 사용할 파일 경로를 준비합니다.
    query_path = project_root / "data" / "sample_query.json"
    manual_path = project_root / "docs" / "alarm_manual.md"

    prompt_template_path = project_root / "templates" / "day1" / "simple_chain_prompt.mustache"
    result_template_path = project_root / "templates" / "day1" / "simple_chain_result.mustache"

    output_path = project_root / "outputs" / "day1" / "simple_chain_result.md"

    # 4. 사용자 요청을 읽습니다.
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

    print(f"설비 ID: {equipment_id}")
    print(f"알람 코드: {alarm_code}")

    # 5. 알람 매뉴얼을 읽습니다.
    manual_text = manual_path.read_text(encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 단계: 매뉴얼에서 알람과 관련된 줄만 골라내기 (단순 검색)
    # ------------------------------------------------------------

    # 6. 알람 코드와 관련된 매뉴얼 일부를 찾습니다.
    # 실제 RAG 검색은 2일차에서 더 자세히 다룹니다.
    # 여기서는 Chain 흐름을 단순하게 보기 위해 알람 코드와 주요 키워드가 포함된 줄만 사용합니다.
    # splitlines()는 긴 매뉴얼 글을 한 줄씩 잘라 목록으로 만듭니다.
    manual_lines = manual_text.splitlines()
    found_lines = []

    # 매뉴얼을 한 줄씩 보면서 알람 코드나 핵심 단어가 들어 있는 줄만 모읍니다.
    # (문장의 '의미'를 이해하는 검색이 아니라, 단어가 들어 있는지만 보는 가장 단순한 검색입니다.)
    for line in manual_lines:
        if alarm_code in line or "온도" in line or "확인" in line or "조치" in line:
            found_lines.append(line)

    manual_section = "\n".join(found_lines[:40])

    if not manual_section.strip():
        manual_section = manual_text[:2000]

    # 7. Prompt Mustache 템플릿을 읽습니다.
    prompt_template = prompt_template_path.read_text(encoding="utf-8-sig")

    # 8. Prompt 템플릿에 넣을 데이터를 준비합니다.
    prompt_data = {
        "user_query": user_query,
        "line_id": line_id,
        "process_name": process_name,
        "equipment_id": equipment_id,
        "alarm_code": alarm_code,
        "manual_section": manual_section,
    }

    # 9. Mustache 템플릿으로 LLM에게 보낼 Prompt를 만듭니다.
    prompt = pystache.render(prompt_template, prompt_data)

    # ------------------------------------------------------------
    # 단계: LLM 호출 (체인의 핵심 단계)
    # ------------------------------------------------------------

    # 10. llm_client.py를 통해 LLM을 호출합니다.
    # 이 파일에서는 provider별 SDK를 직접 import하지 않습니다.
    # .env 내용이나 API Key를 출력하지 않습니다.
    # generate_response(prompt) 하나만 호출하면 되므로, 나중에 LLM 종류를 바꿔도 이 코드는 그대로입니다.
    from llm_client import generate_response

    print("선택된 LLM으로 응답 생성 중...")
    response = generate_response(prompt)

    # 11. Result Mustache 템플릿을 읽습니다.
    result_template = result_template_path.read_text(encoding="utf-8-sig")

    # 12. Result 템플릿에 넣을 데이터를 준비합니다.
    result_data = {
        "scenario_name": scenario_name,
        "user_query": user_query,
        "line_id": line_id,
        "process_name": process_name,
        "equipment_id": equipment_id,
        "alarm_code": alarm_code,
        "manual_section": manual_section,
        "prompt": prompt,
        "response": response,
    }

    # 13. Mustache 템플릿으로 Markdown 결과를 만듭니다.
    result_markdown = pystache.render(result_template, result_data)

    # 14. 결과 Markdown 파일을 저장합니다.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result_markdown, encoding="utf-8-sig")

    print(f"결과 저장: {output_path.relative_to(project_root)}")
    print("다음 단계: mini_graph_runner.py에서 조건부 분기를 실습합니다.")


def main():
    """
    프로그램 시작점입니다.
    """
    run_simple_chain()


# 이 아래 부분은 이 파일을 직접 실행했을 때만 동작합니다.
# (예: 터미널에서 `python src/day1/simple_chain_starter.py` 를 입력했을 때)
# 다른 파일에서 import해서 사용할 때는 실행되지 않습니다.
# 초보자 관점에서는 "이 파일의 시작 버튼"이라고 이해하면 됩니다.
if __name__ == "__main__":
    main()
