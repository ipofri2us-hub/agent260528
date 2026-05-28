"""
mock_llm.py

AI Agent Architecture 1일차 실습용 교육용 Mock LLM입니다.

이 파일은 Gemini API 호출이 실패하거나 API Key가 없을 때,
수업 흐름을 유지하기 위한 예비 응답 생성기로 사용합니다.

중요:
- 외부 API를 호출하지 않습니다.
- 실제 AI 모델이 아닙니다.
- 실제 삼성디스플레이 사내 데이터, 실제 설비명, 실제 라인명,
  실제 알람 코드, 실제 공정 조건, 실제 품질 기준을 사용하지 않습니다.
- 모든 응답은 DisplayEdu Fab 교육용 가상 시나리오를 기준으로 생성됩니다.
"""

from __future__ import annotations

import re
from typing import Dict, Optional


# ---------------------------------------------------------------------
# 공개 함수
# ---------------------------------------------------------------------
def generate_mock_response(prompt: str) -> str:
    """
    입력 프롬프트 안의 교육용 알람 코드를 찾아 Markdown 응답을 반환합니다.

    이 함수는 실제 LLM API를 호출하지 않습니다.
    단순 고정 문장만 반환하는 것이 아니라, 프롬프트 안의 설비 ID, 라인 ID,
    공정명, 알람 코드 같은 문자열을 간단히 추출하여 교육용 리포트에 반영합니다.

    Args:
        prompt: 사용자가 입력한 질문 또는 Agent가 만든 프롬프트 문자열

    Returns:
        Markdown 형식의 교육용 가상 응답 문자열
    """
    if not isinstance(prompt, str):
        return _build_need_more_info_response("")

    normalized_prompt = prompt.upper()

    if "ALM-TEMP-402" in normalized_prompt:
        return _build_temp_alarm_response(prompt)

    if "ALM-VAC-210" in normalized_prompt:
        return _build_vac_alarm_response(prompt)

    if "ALM-PRESS-115" in normalized_prompt:
        return _build_pressure_alarm_response(prompt)

    if "ALM-FLOW-301" in normalized_prompt:
        return _build_flow_alarm_response(prompt)

    return _build_need_more_info_response(prompt)


# ---------------------------------------------------------------------
# 내부 응답 생성 함수
# ---------------------------------------------------------------------
def _build_temp_alarm_response(prompt: str) -> str:
    """
    ALM-TEMP-402 교육용 챔버 온도 편차 알람 응답을 생성합니다.
    """
    context = _extract_simple_context(prompt)

    line_id = context.get("line_id") or "EDU-LINE-07"
    process_name = context.get("process_name") or "Thin Film Deposition"
    equipment_id = context.get("equipment_id") or "EQP-EV-03"
    alarm_code = context.get("alarm_code") or "ALM-TEMP-402"

    return f"""# 교육용 가상 응답: 챔버 온도 편차 반복 알람 분석 리포트

> 이 응답은 DisplayEdu Fab 교육용 가상 시나리오를 기반으로 생성된 Mock LLM 응답입니다.  
> 실제 삼성디스플레이의 사내 데이터, 실제 설비 기준, 실제 공정 조건, 실제 조치 절차와 무관합니다.  
> 실제 현장 조치로 사용하지 말고, AI Agent 응답 구조를 이해하기 위한 예시로만 활용해야 합니다.

---

## 1. 요청 요약

- 가상 회사명: DisplayEdu Fab
- 가상 라인명: {line_id}
- 가상 공정: {process_name}
- 가상 설비 ID: {equipment_id}
- 가상 알람 코드: {alarm_code}
- 알람 유형: 교육용 챔버 온도 편차 알람
- 요청 목적: 알람 로그와 매뉴얼 근거를 참고하여 반복 발생 여부, 원인 후보, 1차 확인 항목, 권장 조치, 추가 확인 필요 사항을 정리합니다.

---

## 2. 로그 기반 관찰 내용

### 2.1 반복 발생 여부

입력 프롬프트에 포함된 상황을 기준으로 보면, `{equipment_id}` 설비에서 `{alarm_code}` 알람이 반복 발생한 것으로 해석할 수 있습니다.  
다만 이 판단은 교육용 로그와 교육용 매뉴얼을 기준으로 한 예시이며, 실제 공정 판단으로 단정하면 안 됩니다.

Agent는 실제 구현 시 다음 항목을 근거로 반복 발생 여부를 확인할 수 있습니다.

- 동일 설비 ID가 여러 번 등장하는지
- 동일 알람 코드가 시간 순서상 반복되는지
- 동일 챔버에서 집중적으로 발생하는지
- `WARNING`과 `CRITICAL` 같은 심각도 변화가 있는지
- 작업자 메모에 “반복”, “재발”, “확인 필요”와 같은 표현이 있는지

### 2.2 관련 설비

현재 요청에서 중심이 되는 가상 설비는 `{equipment_id}`입니다.  
이 설비명은 교육용 가상 ID이며, 실제 회사 설비명으로 해석하면 안 됩니다.

### 2.3 관련 공정

관련 공정은 `{process_name}`입니다.  
박막 제조 상황을 연상할 수 있도록 구성한 교육용 공정명이며, 실제 공정 조건이나 실제 레시피를 의미하지 않습니다.

### 2.4 심각도 해석

`{alarm_code}`가 반복 발생하고 일부 로그에서 높은 심각도로 표시된다면, Agent는 이를 “추가 확인이 필요한 반복 알람”으로 정리할 수 있습니다.  
그러나 심각도만으로 실제 원인이나 실제 품질 영향을 확정해서는 안 됩니다.

---

## 3. 알람 의미 해석

`{alarm_code}`는 교육용 시나리오에서 챔버 온도 편차와 관련된 가상 알람입니다.  
박막 증착 공정에서는 챔버 상태가 일정하게 유지되는지 확인하는 흐름이 중요할 수 있으므로, 이 알람은 다음과 같은 관점에서 확인할 수 있습니다.

- 챔버 온도 편차가 반복되는지
- 특정 챔버에 집중되는지
- 다른 교육용 알람과 근접한 시간대에 함께 발생하는지
- 품질 영향 가능성을 검토할 필요가 있는지

이 응답은 실제 설비 조작 방법이나 실제 공정 기준을 제시하지 않습니다.

---

## 4. 원인 후보

아래 항목은 확정 원인이 아니라 교육용 관점의 원인 후보입니다.

### 4.1 챔버 온도 제어 불안정 가능성

동일 챔버에서 온도 편차 알람이 반복된다면, 챔버 온도 제어 상태를 점검 대상으로 볼 수 있습니다.  
다만 실제 제어 장치 이상으로 단정하지 않고, 로그와 추가 점검 결과를 함께 확인해야 합니다.

### 4.2 센서 측정값 변동 가능성

측정값이 일정하지 않고 반복적으로 변동되는 흐름이 있다면, 센서 측정값 변동 가능성을 확인할 수 있습니다.  
이 경우에도 센서 고장이라고 단정하지 않고, 측정 추이와 현장 메모를 함께 살펴보는 것이 적절합니다.

### 4.3 공정 전후 조건 변화 가능성

알람이 특정 시간대나 특정 작업 흐름 이후 반복된다면, 공정 전후 조건 변화가 영향을 주었는지 확인할 수 있습니다.  
이 문서는 교육용이므로 실제 공정 조건, 실제 레시피, 실제 설비 세팅값은 언급하지 않습니다.

### 4.4 반복 발생에 따른 점검 필요성

동일 알람이 여러 번 발생하면 일회성 알람으로 보기 어렵습니다.  
Agent는 “반복 발생 가능성이 있으므로 추가 확인이 필요합니다”라고 표현하는 것이 안전합니다.

---

## 5. 1차 확인 항목

Agent는 다음 항목을 우선 확인할 수 있습니다.

| 확인 항목 | 확인 이유 |
|---|---|
| 최근 동일 알람 반복 여부 | `{alarm_code}`가 시간 순서상 반복되는지 확인합니다. |
| 동일 설비 집중 여부 | `{equipment_id}`에서 주로 발생하는지 확인합니다. |
| 동일 챔버 집중 여부 | 특정 챔버에 알람이 몰리는지 확인합니다. |
| 관련 알람 동시 발생 여부 | `ALM-VAC-210`, `ALM-PRESS-115`, `ALM-FLOW-301` 같은 교육용 보조 알람과 함께 나타나는지 확인합니다. |
| 심각도 변화 | `WARNING`에서 `CRITICAL`로 변화하는 흐름이 있는지 확인합니다. |
| 품질 영향 가능성 | 박막 균일도와 같은 품질 영향 가능성을 추가 확인 대상으로 볼 수 있습니다. |

---

## 6. 권장 조치 방향

아래 내용은 실제 설비 조작 지시가 아니라 교육용 확인 방향입니다.

1. `{equipment_id}`의 `{alarm_code}` 발생 이력을 시간순으로 정리할 수 있습니다.
2. 동일 챔버에서 반복 발생했는지 확인할 수 있습니다.
3. 주변 시간대의 진공, 압력, 흐름 관련 교육용 알람을 함께 확인할 수 있습니다.
4. 현장 메모에서 “반복”, “재발”, “확인 필요”와 같은 표현을 확인할 수 있습니다.
5. 품질 영향 가능성은 별도 데이터와 함께 검토할 수 있습니다.

실제 조치 여부는 담당자 검토가 필요한 영역이며, Agent가 단독으로 확정하면 안 됩니다.

---

## 7. 추가 확인 필요 사항

- 실제 원인 확정을 위해서는 로그 외의 추가 정보가 필요할 수 있습니다.
- 교육용 매뉴얼에 없는 실제 조치 절차를 Agent가 임의로 만들면 안 됩니다.
- 품질 영향 가능성은 알람 로그만으로 단정하지 않아야 합니다.
- 동일 알람의 반복 횟수, 챔버 집중 여부, 다른 관련 알람과의 시간 관계를 함께 확인할 수 있습니다.

---

## 8. Agent 응답 시 주의 사항

- 로그와 매뉴얼에 없는 내용을 단정하지 않습니다.
- 실제 공정 기준, 실제 설비 세팅값, 실제 품질 기준을 생성하지 않습니다.
- 실제 내부 시스템명이나 실제 절차명처럼 보이는 표현을 사용하지 않습니다.
- “가능성이 있습니다”, “확인할 수 있습니다”, “추가 확인이 필요합니다”와 같은 안전한 표현을 사용합니다.
- 이 응답이 교육용 가상 응답임을 명확히 표시합니다.

---

## 9. 다음 단계 제안

1일차에는 이 응답을 통해 Prompt, Chain, mini Graph, Day1 Agent v0의 기본 흐름을 확인할 수 있습니다.  
2일차 RAG 실습에서는 `alarm_manual.md`에서 검색된 근거 문단을 함께 연결하여, 매뉴얼 기반 답변을 더 명확하게 만들 수 있습니다.
"""


def _build_vac_alarm_response(prompt: str) -> str:
    """
    ALM-VAC-210 교육용 진공 상태 변동 알람 응답을 생성합니다.
    """
    context = _extract_simple_context(prompt)
    equipment_id = context.get("equipment_id") or "EQP-EV-03"
    line_id = context.get("line_id") or "EDU-LINE-07"

    return f"""# 교육용 가상 응답: 챔버 진공 상태 변동 알람 요약

> 이 응답은 DisplayEdu Fab 교육용 Mock LLM 응답입니다.  
> 실제 기업 내부 매뉴얼, 실제 설비 기준, 실제 조치 절차와 무관합니다.

## 1. 요청 요약

- 가상 라인명: {line_id}
- 가상 설비 ID: {equipment_id}
- 가상 알람 코드: ALM-VAC-210
- 알람 유형: 교육용 진공 상태 변동 알람

## 2. 알람 의미

`ALM-VAC-210`은 교육용 시나리오에서 챔버의 진공 상태가 일정하지 않은 상황을 설명하기 위한 가상 알람입니다.  
이 알람만으로 실제 원인을 확정할 수 없으며, 주변 시간대의 다른 알람과 함께 확인할 수 있습니다.

## 3. 확인할 수 있는 항목

- 동일 설비에서 반복 발생했는지 확인할 수 있습니다.
- 동일 챔버에서 집중적으로 발생했는지 확인할 수 있습니다.
- `ALM-TEMP-402` 같은 온도 편차 알람과 근접한 시간대에 함께 발생했는지 확인할 수 있습니다.
- 작업자 메모에 “확인 필요”, “변동”, “관찰” 같은 표현이 있는지 볼 수 있습니다.

## 4. 원인 후보

- 챔버 상태 변화 가능성
- 측정값 변동 가능성
- 온도 편차 알람과의 연관 가능성
- 일시적 상태 변화 가능성

위 항목은 확정 원인이 아니라 교육용 분석 후보입니다.

## 5. 권장 조치 방향

실제 조작 지시를 제시하지 않고, 로그와 매뉴얼을 기준으로 추가 확인 방향을 정리할 수 있습니다.  
Agent는 “진공 상태 변동 가능성이 있으므로 관련 로그를 추가 확인할 수 있습니다”와 같이 표현해야 합니다.

## 6. 주의 사항

이 응답은 교육용 가상 응답입니다. 실제 공정 기준이나 실제 조치 절차로 사용하면 안 됩니다.
"""


def _build_pressure_alarm_response(prompt: str) -> str:
    """
    ALM-PRESS-115 교육용 챔버 압력 변동 알람 응답을 생성합니다.
    """
    context = _extract_simple_context(prompt)
    equipment_id = context.get("equipment_id") or "EQP-EV-03"
    process_name = context.get("process_name") or "Thin Film Deposition"

    return f"""# 교육용 가상 응답: 챔버 압력 변동 알람 요약

> 이 응답은 DisplayEdu Fab 교육용 가상 시나리오를 기반으로 합니다.  
> 실제 삼성디스플레이 데이터, 실제 설비 기준, 실제 공정 조건과 무관합니다.

## 1. 요청 요약

- 가상 공정: {process_name}
- 가상 설비 ID: {equipment_id}
- 가상 알람 코드: ALM-PRESS-115
- 알람 유형: 교육용 챔버 압력 변동 알람

## 2. 알람 의미

`ALM-PRESS-115`는 교육용 시나리오에서 챔버 압력 상태가 일정하지 않은 흐름을 설명하기 위한 가상 알람입니다.  
이 알람은 실제 압력 기준이나 실제 허용오차를 의미하지 않습니다.

## 3. 로그 기반 확인 항목

- 동일 설비에서 반복 발생했는지 확인할 수 있습니다.
- 같은 챔버에서 집중되는지 볼 수 있습니다.
- 온도 편차 또는 흐름 확인 알람과 함께 발생했는지 확인할 수 있습니다.
- 심각도가 변화하는 흐름이 있는지 확인할 수 있습니다.

## 4. 원인 후보

- 챔버 상태 변화 가능성
- 측정값 변동 가능성
- 주변 조건 변화 가능성
- 다른 교육용 알람과의 연관 가능성

원인은 확정하지 않고 후보로만 정리해야 합니다.

## 5. 권장 조치 방향

교육용 관점에서는 압력 변동 알람의 발생 시각, 반복 여부, 관련 알람 동시 발생 여부를 정리할 수 있습니다.  
실제 설비 조작, 실제 공정 조건 변경, 실제 기준값 판단은 Agent가 수행하면 안 됩니다.

## 6. 주의 사항

응답에는 교육용 가상 데이터임을 명시하고, 실제 조치 절차처럼 단정하지 않아야 합니다.
"""


def _build_flow_alarm_response(prompt: str) -> str:
    """
    ALM-FLOW-301 교육용 가스 흐름 확인 알람 응답을 생성합니다.
    """
    context = _extract_simple_context(prompt)
    equipment_id = context.get("equipment_id") or "EQP-EV-03"

    return f"""# 교육용 가상 응답: 가스 흐름 확인 알람 요약

> 이 응답은 DisplayEdu Fab 교육용 Mock LLM 응답입니다.  
> 실제 공정 가스, 실제 유량 기준, 실제 설비 조작 절차와 무관합니다.

## 1. 요청 요약

- 가상 설비 ID: {equipment_id}
- 가상 알람 코드: ALM-FLOW-301
- 알람 유형: 교육용 가스 흐름 확인 알람

## 2. 알람 의미

`ALM-FLOW-301`은 교육용 시나리오에서 챔버 흐름 상태를 참고 확인하기 위한 가상 알람입니다.  
실제 가스 종류나 실제 유량 기준을 의미하지 않습니다.

## 3. 확인할 수 있는 항목

- 흐름 확인 알람이 단독으로 발생했는지 확인할 수 있습니다.
- 온도 편차 알람 또는 압력 변동 알람과 근접한 시간대에 함께 발생했는지 확인할 수 있습니다.
- 동일 설비와 동일 챔버에서 반복되는지 확인할 수 있습니다.
- 현장 메모에 추가 관찰이 필요한 표현이 있는지 확인할 수 있습니다.

## 4. 원인 후보

- 흐름 상태 변동 가능성
- 측정값 변동 가능성
- 챔버 상태 변화와의 연관 가능성
- 다른 교육용 알람과의 동시 발생 가능성

위 내용은 교육용 후보이며, 실제 원인으로 확정하면 안 됩니다.

## 5. 권장 조치 방향

Agent는 관련 로그를 정리하고, 다른 알람과의 시간 관계를 확인할 수 있습니다.  
실제 설비 조작이나 실제 공정 조건 변경을 지시하지 않습니다.

## 6. 주의 사항

이 응답은 교육용 가상 응답이며 실제 사내 매뉴얼이나 실제 조치 기준이 아닙니다.
"""


def _build_need_more_info_response(prompt: str) -> str:
    """
    알람 코드 또는 설비 ID가 부족할 때 추가 정보 요청 응답을 생성합니다.
    """
    context = _extract_simple_context(prompt)

    found_equipment = context.get("equipment_id")
    found_alarm = context.get("alarm_code")

    missing_items = []
    if not found_equipment:
        missing_items.append("설비 ID")
    if not found_alarm:
        missing_items.append("알람 코드")

    if not missing_items:
        missing_text = "분석 대상 정보"
    else:
        missing_text = ", ".join(missing_items)

    return f"""# 교육용 가상 응답: 추가 정보 요청

> 이 응답은 DisplayEdu Fab 교육용 Mock LLM 응답입니다.  
> 실제 삼성디스플레이 업무 요청이나 실제 내부 시스템 응답이 아닙니다.

## 1. 현재 상태

입력 프롬프트에서 `{missing_text}` 정보가 충분히 확인되지 않았습니다.  
제조 알람 분석 Agent가 리포트를 작성하려면 최소한 설비 ID와 알람 코드가 필요합니다.

## 2. 추가로 필요한 정보

다음 정보를 입력해 주면 교육용 알람 분석 리포트를 생성할 수 있습니다.

- 가상 설비 ID 예시: `EQP-EV-03`
- 가상 알람 코드 예시: `ALM-TEMP-402`, `ALM-VAC-210`, `ALM-PRESS-115`, `ALM-FLOW-301`
- 참고 파일 예시: `sample_alarm_logs.csv`, `alarm_manual.md`

## 3. 요청 예시

`EQP-EV-03에서 ALM-TEMP-402 알람이 반복 발생했습니다. sample_alarm_logs.csv와 alarm_manual.md를 참고하여 Markdown 리포트로 정리해 주세요.`

## 4. 주의 사항

이 Mock LLM은 교육용 가상 데이터만 다룹니다.  
실제 사내 데이터, 실제 내부 시스템명, 실제 설비 기준, 실제 조치 절차는 입력하지 않아야 합니다.
"""


# ---------------------------------------------------------------------
# 간단한 문맥 추출 함수
# ---------------------------------------------------------------------
def _extract_simple_context(prompt: str) -> Dict[str, Optional[str]]:
    """
    프롬프트에서 교육용 문맥 정보를 간단히 추출합니다.

    정교한 자연어 처리를 하지 않습니다.
    수업용 예제로 이해하기 쉽게 정규식과 문자열 검색만 사용합니다.

    추출 대상:
    - equipment_id: 예) EQP-EV-03
    - alarm_code: 예) ALM-TEMP-402
    - line_id: 예) EDU-LINE-07
    - process_name: 예) Thin Film Deposition

    Args:
        prompt: 입력 프롬프트 문자열

    Returns:
        추출된 값이 들어 있는 dict
    """
    if not isinstance(prompt, str):
        prompt = ""

    context: Dict[str, Optional[str]] = {
        "equipment_id": None,
        "alarm_code": None,
        "line_id": None,
        "process_name": None,
    }

    equipment_match = re.search(r"\bEQP-[A-Z]+-\d{2,}\b", prompt, flags=re.IGNORECASE)
    if equipment_match:
        context["equipment_id"] = equipment_match.group(0).upper()

    alarm_match = re.search(r"\bALM-[A-Z]+-\d{3,}\b", prompt, flags=re.IGNORECASE)
    if alarm_match:
        context["alarm_code"] = alarm_match.group(0).upper()

    line_match = re.search(r"\bEDU-LINE-\d{2,}\b", prompt, flags=re.IGNORECASE)
    if line_match:
        context["line_id"] = line_match.group(0).upper()

    # 공정명은 이번 교육 시나리오에서 고정적으로 사용하는 가상 공정명만 감지합니다.
    if re.search(r"Thin\s+Film\s+Deposition", prompt, flags=re.IGNORECASE):
        context["process_name"] = "Thin Film Deposition"

    return context


# ---------------------------------------------------------------------
# 단독 실행 테스트
# ---------------------------------------------------------------------
if __name__ == "__main__":
    test_prompts = [
        {
            "name": "ALM-TEMP-402 정상 케이스",
            "prompt": (
                "DisplayEdu Fab의 EDU-LINE-07 Thin Film Deposition 공정에서 "
                "EQP-EV-03 설비의 ALM-TEMP-402 알람이 반복 발생했습니다. "
                "sample_alarm_logs.csv와 alarm_manual.md를 참고하여 Markdown 리포트로 정리해 주세요."
            ),
        },
        {
            "name": "ALM-VAC-210 정상 케이스",
            "prompt": (
                "EDU-LINE-07에서 EQP-EV-03 설비에 ALM-VAC-210 알람이 발생했습니다. "
                "교육용 로그 기준으로 요약해 주세요."
            ),
        },
        {
            "name": "알람 코드 없음 케이스",
            "prompt": (
                "EQP-EV-03 설비에서 반복 알람이 발생한 것 같습니다. "
                "어떤 내용을 확인해야 하나요?"
            ),
        },
        {
            "name": "설비 ID 없음 케이스",
            "prompt": (
                "ALM-TEMP-402 알람이 반복 발생했습니다. "
                "교육용 매뉴얼을 참고해서 원인 후보를 정리해 주세요."
            ),
        },
    ]

    for index, item in enumerate(test_prompts, start=1):
        print("=" * 80)
        print(f"[테스트 {index}] {item['name']}")
        print("-" * 80)

        response = generate_mock_response(item["prompt"])

        # 전체 응답은 길기 때문에 앞부분만 출력합니다.
        preview_length = 800
        print(response[:preview_length])

        if len(response) > preview_length:
            print("\n... [출력 미리보기 생략: 전체 응답은 함수 반환값에서 확인할 수 있습니다.]")

        print()
