"""
local_llm.py

AI Agent Architecture 1일차 실습용 Local LLM 호출 모듈입니다.

역할:
- Ollama 로컬 서버를 HTTP API로 호출합니다.
- llm_client.py에서는 generate_local_llm_response(prompt) 함수만 호출합니다.
- 이 파일은 Local LLM 호출만 담당하고, 실패 시 mock fallback은 수행하지 않습니다.
- fallback은 공통 진입점인 llm_client.py에서 처리합니다.

보안:
- API Key, 토큰, 전체 환경변수 목록을 출력하지 않습니다.
- 실제 회사 내부 데이터, 실제 제조 데이터, 민감정보를 코드에 넣지 않습니다.
- 테스트 프롬프트는 교육용 가상 설비명만 사용합니다.
"""

from __future__ import annotations

import os
from typing import Any, Final

try:
    import requests
except ImportError as exc:
    raise ImportError(
        "requests 패키지를 import하지 못했습니다. "
        "아래 명령어로 설치한 뒤 다시 실행해 주세요.\n"
        "pip install requests"
    ) from exc


DEFAULT_LOCAL_LLM_PROVIDER: Final[str] = "ollama"
DEFAULT_OLLAMA_BASE_URL: Final[str] = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL_NAME: Final[str] = "phi3:mini"
VALID_OLLAMA_MODELS: Final[set[str]] = {"phi3:mini", "llama3.1:8b","qwen2.5:1.5b","gemma2:2b","qwen2.5:0.5b"}
DEFAULT_TIMEOUT_SECONDS: Final[int] = 600
DEFAULT_TEMPERATURE: Final[float] = 0.2
DEFAULT_MAX_OUTPUT_TOKENS: Final[int] = 2048


def generate_local_llm_response(prompt: str) -> str:
    """
    Ollama 기반 Local LLM에 프롬프트를 보내고 응답 문자열을 반환합니다.

    .env 예시:
    - LOCAL_LLM_PROVIDER=ollama
    - OLLAMA_BASE_URL=http://localhost:11434
    - OLLAMA_MODEL_NAME=phi3:mini
    - LLM_TEMPERATURE=0.2
    - LLM_MAX_OUTPUT_TOKENS=2048

    Args:
        prompt: Local LLM에 전달할 프롬프트 문자열

    Returns:
        Ollama가 생성한 응답 문자열

    Raises:
        ValueError: prompt가 비어 있거나, 지원하지 않는 provider/model이 설정된 경우
        RuntimeError: Ollama 서버 연결 실패, 모델 미설치, 응답 오류가 발생한 경우
    """
    _validate_prompt(prompt)

    local_provider = _get_local_llm_provider()
    if local_provider != "ollama":
        raise ValueError(
            f"지원하지 않는 LOCAL_LLM_PROVIDER 값입니다: {local_provider}\n"
            "현재 교육용 프로젝트에서는 LOCAL_LLM_PROVIDER=ollama 만 지원합니다."
        )

    base_url = _get_ollama_base_url()
    model_name = _get_ollama_model_name()

    return _call_ollama_generate(prompt=prompt, model_name=model_name, base_url=base_url)


def _validate_prompt(prompt: str) -> None:
    """
    프롬프트가 Local LLM에 전달 가능한 문자열인지 확인합니다.

    Args:
        prompt: 검사할 프롬프트

    Raises:
        ValueError: prompt가 문자열이 아니거나 빈 문자열인 경우
    """
    if not isinstance(prompt, str):
        raise ValueError("prompt는 문자열(str)이어야 합니다.")

    if not prompt.strip():
        raise ValueError("prompt가 비어 있습니다. Local LLM에 전달할 교육용 질문을 입력해 주세요.")


def _get_local_llm_provider() -> str:
    """
    LOCAL_LLM_PROVIDER 값을 읽습니다.

    환경변수가 없으면 ollama를 기본값으로 사용합니다.

    Returns:
        정리된 local provider 이름
    """
    provider = os.getenv("LOCAL_LLM_PROVIDER", DEFAULT_LOCAL_LLM_PROVIDER)
    return provider.strip().lower()


def _get_ollama_base_url() -> str:
    """
    Ollama 서버 주소를 읽습니다.

    환경변수가 없으면 http://localhost:11434 를 사용합니다.

    Returns:
        끝의 / 문자를 제거한 Ollama base URL
    """
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    base_url = base_url.strip().rstrip("/")

    if not base_url:
        return DEFAULT_OLLAMA_BASE_URL

    return base_url


def _get_ollama_model_name() -> str:
    """
    사용할 Ollama 모델명을 읽고 허용 모델인지 확인합니다.

    허용 모델:
    - phi3:mini
    - llama3.1:8b

    Returns:
        검증된 Ollama 모델명

    Raises:
        ValueError: 허용되지 않은 모델명이 설정된 경우
    """
    model_name = os.getenv("OLLAMA_MODEL_NAME", DEFAULT_OLLAMA_MODEL_NAME)
    model_name = model_name.strip()

    if model_name not in VALID_OLLAMA_MODELS:
        allowed = ", ".join(sorted(VALID_OLLAMA_MODELS))
        raise ValueError(
            f"지원하지 않는 OLLAMA_MODEL_NAME 값입니다: {model_name}\n"
            f"현재 교육용 프로젝트에서 허용하는 모델은 다음과 같습니다: {allowed}\n"
            "예: OLLAMA_MODEL_NAME=phi3:mini"
        )

    return model_name


def _get_generation_options() -> dict[str, Any]:
    """
    LLM 생성 옵션을 환경변수에서 읽습니다.

    LLM_TEMPERATURE와 LLM_MAX_OUTPUT_TOKENS는 mock/local/cloud에서
    공통으로 사용할 수 있는 교육용 생성 옵션입니다.

    Returns:
        Ollama /api/generate options에 전달할 dict
    """
    temperature_text = os.getenv("LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE)).strip()
    max_tokens_text = os.getenv("LLM_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS)).strip()

    try:
        temperature = float(temperature_text)
    except ValueError as exc:
        raise ValueError(
            "LLM_TEMPERATURE 값은 숫자여야 합니다. 예: LLM_TEMPERATURE=0.2"
        ) from exc

    try:
        max_output_tokens = int(max_tokens_text)
    except ValueError as exc:
        raise ValueError(
            "LLM_MAX_OUTPUT_TOKENS 값은 정수여야 합니다. 예: LLM_MAX_OUTPUT_TOKENS=2048"
        ) from exc

    return {
        "temperature": temperature,
        "num_predict": max_output_tokens,
    }




def _get_ollama_response_format() -> str:
    """
    Ollama 응답 형식 옵션을 읽습니다.

    .env에 OLLAMA_RESPONSE_FORMAT=json 을 설정하면 Ollama /api/generate에
    format="json"을 전달합니다.

    이 옵션은 4일차 Tool Selection처럼 JSON 출력 안정성이 중요한 실습에서
    사용하면 좋습니다. 값이 비어 있으면 일반 텍스트 응답을 사용합니다.
    """
    return os.getenv("OLLAMA_RESPONSE_FORMAT", "").strip().lower()


def _call_ollama_generate(prompt: str, model_name: str, base_url: str) -> str:
    """
    Ollama /api/generate 엔드포인트를 호출합니다.

    Args:
        prompt: Ollama에 전달할 프롬프트
        model_name: 사용할 Ollama 모델명
        base_url: Ollama 서버 기본 주소

    Returns:
        response JSON의 response 값

    Raises:
        RuntimeError: 연결 실패, 모델 미설치, 응답 오류가 발생한 경우
    """
    endpoint = f"{base_url}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": _get_generation_options(),
    }

    response_format = _get_ollama_response_format()
    if response_format == "json":
        payload["format"] = "json"

    try:
        response = requests.post(endpoint, json=payload, timeout=DEFAULT_TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Ollama 서버에 연결할 수 없습니다.\n"
            "먼저 Ollama가 설치되어 있고 실행 중인지 확인해 주세요.\n"
            "기본 주소: http://localhost:11434\n"
            "Windows에서는 Ollama 앱을 실행한 뒤 다시 시도해 주세요."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Ollama 응답 대기 시간이 {DEFAULT_TIMEOUT_SECONDS}초를 초과했습니다.\n"
            "모델이 처음 실행되는 중이면 시간이 오래 걸릴 수 있습니다. 잠시 후 다시 실행해 주세요."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "Ollama API 호출 중 네트워크 오류가 발생했습니다.\n"
            "OLLAMA_BASE_URL 값과 Ollama 실행 상태를 확인해 주세요."
        ) from exc

    if response.status_code == 404:
        raise RuntimeError(_build_ollama_model_help_message(model_name))

    if response.status_code >= 400:
        error_text = response.text[:500].strip()
        if "not found" in error_text.lower() or "pull" in error_text.lower():
            raise RuntimeError(_build_ollama_model_help_message(model_name))

        raise RuntimeError(
            f"Ollama API 호출에 실패했습니다. HTTP 상태 코드: {response.status_code}\n"
            f"오류 내용 일부: {error_text}"
        )

    try:
        response_json = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Ollama 응답을 JSON으로 해석하지 못했습니다. "
            "Ollama 서버 상태를 확인해 주세요."
        ) from exc

    if response_json.get("error"):
        error_message = str(response_json.get("error", "")).strip()
        if "not found" in error_message.lower() or "pull" in error_message.lower():
            raise RuntimeError(_build_ollama_model_help_message(model_name))

        raise RuntimeError(f"Ollama가 오류를 반환했습니다: {error_message}")

    generated_text = response_json.get("response", "")
    if not isinstance(generated_text, str) or not generated_text.strip():
        raise RuntimeError(
            "Ollama 응답에서 response 값을 찾지 못했거나 응답이 비어 있습니다. "
            "모델 실행 상태를 확인해 주세요."
        )

    return generated_text


def _build_ollama_model_help_message(model_name: str) -> str:
    """
    Ollama 모델이 없을 때 수강생에게 보여줄 설치 안내 문구를 만듭니다.

    Args:
        model_name: 현재 실행하려던 모델명

    Returns:
        모델 다운로드 안내 메시지
    """
    return (
        f"Ollama에서 모델을 찾지 못했습니다: {model_name}\n"
        "사용할 모델을 먼저 다운로드해 주세요.\n"
        "\n"
        "phi3:mini 모델 사용 시:\n"
        "ollama pull phi3:mini\n"
        "\n"
        "llama3.1:8b 모델 사용 시:\n"
        "ollama pull llama3.1:8b"
    )


if __name__ == "__main__":
    test_prompt = (
        "교육용 가상 설비 EDU-EQP-01에서 온도 관련 알람이 반복 발생했다고 가정합니다. "
        "초보 수강생이 이해할 수 있도록 원인 후보 3개와 1차 확인 항목을 간단히 정리해 주세요."
    )

    print("=" * 80)
    print("Local LLM 단독 실행 테스트")
    print("=" * 80)
    print(f"LOCAL_LLM_PROVIDER: {_get_local_llm_provider()}")
    print(f"OLLAMA_BASE_URL: {_get_ollama_base_url()}")
    print(f"OLLAMA_MODEL_NAME: {_get_ollama_model_name()}")
    print("주의: API Key, 토큰, 전체 환경변수 목록은 출력하지 않습니다.")
    print("-" * 80)

    try:
        result = generate_local_llm_response(test_prompt)
        preview_length = 1000
        print(result[:preview_length])

        if len(result) > preview_length:
            print("\n... [응답 미리보기 생략: 전체 응답은 함수 반환값에서 확인할 수 있습니다.]")

    except Exception as exc:
        print("Local LLM 실행 중 오류가 발생했습니다.")
        print(str(exc))
