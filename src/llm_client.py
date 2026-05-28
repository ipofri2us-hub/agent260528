"""
llm_client.py

AI Agent Architecture 1~5일차 실습용 LLM 공통 진입점입니다.

역할:
- Agent 본체 코드가 특정 LLM 제품명에 직접 의존하지 않게 합니다.
- simple_chain_starter.py, day1_agent_v0_template.py, Day2 RAG Agent,
  Day4 Tool Selector, Day5 Final MCP Multi-Agent 등은 이 파일의
  generate_response(prompt) 또는 generate_json_response(prompt) 함수만 호출합니다.
- llm_client.py는 .env 파일을 자동으로 읽어 LLM_PROVIDER 값을 확인합니다.
- LLM 실행 방식은 .env의 LLM_PROVIDER 값으로 선택합니다.
  1) mock  : API Key 없이 바로 실습 가능한 교육용 Mock LLM
  2) local : Ollama 기반 Local LLM
  3) cloud : Gemini, OpenAI, Anthropic, NVIDIA 같은 Cloud LLM
- 실제 Mock LLM 응답 생성은 llm/mock_llm.py에 위임합니다.
- 실제 Local LLM 호출은 llm/local_llm.py에 위임합니다.
- 실제 Cloud LLM 호출은 llm/cloud_llm.py에 위임합니다.
- Local LLM 또는 Cloud LLM 호출 실패 시 수업이 멈추지 않도록
  기본적으로 Mock LLM으로 fallback합니다.
- Day5 최종 리포트나 Day4 Tool Selection처럼 실제 LLM 호출 성공 여부를
  구분해야 하는 경우에는 generate_response(prompt, allow_fallback=False) 또는
  generate_json_response(prompt, allow_fallback=False)를 사용합니다.

보안:
- API Key를 코드에 직접 작성하지 않습니다.
- API Key 값을 절대 출력하지 않습니다.
- API Key 앞자리/뒷자리 일부도 출력하지 않습니다.
- 전체 환경변수 목록을 출력하지 않습니다.
- .env 파일은 GitHub에 올리면 안 됩니다.
- .env.example 파일만 GitHub에 올리는 구조를 전제로 합니다.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Final, Optional


VALID_LLM_PROVIDERS: Final[set[str]] = {"mock", "local", "cloud"}
DEFAULT_LLM_PROVIDER: Final[str] = "mock"
DEFAULT_NVIDIA_MODEL_NAME: Final[str] = "mistralai/mistral-nemotron"
DEFAULT_NVIDIA_BASE_URL: Final[str] = "https://integrate.api.nvidia.com/v1"
DEFAULT_OPENAI_MODEL_NAME: Final[str] = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL_NAME: Final[str] = "gemini-2.0-flash"
DEFAULT_ANTHROPIC_MODEL_NAME: Final[str] = "claude-3-5-haiku-latest"

LONG_PROMPT_WARNING_LENGTH: Final[int] = 8000
ERROR_PREVIEW_LENGTH: Final[int] = 300
RAW_RESPONSE_PREVIEW_LENGTH: Final[int] = 500

TRUE_VALUES: Final[set[str]] = {"1", "true", "yes", "y", "on"}
FALSE_VALUES: Final[set[str]] = {"0", "false", "no", "n", "off"}

CLOUD_PROVIDER_ALIASES: Final[set[str]] = {
    "gemini",
    "google",
    "openai",
    "anthropic",
    "claude",
    "nvidia",
    "nvidia_nim",
    "nvidia-nim",
    "nim",
}
LOCAL_PROVIDER_ALIASES: Final[set[str]] = {"ollama"}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def generate_response(
    prompt: str,
    allow_fallback: Optional[bool] = None,
    require_real_llm: bool = False,
) -> str:
    """
    Agent 코드에서 호출하는 공통 LLM 호출 함수입니다.

    동작 규칙:
    - LLM_PROVIDER=mock: Mock LLM을 사용합니다.
    - LLM_PROVIDER=local: Local LLM 호출을 먼저 시도합니다.
      실패하면 allow_fallback 설정에 따라 Mock LLM으로 전환하거나 예외를 발생시킵니다.
    - LLM_PROVIDER=cloud: Cloud LLM 호출을 먼저 시도합니다.
      실패하면 allow_fallback 설정에 따라 Mock LLM으로 전환하거나 예외를 발생시킵니다.
    - LLM_PROVIDER 환경변수가 없거나 알 수 없는 값이면 기본적으로 Mock LLM을 사용합니다.
      단, allow_fallback=False이면 잘못된 설정을 예외로 전달합니다.

    Args:
        prompt: LLM에 전달할 프롬프트 문자열
        allow_fallback: True이면 Local/Cloud 실패 시 Mock LLM으로 전환합니다.
            False이면 실패를 호출자에게 전달합니다. None이면 .env의
            LLM_ALLOW_MOCK_FALLBACK 값을 읽고, 값이 없으면 True를 사용합니다.
        require_real_llm: True이면 LLM_PROVIDER=mock을 허용하지 않습니다.
            Day5에서 실제 NVIDIA mistral-nemotron 호출 여부를 엄격히 확인할 때 사용할 수 있습니다.

    Returns:
        LLM 또는 Mock LLM이 생성한 응답 문자열

    Raises:
        ValueError: prompt가 문자열이 아니거나 비어 있는 경우, 또는 엄격 모드에서 설정이 잘못된 경우
        RuntimeError: Local/Cloud LLM 호출 실패를 fallback 없이 호출자에게 전달해야 하는 경우
    """
    _validate_prompt(prompt)

    fallback_enabled = _resolve_allow_fallback(allow_fallback)
    provider = get_llm_provider(strict=not fallback_enabled)

    if provider == "mock":
        if require_real_llm:
            raise RuntimeError(
                "실제 LLM 호출이 필요한 모드이지만 LLM_PROVIDER=mock으로 설정되어 있습니다. "
                "mistral-nemotron을 사용하려면 LLM_PROVIDER=cloud, CLOUD_LLM_PROVIDER=nvidia로 설정하세요."
            )
        return _generate_with_mock(prompt)

    if provider == "local":
        print(f"[LLM] {get_active_llm_label()}을 사용합니다.")
        try:
            return _generate_with_local(prompt)
        except Exception as error:
            safe_reason = _safe_error_message(error)
            if not fallback_enabled:
                raise RuntimeError(f"Local LLM 호출 실패: {safe_reason}") from error
            return _fallback_to_mock(
                prompt=prompt,
                reason=safe_reason,
                failed_provider="Local LLM",
            )

    if provider == "cloud":
        print(f"[LLM] {get_active_llm_label()}을 사용합니다.")
        try:
            return _generate_with_cloud(prompt)
        except Exception as error:
            safe_reason = _safe_error_message(error)
            if not fallback_enabled:
                raise RuntimeError(f"Cloud LLM 호출 실패: {safe_reason}") from error
            return _fallback_to_mock(
                prompt=prompt,
                reason=safe_reason,
                failed_provider="Cloud LLM",
            )

    # get_llm_provider에서 대부분 mock 또는 예외로 정리되지만,
    # 예외적인 상황에서도 수업이 멈추지 않도록 마지막 방어 코드를 둡니다.
    if not fallback_enabled:
        raise RuntimeError(f"지원하지 않는 LLM_PROVIDER 값입니다: {provider}")

    print("[LLM] 알 수 없는 LLM_PROVIDER 값이 감지되어 Mock LLM fallback을 사용합니다.")
    return _generate_with_mock(prompt)


def generate_json_response(
    prompt: str,
    allow_fallback: Optional[bool] = None,
    require_real_llm: bool = False,
) -> dict[str, Any]:
    """
    LLM 응답을 JSON dict로 반환합니다.

    내부적으로 generate_response(prompt)를 사용합니다. 따라서 .env의 LLM_PROVIDER 값에 따라
    mock/local/cloud가 자동 선택됩니다.

    이 함수는 두 종류의 실패를 분리합니다.
    - LLM 호출 실패: LLM_CALL_FAILED, LLM_TEXT_EXTRACTION_FAILED,
      LLM_RESPONSE_EMPTY, LLM_PROVIDER_ERROR, LLM_API_ERROR 등으로 반환합니다.
    - LLM 응답 파싱 실패: LLM_JSON_PARSE_FAILED로 반환합니다.

    JSON 파싱에 실패하거나 LLM 호출 자체가 실패해도 프로그램을 중단하지 않고,
    실패 사유와 원본 응답 일부를 담은 dict를 반환합니다.
    Day4 Tool Selection에서는 이 dict의 llm_error_code 값을 보고 rule-based fallback을 수행할 수 있습니다.

    중요:
    JSON 파싱 실패는 보안 Guardrail 차단이 아닙니다.
    따라서 guardrail_result에 LLM_JSON_PARSE_FAILED를 넣지 않습니다.
    호출하는 쪽에서는 llm_error_code 값을 보고 fallback하면 됩니다.

    사용 권장:
    - 일반 자연어 응답: generate_response(prompt)
    - JSON Tool Plan 생성: generate_json_response(prompt, allow_fallback=False)
      이렇게 호출하면 Cloud/Local LLM 실패가 Mock 응답으로 숨겨지지 않습니다.
    """
    try:
        raw_response = generate_response(
            prompt,
            allow_fallback=allow_fallback,
            require_real_llm=require_real_llm,
        )
    except Exception as error:
        safe_message = _safe_error_message(error)
        error_code = _classify_llm_error_message(safe_message)

        print("[LLM] JSON 응답용 LLM 호출에 실패했습니다.")
        print(f"[LLM] 실패 유형: {error_code}")
        print(f"[LLM] 실패 이유: {safe_message}")

        return {
            "llm_error_code": error_code,
            "tool_plan": [],
            "raw_response_preview": "",
            "error_message": safe_message,
            "llm_error_message": safe_message,
        }

    try:
        return extract_json_from_text(raw_response)
    except Exception as error:
        safe_message = _safe_error_message(error)
        print("[LLM] JSON 파싱에 실패했습니다.")
        print(f"[LLM] 실패 이유: {safe_message}")
        return {
            "llm_error_code": "LLM_JSON_PARSE_FAILED",
            "tool_plan": [],
            "raw_response_preview": _safe_text_preview(raw_response, RAW_RESPONSE_PREVIEW_LENGTH),
            "error_message": safe_message,
            "llm_error_message": safe_message,
        }


def extract_json_from_text(text: str) -> dict[str, Any]:
    """
    LLM 응답 문자열에서 JSON 객체만 추출해 dict로 변환합니다.

    지원하는 형태:
    - 순수 JSON 객체: {"tool_plan": [...]}
    - Markdown 코드블록: ```json ... ```
    - 설명 문장 앞뒤에 JSON이 섞인 응답
    - JSON 배열만 반환된 응답: [...] -> {"tool_plan": [...]} 형태로 정규화

    주의:
    - 이 함수는 입력 파일 인코딩 문제가 아니라 LLM 출력 형식 문제를 다룹니다.
    - JSON key에는 반드시 큰따옴표가 필요합니다.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("LLM 응답이 비어 있어 JSON을 추출할 수 없습니다.")

    cleaned = _remove_markdown_code_fence(text.strip())
    candidates = _collect_json_candidates(text, cleaned)
    errors: list[str] = []

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            return _normalize_json_result(parsed)
        except Exception as exc:
            errors.append(str(exc))

    # 후보 substring이 실패하면 JSONDecoder로 문자열 중간의 첫 유효 JSON을 탐색합니다.
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
            return _normalize_json_result(parsed)
        except Exception as exc:
            errors.append(str(exc))

    if errors:
        raise ValueError(f"LLM 응답에서 유효한 JSON을 찾지 못했습니다. 마지막 오류: {errors[-1]}")
    raise ValueError("LLM 응답에서 JSON 객체를 찾지 못했습니다.")


def get_llm_provider(strict: bool = False) -> str:
    """
    LLM_PROVIDER 환경변수를 읽고 사용할 LLM 실행 방식을 결정합니다.

    권장 허용값:
    - mock
    - local
    - cloud

    교육 중 자주 발생하는 설정 실수를 줄이기 위해 다음 alias도 처리합니다.
    - LLM_PROVIDER=ollama  -> local
    - LLM_PROVIDER=nvidia  -> cloud + CLOUD_LLM_PROVIDER=nvidia
    - LLM_PROVIDER=openai  -> cloud + CLOUD_LLM_PROVIDER=openai
    - LLM_PROVIDER=gemini  -> cloud + CLOUD_LLM_PROVIDER=gemini
    - LLM_PROVIDER=claude  -> cloud + CLOUD_LLM_PROVIDER=anthropic

    Args:
        strict: True이면 LLM_PROVIDER가 없거나 잘못된 경우 예외를 발생시킵니다.
            False이면 Mock LLM으로 fallback합니다. alias는 strict=True에서도 정상 설정으로 처리합니다.

    Returns:
        "mock", "local", "cloud" 중 하나
    """
    _load_env_if_available()

    raw_provider = os.getenv("LLM_PROVIDER", "")
    provider = raw_provider.strip().lower()

    if provider in VALID_LLM_PROVIDERS:
        return provider

    if provider in LOCAL_PROVIDER_ALIASES:
        print(
            f"[LLM] 경고: LLM_PROVIDER='{provider}' 값은 alias입니다. "
            "LLM_PROVIDER=local로 해석합니다."
        )
        return "local"

    if provider in CLOUD_PROVIDER_ALIASES:
        normalized_cloud_provider = _normalize_cloud_provider(provider)
        os.environ["CLOUD_LLM_PROVIDER"] = normalized_cloud_provider
        print(
            f"[LLM] 경고: LLM_PROVIDER='{provider}' 값은 권장 형식이 아닙니다. "
            f"LLM_PROVIDER=cloud, CLOUD_LLM_PROVIDER={normalized_cloud_provider}로 해석합니다."
        )
        return "cloud"

    if not provider:
        if strict:
            raise ValueError(
                "LLM_PROVIDER 환경변수가 설정되어 있지 않습니다. "
                "mock, local, cloud 중 하나를 설정해 주세요."
            )
        return DEFAULT_LLM_PROVIDER

    if strict:
        raise ValueError(
            f"지원하지 않는 LLM_PROVIDER 값입니다: {provider}. "
            "mock, local, cloud 중 하나를 사용해 주세요."
        )

    print(
        f"[LLM] 경고: LLM_PROVIDER='{provider}' 값은 지원하지 않습니다. "
        "Mock LLM으로 전환합니다."
    )
    return DEFAULT_LLM_PROVIDER


def get_active_llm_label() -> str:
    """
    현재 설정 기준으로 사용 예정인 LLM 경로를 사람이 읽기 쉬운 문자열로 반환합니다.

    Returns:
        예: "Mock LLM", "Local LLM (llama3.1)",
            "Cloud LLM (nvidia / mistralai/mistral-nemotron)"
    """
    _load_env_if_available()

    provider = get_llm_provider(strict=False)

    if provider == "mock":
        return "Mock LLM"

    if provider == "local":
        model_name = os.getenv("OLLAMA_MODEL_NAME", "").strip()
        return f"Local LLM ({model_name})" if model_name else "Local LLM"

    if provider == "cloud":
        cloud_provider = _normalize_cloud_provider(os.getenv("CLOUD_LLM_PROVIDER", ""))
        model_name = _get_cloud_model_name_for_label(cloud_provider)
        if cloud_provider and model_name:
            return f"Cloud LLM ({cloud_provider} / {model_name})"
        if cloud_provider:
            return f"Cloud LLM ({cloud_provider})"
        return "Cloud LLM"

    return "Mock LLM fallback"


def get_llm_runtime_info() -> dict[str, Any]:
    """
    Trace, 디버깅, 강의 운영용 LLM 런타임 정보를 반환합니다.

    API Key 값은 절대 반환하지 않고, 설정 여부만 boolean으로 제공합니다.
    """
    _load_env_if_available()

    provider = get_llm_provider(strict=False)
    cloud_provider = _normalize_cloud_provider(os.getenv("CLOUD_LLM_PROVIDER", ""))

    runtime_info: dict[str, Any] = {
        "llm_provider": provider,
        "active_llm_label": get_active_llm_label(),
        "mock_fallback_enabled": _resolve_allow_fallback(None),
    }

    if provider == "local":
        runtime_info.update(
            {
                "local_llm_provider": os.getenv("LOCAL_LLM_PROVIDER", "").strip(),
                "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "").strip(),
                "ollama_model_name": os.getenv("OLLAMA_MODEL_NAME", "").strip(),
            }
        )

    if provider == "cloud":
        runtime_info.update(
            {
                "cloud_llm_provider": cloud_provider,
                "cloud_model_name": _get_cloud_model_name_for_label(cloud_provider),
                "nvidia_base_url": os.getenv("NVIDIA_BASE_URL", "").strip() or DEFAULT_NVIDIA_BASE_URL,
                "nvidia_api_key_configured": bool(os.getenv("NVIDIA_API_KEY", "").strip()),
                "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
                "gemini_api_key_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
                "anthropic_api_key_configured": bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
            }
        )

    return runtime_info


# -----------------------------------------------------------------------------
# Provider dispatch
# -----------------------------------------------------------------------------


def _generate_with_local(prompt: str) -> str:
    """
    Local LLM 호출부입니다.

    실제 Ollama 기반 Local LLM 호출 코드는 llm/local_llm.py에 위임합니다.
    이 함수는 mock fallback을 직접 수행하지 않습니다.
    예외를 generate_response로 전달하여 fallback 흐름을 한 곳에서 관리합니다.
    """
    try:
        from llm.local_llm import generate_local_llm_response
    except ImportError as exc:
        raise ImportError(
            "llm.local_llm 모듈을 import하지 못했습니다. "
            "프로젝트 루트에서 실행하고 있는지 확인해 주세요. "
            "src/llm/local_llm.py 파일이 존재해야 합니다."
        ) from exc

    return generate_local_llm_response(prompt)


def _generate_with_cloud(prompt: str) -> str:
    """
    Cloud LLM 호출부입니다.

    실제 API 호출 코드는 llm/cloud_llm.py에 위임합니다.
    Gemini, OpenAI, Anthropic, NVIDIA 중 어떤 Cloud LLM을 사용할지는
    llm/cloud_llm.py에서 CLOUD_LLM_PROVIDER 값을 기준으로 처리합니다.
    """
    try:
        from llm.cloud_llm import generate_cloud_llm_response
    except ImportError as exc:
        raise ImportError(
            "llm.cloud_llm 모듈을 import하지 못했습니다. "
            "프로젝트 루트에서 실행하고 있는지 확인해 주세요. "
            "src/llm/cloud_llm.py 파일이 존재해야 합니다."
        ) from exc

    return generate_cloud_llm_response(prompt)


def _generate_with_mock(prompt: str) -> str:
    """
    Mock LLM 호출부입니다.

    실제 Mock 응답 생성 코드는 llm/mock_llm.py에 위임합니다.
    Mock LLM은 API Key 없이 실행되므로 수강생 실습의 기본값으로 사용합니다.
    """
    print("[LLM] Mock LLM을 사용합니다.")

    try:
        from llm.mock_llm import generate_mock_response
    except ImportError as exc:
        raise ImportError(
            "llm.mock_llm 모듈을 import하지 못했습니다. "
            "프로젝트 루트에서 실행하고 있는지 확인해 주세요. "
            "src/llm/mock_llm.py 파일이 존재해야 합니다."
        ) from exc

    return generate_mock_response(prompt)


def _fallback_to_mock(prompt: str, reason: str, failed_provider: str = "LLM") -> str:
    """
    Local LLM 또는 Cloud LLM 호출 실패 시 Mock LLM으로 자동 전환합니다.
    """
    print(f"[LLM] {failed_provider} 호출에 실패하여 Mock LLM으로 전환합니다.")
    if reason:
        print(f"[LLM] 전환 이유: {reason}")

    return _generate_with_mock(prompt)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _load_env_if_available() -> None:
    """
    python-dotenv가 설치되어 있으면 .env 파일을 자동으로 읽습니다.

    Windows 메모장 저장 파일을 고려해 utf-8-sig 인코딩을 사용합니다.
    python-dotenv가 설치되어 있지 않으면 오류 없이 넘어갑니다.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_path = Path.cwd() / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, encoding="utf-8-sig")
    else:
        load_dotenv(encoding="utf-8-sig")


def _validate_prompt(prompt: str) -> None:
    """
    프롬프트가 올바른 문자열인지 확인합니다.

    너무 긴 prompt는 차단하지 않고 경고만 출력합니다.
    수업 중 긴 프롬프트를 실험할 수 있기 때문입니다.
    """
    if not isinstance(prompt, str):
        raise ValueError("prompt는 문자열(str)이어야 합니다.")

    if not prompt.strip():
        raise ValueError("prompt가 비어 있습니다. 교육용 질문 문장을 입력해 주세요.")

    if len(prompt) > LONG_PROMPT_WARNING_LENGTH:
        print(
            "[LLM] 경고: prompt가 긴 편입니다. "
            "RAG/Trace/Tool 결과가 너무 많이 포함되어 있으면 입력을 요약해 주세요."
        )


def _resolve_allow_fallback(allow_fallback: Optional[bool]) -> bool:
    """
    함수 인자와 환경변수 기준으로 Mock fallback 허용 여부를 결정합니다.

    우선순위:
    1. 함수 인자 allow_fallback
    2. LLM_DISABLE_MOCK_FALLBACK=true이면 비활성화
    3. LLM_ALLOW_MOCK_FALLBACK 또는 LLM_FALLBACK_TO_MOCK
    4. 기본값 True
    """
    _load_env_if_available()

    if isinstance(allow_fallback, bool):
        return allow_fallback

    disable_value = os.getenv("LLM_DISABLE_MOCK_FALLBACK", "").strip().lower()
    if disable_value in TRUE_VALUES:
        return False

    for env_name in ("LLM_ALLOW_MOCK_FALLBACK", "LLM_FALLBACK_TO_MOCK"):
        raw_value = os.getenv(env_name, "").strip().lower()
        if raw_value in TRUE_VALUES:
            return True
        if raw_value in FALSE_VALUES:
            return False

    # 강의 운영 안정성을 위해 기본값은 fallback 허용입니다.
    return True


def _normalize_cloud_provider(provider: str) -> str:
    """CLOUD_LLM_PROVIDER alias를 사람이 읽기 쉬운 표준값으로 정규화합니다."""
    normalized = (provider or "").strip().lower()
    if normalized in {"nvidia", "nvidia_nim", "nvidia-nim", "nim"}:
        return "nvidia"
    if normalized == "google":
        return "gemini"
    if normalized == "claude":
        return "anthropic"
    return normalized


def _get_cloud_model_name_for_label(cloud_provider: str) -> str:
    """Cloud provider별 모델명을 label/runtime info용으로 반환합니다."""
    provider = _normalize_cloud_provider(cloud_provider)

    if provider == "nvidia":
        return os.getenv("NVIDIA_MODEL_NAME", "").strip() or DEFAULT_NVIDIA_MODEL_NAME
    if provider == "openai":
        return os.getenv("OPENAI_MODEL_NAME", "").strip() or DEFAULT_OPENAI_MODEL_NAME
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL_NAME", "").strip() or DEFAULT_GEMINI_MODEL_NAME
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL_NAME", "").strip() or DEFAULT_ANTHROPIC_MODEL_NAME

    return ""


def _remove_markdown_code_fence(text: str) -> str:
    """Markdown JSON 코드블록 표기를 제거합니다."""
    cleaned = text.strip()

    # 응답 전체가 코드블록이면 fence만 제거합니다.
    cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _collect_json_candidates(original_text: str, cleaned_text: str) -> list[str]:
    """LLM 응답에서 JSON으로 파싱해 볼 후보 문자열을 수집합니다."""
    candidates: list[str] = []

    # 1) fenced code block 안의 JSON 후보
    for match in re.finditer(r"```(?:json|JSON)?\s*(.*?)```", original_text, flags=re.DOTALL):
        block = match.group(1).strip()
        if block:
            candidates.append(block)

    # 2) 정리된 전체 문자열
    candidates.append(cleaned_text)

    # 3) JSON object/array substring 후보
    # 배열이 객체보다 먼저 시작하는 경우([{"tool": ...}])에는 array 후보를 먼저 둡니다.
    # 그렇지 않으면 object 후보를 먼저 둡니다.
    object_start = cleaned_text.find("{")
    object_end = cleaned_text.rfind("}")
    array_start = cleaned_text.find("[")
    array_end = cleaned_text.rfind("]")

    array_candidate = ""
    object_candidate = ""

    if array_start != -1 and array_end != -1 and array_start < array_end:
        array_candidate = cleaned_text[array_start : array_end + 1]

    if object_start != -1 and object_end != -1 and object_start < object_end:
        object_candidate = cleaned_text[object_start : object_end + 1]

    if array_candidate and (object_start == -1 or array_start < object_start):
        candidates.append(array_candidate)
        if object_candidate:
            candidates.append(object_candidate)
    else:
        if object_candidate:
            candidates.append(object_candidate)
        if array_candidate:
            candidates.append(array_candidate)

    # 순서 유지 중복 제거
    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    return unique_candidates


def _normalize_json_result(parsed: Any) -> dict[str, Any]:
    """JSON 파싱 결과를 dict 형태로 정규화합니다."""
    if isinstance(parsed, dict):
        return parsed

    if isinstance(parsed, list):
        return {"tool_plan": parsed}

    raise ValueError(f"JSON 최상위 타입이 dict/list가 아닙니다: {type(parsed).__name__}")




def _classify_llm_error_message(message: str) -> str:
    """
    LLM 호출 실패 메시지를 보고 보고서에 남길 오류 코드를 분류합니다.

    이 함수는 보안 처리가 끝난 안전한 오류 메시지를 대상으로 동작합니다.
    반환값은 Day4 Tool Selector가 fallback 사유를 구분할 때 사용합니다.
    """
    if not message:
        return "LLM_CALL_FAILED"

    lowered = message.lower()

    text_extraction_keywords = [
        "텍스트를 추출하지 못",
        "message.content",
        "content가 비어",
        "content is empty",
        "text extraction",
        "extract text",
        "choices가 비어",
        "choices is empty",
    ]
    if any(keyword.lower() in lowered for keyword in text_extraction_keywords):
        return "LLM_TEXT_EXTRACTION_FAILED"

    empty_response_keywords = [
        "응답이 비어",
        "비어 있어",
        "empty response",
        "response empty",
        "blank response",
        "no response",
    ]
    if any(keyword.lower() in lowered for keyword in empty_response_keywords):
        return "LLM_RESPONSE_EMPTY"

    provider_config_keywords = [
        "llm_provider",
        "cloud_llm_provider",
        "지원하지 않는",
        "환경변수가 설정되어 있지",
        "잘못된 설정",
        "provider",
    ]
    if any(keyword.lower() in lowered for keyword in provider_config_keywords):
        return "LLM_PROVIDER_ERROR"

    api_keywords = [
        "api key",
        "api_key",
        "api 호출",
        "호출 중 오류",
        "사용 권한",
        "permission",
        "unauthorized",
        "forbidden",
        "401",
        "403",
        "429",
        "rate limit",
        "quota",
        "network",
        "connection",
        "timeout",
        "http",
        "ssl",
        "tls",
    ]
    if any(keyword.lower() in lowered for keyword in api_keywords):
        return "LLM_API_ERROR"

    dependency_keywords = [
        "import하지 못했습니다",
        "패키지가 설치되어 있지",
        "module not found",
        "no module named",
        "importerror",
    ]
    if any(keyword.lower() in lowered for keyword in dependency_keywords):
        return "LLM_PROVIDER_ERROR"

    return "LLM_CALL_FAILED"


def _safe_text_preview(text: str, max_length: int) -> str:
    """응답 미리보기에서 민감정보처럼 보이는 값을 마스킹합니다."""
    if not isinstance(text, str):
        return ""

    masked = _mask_sensitive_text(text)
    if len(masked) > max_length:
        return masked[:max_length].rstrip() + "..."
    return masked


def _mask_sensitive_text(text: str) -> str:
    """API Key, Token, Bearer 인증값 후보를 마스킹합니다."""
    masked = text
    masked = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"\1[MASKED_SECRET]",
        masked,
    )
    masked = re.sub(
        r"(?i)(bearer\s+)[A-Za-z0-9_\-\.]{16,}",
        r"\1[MASKED_SECRET]",
        masked,
    )
    masked = re.sub(
        r"(?i)(api[_-]?key|token|secret|password|credential)\s*[:=]\s*[^\s,;]+",
        r"\1=[MASKED_SECRET]",
        masked,
    )
    masked = re.sub(r"[A-Za-z0-9_\-\.]{24,}", "[MASKED_SECRET]", masked)
    return masked


def _safe_error_message(error: Exception) -> str:
    """
    예외 메시지를 수업용으로 안전하게 정리합니다.

    보안 처리:
    - API Key처럼 보이는 긴 문자열을 마스킹합니다.
    - Authorization Bearer 토큰 형태를 마스킹합니다.
    - 전체 환경변수 목록이나 민감정보가 노출되지 않도록 메시지를 짧게 자릅니다.
    """
    raw_message = str(error) if error else "알 수 없는 오류가 발생했습니다."
    masked = _mask_sensitive_text(raw_message)

    # 너무 긴 오류 메시지는 수업 진행에 방해되므로 앞부분만 보여줍니다.
    if len(masked) > ERROR_PREVIEW_LENGTH:
        masked = masked[:ERROR_PREVIEW_LENGTH].rstrip() + "..."

    return masked


if __name__ == "__main__":
    test_prompt = (
        "DisplayEdu Fab의 EDU-LINE-07에서 EQP-EV-03 설비에 "
        "ALM-TEMP-402 교육용 알람이 반복 발생했습니다. "
        "교육용 로그와 매뉴얼을 참고한다는 가정으로 "
        "원인 후보와 1차 확인 항목을 Markdown으로 정리해 주세요."
    )

    print("=" * 80)
    print("LLM Client 단독 실행 테스트")
    print("=" * 80)
    print(f"현재 설정된 LLM 경로: {get_active_llm_label()}")
    print("주의: LLM_PROVIDER=mock이면 API Key 없이 Mock 응답을 사용합니다.")
    print("주의: LLM_PROVIDER=local이면 Local LLM 호출을 시도하고 실패 시 Mock으로 전환합니다.")
    print("주의: LLM_PROVIDER=cloud이면 Cloud LLM 호출을 시도하고 실패 시 Mock으로 전환합니다.")
    print("주의: NVIDIA 설정 시 CLOUD_LLM_PROVIDER=nvidia와 NVIDIA_MODEL_NAME 환경변수를 사용합니다.")
    print(f"Mock fallback 허용 여부: {_resolve_allow_fallback(None)}")
    print("-" * 80)

    try:
        result = generate_response(test_prompt)
        preview_length = 1000
        print(result[:preview_length])

        if len(result) > preview_length:
            print("\n... [응답 미리보기 생략: 전체 응답은 함수 반환값에서 확인할 수 있습니다.]")

    except Exception as exc:
        print("LLM Client 실행 중 오류가 발생했습니다.")
        print(_safe_error_message(exc))
        print(
            "프로젝트 루트에서 실행 중인지, "
            "src/llm/mock_llm.py, src/llm/local_llm.py, src/llm/cloud_llm.py 파일이 있는지 확인해 주세요."
        )
