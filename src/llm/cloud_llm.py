"""
cloud_llm.py

AI Agent Architecture 1~5일차 실습용 Cloud LLM 호출 모듈입니다.

역할:
- Cloud LLM 호출부를 한 곳에 모읍니다.
- CLOUD_LLM_PROVIDER 값에 따라 Gemini, OpenAI API, Anthropic API, NVIDIA API 중 하나를 호출합니다.
- NVIDIA API Catalog / NIM의 OpenAI-compatible endpoint를 통해 환경변수에 지정한 모델을 사용할 수 있습니다.
- Agent 본체는 특정 LLM SDK나 제품명에 직접 의존하지 않도록 설계합니다.
- llm_client.py는 LLM_PROVIDER=cloud일 때 이 파일의
  generate_cloud_llm_response(prompt) 함수만 호출합니다.
- mock fallback은 이 파일에서 수행하지 않습니다.
  실패 시 예외를 발생시키고, fallback은 상위 모듈인 llm_client.py가 처리합니다.

Cloud LLM 진단 Trace:
- NVIDIA/OpenAI-compatible ChatCompletion 호출의 request/response/extraction/error event를
  JSONL로 남길 수 있습니다.
- 기본 로그 경로는 outputs/day4/cloud_llm_diagnostics.jsonl 입니다.
- API Key, Bearer token, password, secret, token_ids 계열은 로그에 남기지 않습니다.
- max_tokens, prompt_tokens, completion_tokens, total_tokens 같은 token count는 진단에 필요하므로
  마스킹하지 않습니다.

보안:
- API Key를 코드에 직접 작성하지 않습니다.
- API Key 값은 Provider별 환경변수에서만 읽습니다.
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
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Final


DEFAULT_CLOUD_PROVIDER: Final[str] = "gemini"
VALID_CLOUD_PROVIDERS: Final[set[str]] = {"gemini", "openai", "anthropic", "nvidia"}
NVIDIA_PROVIDER_ALIASES: Final[set[str]] = {"nvidia_nim", "nvidia-nim", "nim"}

DEFAULT_GEMINI_MODEL_NAME: Final[str] = "gemini-2.0-flash"
DEFAULT_OPENAI_MODEL_NAME: Final[str] = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL_NAME: Final[str] = "claude-3-5-haiku-latest"
DEFAULT_NVIDIA_MODEL_NAME: Final[str] = "mistralai/mistral-nemotron"
DEFAULT_NVIDIA_BASE_URL: Final[str] = "https://integrate.api.nvidia.com/v1"

# 교육 실습에서는 안정적이고 일관된 응답을 위해 낮은 temperature를 기본값으로 둡니다.
DEFAULT_TEMPERATURE: Final[float] = 0.2

# NVIDIA 예제와 맞춘 기본 top_p입니다.
DEFAULT_TOP_P: Final[float] = 0.7

# 5일차 최종 리포트가 중간에 끊기지 않도록 기본 길이를 4096으로 둡니다.
DEFAULT_MAX_OUTPUT_TOKENS: Final[int] = 4096

ERROR_PREVIEW_LENGTH: Final[int] = 300

DEFAULT_DIAGNOSTICS_LOG_PATH: Final[str] = "outputs/llm/cloud_llm_diagnostics.jsonl"
DEFAULT_DIAGNOSTICS_PREVIEW_CHARS: Final[int] = 500
DEFAULT_DIAGNOSTICS_MAX_CHOICES: Final[int] = 3

DEFAULT_NVIDIA_SYSTEM_PROMPT: Final[str] = (
    "너는 제조 장애 대응 AI Agent다. "
    "제공된 근거 안에서만 판단하고, 근거 없는 원인 단정은 하지 않는다. "
    "실제 사내 시스템 접근, 민감정보 노출, 확인되지 않은 수치 생성은 하지 않는다. "
    "응답은 한국어로 작성한다."
)

LOGIN_PROVIDER_LABELS: Final[dict[str, str]] = {
    "gemini": "Gemini",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "nvidia": "NVIDIA",
}

# 수치형 token count는 진단에 필요하므로 마스킹하지 않습니다.
SAFE_TOKEN_COUNT_KEYS: Final[set[str]] = {
    "max_tokens",
    "max_output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "audio_tokens",
    "accepted_prediction_tokens",
    "rejected_prediction_tokens",
}

SENSITIVE_EXACT_KEYS: Final[set[str]] = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "access_token",
    "refresh_token",
    "password",
    "passwd",
    "secret",
    "client_secret",
    "token",
    "id_token",
    "session_token",
    "token_ids",
    "input_token_ids",
    "output_token_ids",
}


class CloudLLMError(RuntimeError):
    """Cloud LLM 호출 또는 응답 처리 오류를 구조화해서 전달하는 예외입니다."""

    def __init__(self, error_code: str, message: str, diagnostic: dict[str, Any] | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.diagnostic = diagnostic or {}


def generate_cloud_llm_response(prompt: str, trace_context: dict[str, Any] | None = None) -> str:
    """
    환경변수에 설정된 Cloud LLM Provider를 사용하여 응답을 생성합니다.

    지원 Provider:
    - gemini: Google Gemini API
    - openai: OpenAI API
    - anthropic: Anthropic API
    - nvidia: NVIDIA API Catalog / NIM OpenAI-compatible API

    Args:
        prompt: Cloud LLM에 전달할 프롬프트 문자열
        trace_context: 선택 입력입니다. case_id, user_query 등 호출 맥락을 diagnostic event에 함께 남길 수 있습니다.

    Returns:
        Cloud LLM이 생성한 응답 문자열

    Raises:
        ValueError: prompt가 비어 있거나 provider/API Key 설정이 잘못된 경우
        RuntimeError: SDK 설치 누락 또는 API 호출 실패 시
    """
    _load_env_if_available()
    _validate_prompt(prompt)

    provider = _get_cloud_provider()

    if provider == "gemini":
        return _generate_with_gemini(prompt)

    if provider == "openai":
        return _generate_with_openai(prompt, trace_context=trace_context)

    if provider == "anthropic":
        return _generate_with_anthropic(prompt)

    if provider == "nvidia":
        return _generate_with_nvidia(prompt, trace_context=trace_context)

    # _get_cloud_provider에서 대부분 차단되지만, 수업 중 예외 상황에 대비한 방어 코드입니다.
    raise ValueError(
        "지원하지 않는 CLOUD_LLM_PROVIDER 값입니다. "
        "gemini, openai, anthropic, nvidia 중 하나를 사용해 주세요."
    )


def _generate_with_gemini(prompt: str) -> str:
    """
    google-genai 패키지를 이용해 Gemini API를 호출합니다.

    주의:
    - 실제 외부 API를 호출합니다.
    - API Key 값은 출력하지 않습니다.
    - 호출 실패 시 mock fallback을 직접 수행하지 않고 예외를 발생시킵니다.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "google-genai 패키지가 설치되어 있지 않습니다. "
            "다음 명령으로 설치해 주세요: python -m pip install google-genai"
        ) from exc

    api_key = _get_api_key("gemini")
    model_name = _get_model_name("gemini")
    generation_config = _get_generation_config()

    try:
        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            temperature=generation_config["temperature"],
            max_output_tokens=generation_config["max_output_tokens"],
        )
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        return _clean_gemini_response_text(response)

    except Exception as exc:
        safe_message = _safe_short_error(exc)
        raise RuntimeError(
            "Gemini API 호출 중 오류가 발생했습니다. "
            "API Key 설정, 네트워크 연결, 모델명, 사용 권한을 확인해 주세요. "
            f"안전 처리된 오류: {safe_message}"
        ) from exc


def _generate_with_openai(prompt: str, trace_context: dict[str, Any] | None = None) -> str:
    """
    openai 패키지를 이용해 OpenAI API를 호출합니다.

    OPENAI_API_KEY와 OPENAI_MODEL_NAME 환경변수를 사용합니다.
    호출 실패 시 mock fallback을 직접 수행하지 않고 예외를 발생시킵니다.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai 패키지가 설치되어 있지 않습니다. "
            "다음 명령으로 설치해 주세요: python -m pip install openai"
        ) from exc

    api_key = _get_api_key("openai")
    model_name = _get_model_name("openai")
    generation_config = _get_generation_config()
    messages = _build_chat_messages(prompt)
    trace_id = _new_trace_id("openai")
    base_url = "https://api.openai.com/v1"

    _write_request_event(
        trace_id=trace_id,
        provider="openai",
        model_name=model_name,
        base_url=base_url,
        messages=messages,
        generation_config=generation_config,
        stream=False,
        response_format=None,
        trace_context=trace_context,
    )

    started_at = time.perf_counter()
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            max_tokens=generation_config["max_output_tokens"],
        )
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        _write_response_event(
            trace_id=trace_id,
            provider="openai",
            model_name=model_name,
            base_url=base_url,
            response=response,
            latency_ms=latency_ms,
            trace_context=trace_context,
        )
        return _clean_chat_completion_text(
            response,
            provider_label="OpenAI",
            model_name=model_name,
            base_url=base_url,
            trace_id=trace_id,
            trace_context=trace_context,
        )

    except Exception as exc:
        error_code = _get_cloud_error_code(exc)
        safe_message = _safe_short_error(exc)
        _write_error_event(
            trace_id=trace_id,
            provider="openai",
            model_name=model_name,
            base_url=base_url,
            error_code=error_code,
            error_message=safe_message,
            diagnostic=_get_exception_diagnostic(exc),
            trace_context=trace_context,
        )
        raise RuntimeError(
            "OpenAI API 호출 중 오류가 발생했습니다. "
            "API Key 설정, 네트워크 연결, 모델명, 사용 권한을 확인해 주세요. "
            f"안전 처리된 오류: {safe_message}"
        ) from exc


def _generate_with_anthropic(prompt: str) -> str:
    """
    anthropic 패키지를 이용해 Anthropic API를 호출합니다.

    ANTHROPIC_API_KEY와 ANTHROPIC_MODEL_NAME 환경변수를 사용합니다.
    호출 실패 시 mock fallback을 직접 수행하지 않고 예외를 발생시킵니다.
    """
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "anthropic 패키지가 설치되어 있지 않습니다. "
            "다음 명령으로 설치해 주세요: python -m pip install anthropic"
        ) from exc

    api_key = _get_api_key("anthropic")
    model_name = _get_model_name("anthropic")
    generation_config = _get_generation_config()

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_name,
            max_tokens=generation_config["max_output_tokens"],
            temperature=generation_config["temperature"],
            messages=[{"role": "user", "content": prompt}],
        )
        text = _clean_anthropic_response_text(response)
        if text:
            return text
        raise ValueError("Anthropic 응답에서 텍스트를 추출하지 못했습니다.")

    except Exception as exc:
        safe_message = _safe_short_error(exc)
        raise RuntimeError(
            "Anthropic API 호출 중 오류가 발생했습니다. "
            "API Key 설정, 네트워크 연결, 모델명, 사용 권한을 확인해 주세요. "
            f"안전 처리된 오류: {safe_message}"
        ) from exc


def _generate_with_nvidia(prompt: str, trace_context: dict[str, Any] | None = None) -> str:
    """
    NVIDIA API Catalog / NIM의 OpenAI-compatible Chat Completions API를 호출합니다.

    사용 환경변수:
    - NVIDIA_API_KEY: NVIDIA API Key
    - NVIDIA_BASE_URL: 기본값 https://integrate.api.nvidia.com/v1
    - NVIDIA_MODEL_NAME: 사용할 NVIDIA 모델명. 값이 없으면 DEFAULT_NVIDIA_MODEL_NAME을 사용합니다.

    호출 실패 시 mock fallback을 직접 수행하지 않고 예외를 발생시킵니다.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai 패키지가 설치되어 있지 않습니다. "
            "NVIDIA endpoint는 OpenAI-compatible API로 호출하므로 "
            "다음 명령으로 설치해 주세요: python -m pip install openai"
        ) from exc

    api_key = _get_api_key("nvidia")
    model_name = _get_model_name("nvidia")
    base_url = _get_nvidia_base_url()
    nvidia_system_prompt = os.getenv("NVIDIA_SYSTEM_PROMPT", "").strip() or DEFAULT_NVIDIA_SYSTEM_PROMPT
    generation_config = _get_generation_config()
    messages = _build_chat_messages(
        prompt,
        default_system_prompt=nvidia_system_prompt,
    )
    trace_id = _new_trace_id("nvidia")

    _write_request_event(
        trace_id=trace_id,
        provider="nvidia",
        model_name=model_name,
        base_url=base_url,
        messages=messages,
        generation_config=generation_config,
        stream=False,
        response_format=None,
        trace_context=trace_context,
    )

    started_at = time.perf_counter()
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            max_tokens=generation_config["max_output_tokens"],
            stream=False,
        )
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        _write_response_event(
            trace_id=trace_id,
            provider="nvidia",
            model_name=model_name,
            base_url=base_url,
            response=response,
            latency_ms=latency_ms,
            trace_context=trace_context,
        )
        return _clean_chat_completion_text(
            response,
            provider_label="NVIDIA",
            model_name=model_name,
            base_url=base_url,
            trace_id=trace_id,
            trace_context=trace_context,
        )

    except Exception as exc:
        error_code = _get_cloud_error_code(exc)
        safe_message = _safe_short_error(exc)
        _write_error_event(
            trace_id=trace_id,
            provider="nvidia",
            model_name=model_name,
            base_url=base_url,
            error_code=error_code,
            error_message=safe_message,
            diagnostic=_get_exception_diagnostic(exc),
            trace_context=trace_context,
        )
        raise RuntimeError(
            "NVIDIA API 호출 중 오류가 발생했습니다. "
            "NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL_NAME, 네트워크 연결, 모델 사용 권한을 확인해 주세요. "
            f"안전 처리된 오류: {safe_message}"
        ) from exc


def _get_cloud_provider() -> str:
    """
    CLOUD_LLM_PROVIDER 환경변수를 읽어 Cloud LLM Provider를 결정합니다.

    허용값:
    - gemini
    - openai
    - anthropic
    - nvidia
    """
    provider = os.getenv("CLOUD_LLM_PROVIDER", DEFAULT_CLOUD_PROVIDER).strip().lower()

    if provider in NVIDIA_PROVIDER_ALIASES:
        return "nvidia"

    if provider in VALID_CLOUD_PROVIDERS:
        return provider

    if provider == "claude":
        raise ValueError(
            "CLOUD_LLM_PROVIDER=claude는 사용하지 않습니다. "
            "Anthropic API를 사용하려면 CLOUD_LLM_PROVIDER=anthropic으로 설정해 주세요."
        )

    if provider == "local":
        raise ValueError(
            "CLOUD_LLM_PROVIDER=local은 잘못된 설정입니다. "
            "Local LLM은 LLM_PROVIDER=local로 설정하고 src/llm/local_llm.py에서 처리합니다."
        )

    raise ValueError(
        f"지원하지 않는 CLOUD_LLM_PROVIDER 값입니다: {provider}. "
        "gemini, openai, anthropic, nvidia 중 하나를 사용해 주세요."
    )


def _get_api_key(provider: str) -> str:
    """
    Provider별 API Key를 환경변수에서 읽습니다.

    사용 환경변수:
    - gemini: GEMINI_API_KEY
    - openai: OPENAI_API_KEY
    - anthropic: ANTHROPIC_API_KEY
    - nvidia: NVIDIA_API_KEY
    """
    env_name_by_provider = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
    }
    env_name = env_name_by_provider.get(provider)

    if not env_name:
        raise ValueError("지원하지 않는 Cloud LLM Provider입니다.")

    api_key = os.getenv(env_name, "").strip()
    if not api_key:
        provider_label = LOGIN_PROVIDER_LABELS.get(provider, provider)
        raise ValueError(
            f"{provider_label} API Key가 설정되어 있지 않습니다. "
            f".env 파일에 {env_name}= 값을 설정해 주세요. "
            "실제 API Key는 .env.example이 아니라 .env 파일에만 작성해야 합니다."
        )

    return api_key


def _get_model_name(provider: str) -> str:
    """Provider별 모델명을 환경변수에서 읽습니다."""
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL_NAME", "").strip() or DEFAULT_GEMINI_MODEL_NAME

    if provider == "openai":
        return os.getenv("OPENAI_MODEL_NAME", "").strip() or DEFAULT_OPENAI_MODEL_NAME

    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL_NAME", "").strip() or DEFAULT_ANTHROPIC_MODEL_NAME

    if provider == "nvidia":
        return os.getenv("NVIDIA_MODEL_NAME", "").strip() or DEFAULT_NVIDIA_MODEL_NAME

    raise ValueError("지원하지 않는 Cloud LLM Provider입니다.")


def _get_nvidia_base_url() -> str:
    """NVIDIA OpenAI-compatible endpoint base URL을 읽습니다."""
    base_url = os.getenv("NVIDIA_BASE_URL", "").strip() or DEFAULT_NVIDIA_BASE_URL
    return base_url.rstrip("/")


def _get_generation_config() -> Dict[str, Any]:
    """
    LLM 응답 생성 옵션을 환경변수에서 읽습니다.

    주 설정:
    - LLM_TEMPERATURE
    - LLM_TOP_P
    - LLM_MAX_OUTPUT_TOKENS

    하위 호환:
    - CLOUD_LLM_TEMPERATURE
    - CLOUD_LLM_TOP_P
    - CLOUD_LLM_MAX_OUTPUT_TOKENS
    """
    raw_temperature = (
        os.getenv("LLM_TEMPERATURE", "").strip()
        or os.getenv("CLOUD_LLM_TEMPERATURE", "").strip()
    )
    raw_top_p = (
        os.getenv("LLM_TOP_P", "").strip()
        or os.getenv("CLOUD_LLM_TOP_P", "").strip()
    )
    raw_max_tokens = (
        os.getenv("LLM_MAX_OUTPUT_TOKENS", "").strip()
        or os.getenv("CLOUD_LLM_MAX_OUTPUT_TOKENS", "").strip()
    )

    temperature = _parse_float_or_default(raw_temperature, DEFAULT_TEMPERATURE)
    top_p = _parse_float_or_default(raw_top_p, DEFAULT_TOP_P)
    max_output_tokens = _parse_int_or_default(raw_max_tokens, DEFAULT_MAX_OUTPUT_TOKENS)

    # API별 오류를 줄이기 위해 일반적인 범위로 정규화합니다.
    temperature = max(0.0, min(2.0, temperature))
    top_p = max(0.01, min(1.0, top_p))
    max_output_tokens = max(1, max_output_tokens)

    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
    }


def get_cloud_llm_runtime_info() -> Dict[str, Any]:
    """
    디버깅과 Trace 기록용 Cloud LLM 설정 정보를 반환합니다.

    API Key 값은 절대 반환하지 않고, 설정 여부만 True/False로 반환합니다.
    """
    _load_env_if_available()
    provider = _get_cloud_provider()
    info: Dict[str, Any] = {
        "cloud_llm_provider": provider,
        "model_name": _get_model_name(provider),
        "generation_config": _get_generation_config(),
        "api_key_configured": bool(os.getenv(_get_api_key_env_name(provider), "").strip()),
        "diagnostics": _get_diagnostics_config(include_path_object=False),
    }
    if provider == "nvidia":
        info["nvidia_base_url"] = _get_nvidia_base_url()
    return info


def _get_api_key_env_name(provider: str) -> str:
    """Provider별 API Key 환경변수명을 반환합니다. 실제 Key 값은 반환하지 않습니다."""
    env_name_by_provider = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
    }
    env_name = env_name_by_provider.get(provider)
    if not env_name:
        raise ValueError("지원하지 않는 Cloud LLM Provider입니다.")
    return env_name


def _build_chat_messages(prompt: str, default_system_prompt: str | None = None) -> list[dict[str, str]]:
    """
    Chat Completions API에 전달할 messages 배열을 구성합니다.

    우선순위:
    1. LLM_SYSTEM_PROMPT
    2. CLOUD_LLM_SYSTEM_PROMPT
    3. 함수 인자로 받은 default_system_prompt
    """
    system_prompt = (
        os.getenv("LLM_SYSTEM_PROMPT", "").strip()
        or os.getenv("CLOUD_LLM_SYSTEM_PROMPT", "").strip()
        or (default_system_prompt or "")
    )

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _parse_float_or_default(raw_value: str, default_value: float) -> float:
    """문자열을 float로 변환하고 실패 시 기본값을 반환합니다."""
    if not raw_value:
        return default_value
    try:
        parsed = float(raw_value)
    except ValueError:
        return default_value
    return parsed


def _parse_int_or_default(raw_value: str, default_value: int) -> int:
    """문자열을 int로 변환하고 실패 시 기본값을 반환합니다."""
    if not raw_value:
        return default_value
    try:
        parsed = int(raw_value)
    except ValueError:
        return default_value
    return parsed


def _validate_prompt(prompt: str) -> None:
    """Cloud LLM에 전달할 프롬프트가 올바른 문자열인지 확인합니다."""
    if not isinstance(prompt, str):
        raise ValueError("Cloud LLM에 전달할 prompt는 문자열(str)이어야 합니다.")

    if not prompt.strip():
        raise ValueError("Cloud LLM에 전달할 prompt가 비어 있습니다. 교육용 질문 문장을 입력해 주세요.")


# -----------------------------------------------------------------------------
# Cloud LLM diagnostics
# -----------------------------------------------------------------------------


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_diagnostics_config(include_path_object: bool = True) -> dict[str, Any]:
    enabled = _get_bool_env("CLOUD_LLM_DIAGNOSTICS_ENABLED", True)
    raw_log_path = os.getenv("CLOUD_LLM_DIAGNOSTICS_LOG_PATH", "").strip() or DEFAULT_DIAGNOSTICS_LOG_PATH
    # .env에서 따옴표를 붙인 경우에도 실제 파일 경로로 해석되도록 정리합니다.
    raw_log_path = raw_log_path.strip().strip("'").strip('"')
    log_path = Path(raw_log_path).expanduser()
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path

    config: dict[str, Any] = {
        "enabled": enabled,
        "level": os.getenv("CLOUD_LLM_DIAGNOSTICS_LEVEL", "verbose").strip().lower() or "verbose",
        "log_path": log_path if include_path_object else str(log_path),
        "preview_chars": max(0, _get_int_env("CLOUD_LLM_DIAGNOSTICS_PREVIEW_CHARS", DEFAULT_DIAGNOSTICS_PREVIEW_CHARS)),
        "include_prompt": _get_bool_env("CLOUD_LLM_DIAGNOSTICS_INCLUDE_PROMPT", False),
        "include_raw_response": _get_bool_env("CLOUD_LLM_DIAGNOSTICS_INCLUDE_RAW_RESPONSE", False),
        "max_choices": max(1, _get_int_env("CLOUD_LLM_DIAGNOSTICS_MAX_CHOICES", DEFAULT_DIAGNOSTICS_MAX_CHOICES)),
    }
    return config


def _new_trace_id(provider: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    suffix = uuid.uuid4().hex[:8]
    clean_provider = re.sub(r"[^a-zA-Z0-9_-]+", "-", provider or "cloud")
    return f"cllm-{clean_provider}-{timestamp}-{suffix}"


def _write_diagnostic_event(event: dict[str, Any]) -> None:
    """Cloud LLM diagnostic event를 JSONL에 append합니다.

    저장 실패가 LLM 호출을 중단하지 않게 하되, 디버깅 중에는 실패 원인을 볼 수 있도록
    CLOUD_LLM_DIAGNOSTICS_PRINT_WRITE_ERRORS=true이면 콘솔에 짧게 출력합니다.
    """
    config = _get_diagnostics_config(include_path_object=True)
    if not config["enabled"]:
        return

    log_path = config["log_path"]
    assert isinstance(log_path, Path)

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        safe_event = _to_jsonable(event)
        line = json.dumps(safe_event, ensure_ascii=False, separators=(",", ":"), default=str)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
    except Exception as exc:
        # 진단 로그 저장 실패가 실제 LLM 호출 흐름을 막으면 안 됩니다.
        if _get_bool_env("CLOUD_LLM_DIAGNOSTICS_PRINT_WRITE_ERRORS", True):
            try:
                print(f"[LLM][diagnostics] 로그 저장 실패: {_safe_short_error(exc)}")
                print(f"[LLM][diagnostics] 로그 경로: {log_path}")
            except Exception:
                pass
        return



def write_cloud_llm_diagnostics_self_check_event() -> Path:
    """진단 JSONL writer가 실제로 파일에 append하는지 확인하기 위한 수동 점검 함수입니다."""
    config = _get_diagnostics_config(include_path_object=True)
    log_path = config["log_path"]
    assert isinstance(log_path, Path)
    _write_diagnostic_event(
        _build_base_event(
            trace_id=_new_trace_id("self-check"),
            event_type="cloud_llm_diagnostics_self_check",
            provider="self-check",
            model_name="self-check",
            base_url="",
            diagnostic={
                "enabled": config["enabled"],
                "log_path": str(log_path),
                "message": "Cloud LLM diagnostics writer self-check event",
            },
            trace_context=None,
        )
    )
    return log_path


def _build_base_event(
    *,
    trace_id: str,
    event_type: str,
    provider: str,
    model_name: str | None = None,
    base_url: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    diagnostic: dict[str, Any] | None = None,
    trace_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _get_diagnostics_config(include_path_object=False)
    event: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "trace_id": trace_id,
        "event_type": event_type,
        "provider": provider,
        "model": model_name or "",
        "base_url": base_url or "",
        "diagnostics_level": config["level"],
        "error_code": error_code,
        "error_message": error_message,
        "diagnostic": diagnostic or {},
    }
    if trace_context:
        event["trace_context"] = trace_context
    return event


def _write_request_event(
    *,
    trace_id: str,
    provider: str,
    model_name: str,
    base_url: str | None,
    messages: list[dict[str, Any]],
    generation_config: dict[str, Any],
    stream: bool,
    response_format: Any,
    trace_context: dict[str, Any] | None,
) -> None:
    diagnostic = _build_request_diagnostic(
        messages=messages,
        generation_config=generation_config,
        stream=stream,
        response_format=response_format,
    )
    event = _build_base_event(
        trace_id=trace_id,
        event_type="cloud_llm_request",
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        diagnostic=diagnostic,
        trace_context=trace_context,
    )
    _write_diagnostic_event(event)


def _write_response_event(
    *,
    trace_id: str,
    provider: str,
    model_name: str,
    base_url: str | None,
    response: Any,
    latency_ms: float,
    trace_context: dict[str, Any] | None,
) -> None:
    diagnostic = _build_response_diagnostic(
        response=response,
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        latency_ms=latency_ms,
    )
    event = _build_base_event(
        trace_id=trace_id,
        event_type="cloud_llm_response",
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        diagnostic=diagnostic,
        trace_context=trace_context,
    )
    _write_diagnostic_event(event)


def _write_error_event(
    *,
    trace_id: str,
    provider: str,
    model_name: str | None,
    base_url: str | None,
    error_code: str,
    error_message: str,
    diagnostic: dict[str, Any] | None,
    trace_context: dict[str, Any] | None,
) -> None:
    event = _build_base_event(
        trace_id=trace_id,
        event_type="cloud_llm_error",
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        error_code=error_code,
        error_message=error_message,
        diagnostic=diagnostic or {},
        trace_context=trace_context,
    )
    _write_diagnostic_event(event)


def _write_extraction_event(
    *,
    trace_id: str,
    event_type: str,
    provider_label: str,
    model_name: str | None,
    base_url: str | None,
    diagnostic: dict[str, Any],
    error_code: str | None = None,
    error_message: str | None = None,
    trace_context: dict[str, Any] | None = None,
) -> None:
    provider = (provider_label or "cloud").lower()
    event = _build_base_event(
        trace_id=trace_id,
        event_type=event_type,
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        error_code=error_code,
        error_message=error_message,
        diagnostic=diagnostic,
        trace_context=trace_context,
    )
    _write_diagnostic_event(event)


def _build_request_diagnostic(
    *,
    messages: list[dict[str, Any]],
    generation_config: dict[str, Any],
    stream: bool,
    response_format: Any,
) -> dict[str, Any]:
    config = _get_diagnostics_config(include_path_object=False)
    include_prompt = bool(config["include_prompt"])
    preview_chars = int(config["preview_chars"])

    message_summaries = []
    for index, message in enumerate(messages or []):
        role = message.get("role", "") if isinstance(message, dict) else ""
        content = message.get("content", "") if isinstance(message, dict) else ""
        item: dict[str, Any] = {
            "index": index,
            "role": role,
            "content_type": type(content).__name__,
            "content_length": len(content) if isinstance(content, str) else 0,
        }
        if include_prompt:
            item["content_preview"] = _safe_text_preview(content, preview_chars)
        message_summaries.append(item)

    return {
        "temperature": generation_config.get("temperature"),
        "top_p": generation_config.get("top_p"),
        "max_tokens": generation_config.get("max_output_tokens"),
        "messages_count": len(messages or []),
        "messages": message_summaries,
        "stream": stream,
        "response_format": _to_jsonable(response_format),
    }


def _build_response_diagnostic(
    response: Any,
    provider: str,
    model_name: str | None,
    base_url: str | None,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    config = _get_diagnostics_config(include_path_object=False)
    response_id = _read_optional_value(response, "id")
    response_model = _read_optional_value(response, "model") or model_name
    response_created = _read_optional_value(response, "created")
    choices = _read_optional_value(response, "choices")
    choices_count = len(choices) if _is_non_empty_sequence(choices) else 0
    max_choices = int(config["max_choices"])

    diagnostic: dict[str, Any] = {
        "provider": provider,
        "model": model_name or "",
        "base_url": base_url or "",
        "response_type": type(response).__name__,
        "response_id": response_id,
        "response_model": response_model,
        "response_created": response_created,
        "choices_count": choices_count,
        "usage": _build_usage_diagnostic(response),
        "api_latency_ms": latency_ms,
        "request_id": _extract_request_id(response),
        "choices": [],
    }

    if _is_non_empty_sequence(choices):
        for index, choice in enumerate(choices[:max_choices]):
            diagnostic["choices"].append(_build_choice_diagnostic(choice, index))

    if config["include_raw_response"]:
        diagnostic["raw_response_preview"] = _safe_text_preview(_to_jsonable(response), int(config["preview_chars"]))

    first_choice_summary = diagnostic["choices"][0] if diagnostic["choices"] else {}
    diagnostic.update(_flatten_first_choice_summary(first_choice_summary))
    return diagnostic


def _flatten_first_choice_summary(choice_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "finish_reason": choice_summary.get("finish_reason"),
        "message_content_type": choice_summary.get("message_content_type"),
        "message_content_length": choice_summary.get("message_content_length"),
        "tool_calls_count": choice_summary.get("tool_calls_count"),
    }


def _build_choice_diagnostic(choice: Any, index: int) -> dict[str, Any]:
    config = _get_diagnostics_config(include_path_object=False)
    include_raw_response = bool(config["include_raw_response"])
    preview_chars = int(config["preview_chars"])

    message = _read_optional_value(choice, "message")
    content = _read_optional_value(message, "content") if message is not None else None
    refusal = _read_optional_value(message, "refusal") if message is not None else None
    tool_calls = _read_optional_value(message, "tool_calls") if message is not None else None
    function_call = _read_optional_value(message, "function_call") if message is not None else None
    choice_text = _read_optional_value(choice, "text")
    delta = _read_optional_value(choice, "delta")
    delta_content = _read_optional_value(delta, "content") if delta is not None else None

    item: dict[str, Any] = {
        "index": index,
        "finish_reason": _read_optional_value(choice, "finish_reason"),
        "message_type": type(message).__name__ if message is not None else "None",
        "message_role": _read_optional_value(message, "role") if message is not None else None,
        "message_content_type": type(content).__name__ if content is not None else "None",
        "message_content_length": _content_length(content),
        "has_refusal": bool(refusal),
        "refusal_length": _content_length(refusal),
        "tool_calls_count": _sequence_length(tool_calls),
        "has_function_call": function_call is not None,
        "choice_text_type": type(choice_text).__name__ if choice_text is not None else "None",
        "choice_text_length": _content_length(choice_text),
        "delta_content_type": type(delta_content).__name__ if delta_content is not None else "None",
        "delta_content_length": _content_length(delta_content),
    }

    if include_raw_response:
        item["message_content_preview"] = _safe_text_preview(content, preview_chars)
        item["choice_text_preview"] = _safe_text_preview(choice_text, preview_chars)
        item["delta_content_preview"] = _safe_text_preview(delta_content, preview_chars)
        item["tool_calls_preview"] = _safe_text_preview(_to_jsonable(tool_calls), preview_chars)

    return item


def _build_usage_diagnostic(response: Any) -> dict[str, Any]:
    usage = _read_optional_value(response, "usage")
    if usage is None:
        return {}

    keys = [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "audio_tokens",
    ]
    result: dict[str, Any] = {}
    for key in keys:
        value = _read_optional_value(usage, key)
        if value is not None:
            result[key] = value

    # OpenAI SDK의 세부 token details도 가능한 범위에서 보존합니다.
    for nested_key in (
        "prompt_tokens_details",
        "completion_tokens_details",
        "input_tokens_details",
        "output_tokens_details",
    ):
        value = _read_optional_value(usage, nested_key)
        if value is not None:
            result[nested_key] = _to_jsonable(value)

    return result


def _extract_request_id(response: Any) -> str:
    for key in ("request_id", "_request_id", "x_request_id", "id"):
        value = _read_optional_value(response, key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _content_length(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, tuple, dict)):
        try:
            return len(value)
        except Exception:
            return 0
    return 0


def _sequence_length(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return len(value)
    if value is None:
        return 0
    return 1


def _get_exception_diagnostic(error: Exception) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {
        "exception_type": type(error).__name__,
        "message": _safe_short_error(error),
    }
    if isinstance(error, CloudLLMError):
        diagnostic["cloud_llm_error_code"] = error.error_code
        diagnostic["cloud_llm_diagnostic"] = error.diagnostic
    return diagnostic


def _get_cloud_error_code(error: Exception) -> str:
    if isinstance(error, CloudLLMError):
        return error.error_code

    message = str(error)
    lowered = message.lower()
    if "텍스트를 추출하지 못" in message or "text extraction" in lowered:
        return "LLM_TEXT_EXTRACTION_FAILED"
    if "empty" in lowered or "비어" in message:
        return "LLM_RESPONSE_EMPTY"
    if "api" in lowered or "http" in lowered or "rate limit" in lowered or "timeout" in lowered:
        return "LLM_API_ERROR"
    return "LLM_CALL_FAILED"


def _safe_text_preview(value: Any, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    try:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(_to_jsonable(value), ensure_ascii=False)
    except Exception:
        text = str(value)
    text = _mask_sensitive_text(text)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def _to_jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "[MAX_DEPTH_EXCEEDED]"

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                result[key_text] = "[MASKED_SECRET]"
            else:
                result[key_text] = _to_jsonable(item, depth + 1)
        return result

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item, depth + 1) for item in list(value)]

    # pydantic/openai SDK 객체 대응
    for method_name in ("model_dump", "dict", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _to_jsonable(method(), depth + 1)
            except Exception:
                pass

    # 마지막으로 공개 속성 일부만 가져옵니다.
    try:
        data: dict[str, Any] = {}
        for attr_name in ("id", "model", "created", "choices", "usage", "object"):
            attr_value = _read_optional_value(value, attr_name)
            if attr_value is not None:
                data[attr_name] = _to_jsonable(attr_value, depth + 1)
        if data:
            return data
    except Exception:
        pass

    return _mask_sensitive_text(str(value))


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if normalized in SAFE_TOKEN_COUNT_KEYS:
        return False
    if normalized in SENSITIVE_EXACT_KEYS:
        return True
    if normalized.endswith("_token") and normalized not in SAFE_TOKEN_COUNT_KEYS:
        return True
    if normalized.endswith("_key") and "api" in normalized:
        return True
    if "password" in normalized or "secret" in normalized:
        return True
    if normalized.endswith("token_ids"):
        return True
    return False


def _mask_sensitive_text(text: str) -> str:
    if not text:
        return ""

    masked = text
    masked = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;\}\]]+",
        r"\1[MASKED_SECRET]",
        masked,
    )
    masked = re.sub(
        r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|session[_-]?token|password|secret)\s*[:=]\s*[^\s,;\}\]]+",
        r"\1=[MASKED_SECRET]",
        masked,
    )
    masked = re.sub(r"\bsk-[A-Za-z0-9_\-]{12,}\b", "[MASKED_SECRET]", masked)
    masked = re.sub(r"\bnvapi-[A-Za-z0-9_\-\.]{12,}\b", "[MASKED_SECRET]", masked)
    return masked


def _clean_gemini_response_text(response: Any) -> str:
    """google-genai 응답 객체에서 텍스트를 안전하게 추출합니다."""
    text = ""

    try:
        text = getattr(response, "text", "") or ""
    except Exception:
        text = ""

    if isinstance(text, str) and text.strip():
        return text.strip()

    try:
        candidates = getattr(response, "candidates", None) or []
        extracted_parts = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            for part in parts:
                part_text = getattr(part, "text", "")
                if isinstance(part_text, str) and part_text.strip():
                    extracted_parts.append(part_text.strip())
        if extracted_parts:
            return "\n".join(extracted_parts).strip()
    except Exception:
        pass

    raise ValueError(
        "Gemini 응답에서 텍스트를 추출하지 못했습니다. "
        "응답이 비어 있거나 안전 설정, 모델 설정, 입력 프롬프트 문제로 응답이 생성되지 않았을 수 있습니다."
    )


def _clean_chat_completion_text(
    response: Any,
    provider_label: str,
    model_name: str | None = None,
    base_url: str | None = None,
    trace_id: str | None = None,
    trace_context: dict[str, Any] | None = None,
) -> str:
    """
    OpenAI-compatible Chat Completions 응답 객체에서 텍스트를 안전하게 추출합니다.

    provider별 SDK나 endpoint에 따라 응답 구조가 조금씩 다를 수 있으므로
    다음 순서로 텍스트를 확인합니다.
    1. response.output_text, response.text, response.response, response.content
    2. response.choices[0].message.content
    3. content가 list인 경우 각 block의 text/content/value/output_text
    4. response.choices[0].message.refusal
    5. response.choices[0].text
    6. response.choices[0].delta.content

    텍스트를 찾지 못하면 API Key나 원문 응답을 출력하지 않고,
    choices 수, finish_reason, content 타입 같은 안전한 진단 정보만 포함해 예외를 발생시킵니다.
    """
    checked_paths: list[str] = []
    active_trace_id = trace_id or _new_trace_id(provider_label.lower())

    direct_text, direct_path = _extract_direct_response_text(response, checked_paths)
    if direct_text:
        _write_text_extraction_success_event(
            trace_id=active_trace_id,
            provider_label=provider_label,
            model_name=model_name,
            base_url=base_url,
            extraction_path=direct_path,
            checked_paths=checked_paths,
            extracted_text=direct_text,
            response=response,
            trace_context=trace_context,
        )
        return direct_text

    choices = _read_optional_value(response, "choices")
    checked_paths.append("response.choices")
    if not _is_non_empty_sequence(choices):
        diagnostic = _build_text_extraction_failure_diagnostic(response, provider_label, model_name, base_url, checked_paths)
        _write_text_extraction_failed_event(
            trace_id=active_trace_id,
            provider_label=provider_label,
            model_name=model_name,
            base_url=base_url,
            diagnostic=diagnostic,
            trace_context=trace_context,
        )
        raise CloudLLMError(
            error_code="LLM_TEXT_EXTRACTION_FAILED",
            message=f"{provider_label} 응답에서 choices를 찾지 못했거나 choices가 비어 있습니다.",
            diagnostic=diagnostic,
        )

    first_choice = choices[0]
    message = _read_optional_value(first_choice, "message")
    checked_paths.append("choices[0].message")

    if message is not None:
        content = _read_optional_value(message, "content")
        checked_paths.append("choices[0].message.content")
        text, content_path = _extract_text_from_content(content, "choices[0].message.content", checked_paths)
        if text:
            _write_text_extraction_success_event(
                trace_id=active_trace_id,
                provider_label=provider_label,
                model_name=model_name,
                base_url=base_url,
                extraction_path=content_path,
                checked_paths=checked_paths,
                extracted_text=text,
                response=response,
                trace_context=trace_context,
            )
            return text

        # 일부 provider는 안전 거절 또는 대체 응답을 별도 필드에 넣을 수 있습니다.
        checked_paths.append("choices[0].message.refusal")
        refusal_text, refusal_path = _extract_text_from_content(
            _read_optional_value(message, "refusal"),
            "choices[0].message.refusal",
            checked_paths,
        )
        if refusal_text:
            _write_text_extraction_success_event(
                trace_id=active_trace_id,
                provider_label=provider_label,
                model_name=model_name,
                base_url=base_url,
                extraction_path=refusal_path,
                checked_paths=checked_paths,
                extracted_text=refusal_text,
                response=response,
                trace_context=trace_context,
            )
            return refusal_text

    checked_paths.append("choices[0].text")
    choice_text, choice_text_path = _extract_text_from_content(
        _read_optional_value(first_choice, "text"),
        "choices[0].text",
        checked_paths,
    )
    if choice_text:
        _write_text_extraction_success_event(
            trace_id=active_trace_id,
            provider_label=provider_label,
            model_name=model_name,
            base_url=base_url,
            extraction_path=choice_text_path,
            checked_paths=checked_paths,
            extracted_text=choice_text,
            response=response,
            trace_context=trace_context,
        )
        return choice_text

    delta = _read_optional_value(first_choice, "delta")
    checked_paths.append("choices[0].delta")
    if delta is not None:
        checked_paths.append("choices[0].delta.content")
        delta_text, delta_path = _extract_text_from_content(
            _read_optional_value(delta, "content"),
            "choices[0].delta.content",
            checked_paths,
        )
        if delta_text:
            _write_text_extraction_success_event(
                trace_id=active_trace_id,
                provider_label=provider_label,
                model_name=model_name,
                base_url=base_url,
                extraction_path=delta_path,
                checked_paths=checked_paths,
                extracted_text=delta_text,
                response=response,
                trace_context=trace_context,
            )
            return delta_text

    diagnostic = _build_text_extraction_failure_diagnostic(response, provider_label, model_name, base_url, checked_paths)
    _write_text_extraction_failed_event(
        trace_id=active_trace_id,
        provider_label=provider_label,
        model_name=model_name,
        base_url=base_url,
        diagnostic=diagnostic,
        trace_context=trace_context,
    )
    raise CloudLLMError(
        error_code="LLM_TEXT_EXTRACTION_FAILED",
        message=f"{provider_label} 응답에서 텍스트를 추출하지 못했습니다.",
        diagnostic=diagnostic,
    )


def _write_text_extraction_success_event(
    *,
    trace_id: str,
    provider_label: str,
    model_name: str | None,
    base_url: str | None,
    extraction_path: str,
    checked_paths: list[str],
    extracted_text: str,
    response: Any,
    trace_context: dict[str, Any] | None,
) -> None:
    config = _get_diagnostics_config(include_path_object=False)
    diagnostic = {
        "extraction_path": extraction_path,
        "checked_paths": checked_paths,
        "extracted_text_length": len(extracted_text),
        "response_type": type(response).__name__,
        "choices_count": _sequence_length(_read_optional_value(response, "choices")),
        "finish_reason": _first_choice_value(response, "finish_reason"),
        "usage": _build_usage_diagnostic(response),
    }
    if config["include_raw_response"]:
        diagnostic["extracted_text_preview"] = _safe_text_preview(extracted_text, int(config["preview_chars"]))
    _write_extraction_event(
        trace_id=trace_id,
        event_type="cloud_llm_text_extraction_succeeded",
        provider_label=provider_label,
        model_name=model_name,
        base_url=base_url,
        diagnostic=diagnostic,
        trace_context=trace_context,
    )


def _write_text_extraction_failed_event(
    *,
    trace_id: str,
    provider_label: str,
    model_name: str | None,
    base_url: str | None,
    diagnostic: dict[str, Any],
    trace_context: dict[str, Any] | None,
) -> None:
    _write_extraction_event(
        trace_id=trace_id,
        event_type="cloud_llm_text_extraction_failed",
        provider_label=provider_label,
        model_name=model_name,
        base_url=base_url,
        error_code="LLM_TEXT_EXTRACTION_FAILED",
        error_message=f"{provider_label} 응답에서 텍스트를 추출하지 못했습니다.",
        diagnostic=diagnostic,
        trace_context=trace_context,
    )


def _build_text_extraction_failure_diagnostic(
    response: Any,
    provider_label: str,
    model_name: str | None,
    base_url: str | None,
    checked_paths: list[str],
) -> dict[str, Any]:
    diagnostic = _build_response_diagnostic(
        response=response,
        provider=provider_label.lower(),
        model_name=model_name,
        base_url=base_url,
        latency_ms=None,
    )
    diagnostic["checked_paths"] = checked_paths
    diagnostic["error_code"] = "LLM_TEXT_EXTRACTION_FAILED"
    return diagnostic


def _first_choice_value(response: Any, key: str) -> Any:
    choices = _read_optional_value(response, "choices")
    if not _is_non_empty_sequence(choices):
        return None
    return _read_optional_value(choices[0], key)


def _read_optional_value(source: Any, key: str) -> Any:
    """
    dict와 SDK 객체 양쪽에서 값을 안전하게 읽습니다.

    OpenAI-compatible 응답은 SDK 버전과 provider에 따라
    dict처럼 오거나 attribute 객체처럼 올 수 있습니다.
    """
    if source is None:
        return None

    if isinstance(source, dict):
        return source.get(key)

    try:
        return getattr(source, key)
    except Exception:
        return None


def _is_non_empty_sequence(value: Any) -> bool:
    """list/tuple 형태이고 원소가 1개 이상인지 확인합니다."""
    return isinstance(value, (list, tuple)) and len(value) > 0


def _extract_direct_response_text(response: Any, checked_paths: list[str]) -> tuple[str, str]:
    """
    ChatCompletion 형식이 아닌 응답 객체에서 직접 텍스트 필드를 찾습니다.

    일부 OpenAI-compatible 또는 Responses API 계열 응답은
    choices 대신 output_text/text/content 같은 필드를 사용할 수 있습니다.
    """
    for field_name in ("output_text", "text", "response", "content"):
        path = f"response.{field_name}"
        checked_paths.append(path)
        text, found_path = _extract_text_from_content(_read_optional_value(response, field_name), path, checked_paths)
        if text:
            return text, found_path
    return "", ""


def _extract_text_from_content(content: Any, path: str, checked_paths: list[str]) -> tuple[str, str]:
    """
    문자열, list block, dict block, SDK content 객체에서 텍스트를 추출합니다.
    """
    if content is None:
        return "", ""

    if isinstance(content, str):
        text = content.strip()
        return (text, path) if text else ("", "")

    if isinstance(content, (int, float, bool)):
        return "", ""

    if isinstance(content, (list, tuple)):
        parts = []
        found_paths = []
        for index, item in enumerate(content):
            item_path = f"{path}[{index}]"
            checked_paths.append(item_path)
            item_text, found_path = _extract_text_from_content(item, item_path, checked_paths)
            if item_text:
                parts.append(item_text)
                found_paths.append(found_path or item_path)
        joined = "\n".join(parts).strip()
        return (joined, found_paths[0] if found_paths else path) if joined else ("", "")

    if isinstance(content, dict):
        for key in ("text", "content", "output_text", "value"):
            item_path = f"{path}.{key}"
            checked_paths.append(item_path)
            text, found_path = _extract_text_from_content(content.get(key), item_path, checked_paths)
            if text:
                return text, found_path
        return "", ""

    # SDK content block 객체 대응: item.text 또는 item.content 형태
    for attr_name in ("text", "content", "output_text", "value"):
        item_path = f"{path}.{attr_name}"
        checked_paths.append(item_path)
        text, found_path = _extract_text_from_content(_read_optional_value(content, attr_name), item_path, checked_paths)
        if text:
            return text, found_path

    return "", ""


def _build_chat_completion_diagnostic(
    response: Any,
    provider_label: str = "",
    model_name: str | None = None,
    base_url: str | None = None,
) -> str:
    """
    텍스트 추출 실패 원인을 추적하기 위한 안전한 진단 문자열을 만듭니다.

    응답 원문이나 API Key는 포함하지 않습니다.
    """
    diagnostic = _build_response_diagnostic(
        response=response,
        provider=provider_label,
        model_name=model_name,
        base_url=base_url,
        latency_ms=None,
    )
    parts = [
        f"provider={diagnostic.get('provider')}" if diagnostic.get("provider") else "",
        f"model={diagnostic.get('model')}" if diagnostic.get("model") else "",
        f"base_url={diagnostic.get('base_url')}" if diagnostic.get("base_url") else "",
        f"response_type={diagnostic.get('response_type')}",
        f"choices_count={diagnostic.get('choices_count')}",
        f"finish_reason={diagnostic.get('finish_reason')}" if diagnostic.get("finish_reason") else "",
        f"content_type={diagnostic.get('message_content_type')}" if diagnostic.get("message_content_type") else "",
        f"content_length={diagnostic.get('message_content_length')}" if diagnostic.get("message_content_length") is not None else "",
        f"tool_calls_count={diagnostic.get('tool_calls_count')}" if diagnostic.get("tool_calls_count") is not None else "",
    ]
    return _safe_short_error(Exception(", ".join(part for part in parts if part)))


def _clean_anthropic_response_text(response: Any) -> str:
    """Anthropic 응답 객체에서 텍스트를 안전하게 추출합니다."""
    content_blocks = getattr(response, "content", None) or []
    extracted_parts = []

    for block in content_blocks:
        block_text = getattr(block, "text", "")
        if isinstance(block_text, str) and block_text.strip():
            extracted_parts.append(block_text.strip())

    return "\n".join(extracted_parts).strip()


def _load_env_if_available() -> None:
    """
    python-dotenv가 설치되어 있으면 .env 파일을 자동으로 읽습니다.

    utf-8-sig 인코딩을 명시해 Windows/메모장 환경에서 BOM이 붙은 .env도 읽을 수 있게 합니다.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    # 현재 작업 폴더 기준 .env를 우선 읽습니다.
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        # 실습 프로젝트에서는 .env 설정이 현재 실행 환경보다 우선하도록 override=True를 사용합니다.
        load_dotenv(dotenv_path=env_path, encoding="utf-8-sig", override=True)
    else:
        load_dotenv(encoding="utf-8-sig", override=True)


def _safe_short_error(error: Exception) -> str:
    """민감정보를 마스킹한 짧은 오류 메시지를 반환합니다. 일반 base_url은 보존합니다."""
    message = str(error) if error else "알 수 없는 오류"
    message = _mask_sensitive_text(message)
    if len(message) > ERROR_PREVIEW_LENGTH:
        message = message[:ERROR_PREVIEW_LENGTH].rstrip() + "..."
    return message


if __name__ == "__main__":
    _load_env_if_available()

    test_prompt = (
        "교육용 가상 공정 라인의 EDU-EQP-01 설비에서 "
        "TEMP-WARN-001 알람이 반복 발생했다고 가정합니다. "
        "초보 수강생도 이해할 수 있도록 원인 후보와 1차 확인 항목을 Markdown으로 정리해 주세요."
    )

    current_provider = os.getenv("CLOUD_LLM_PROVIDER", DEFAULT_CLOUD_PROVIDER).strip().lower()

    print("=" * 80)
    print("Cloud LLM 단독 실행 테스트")
    print("=" * 80)
    print("주의: 이 테스트는 실제 Cloud LLM API를 호출합니다.")
    print(f"현재 CLOUD_LLM_PROVIDER: {current_provider}")
    try:
        runtime_info = get_cloud_llm_runtime_info()
        print(f"현재 모델명: {runtime_info.get('model_name')}")
        if runtime_info.get("nvidia_base_url"):
            print(f"NVIDIA_BASE_URL: {runtime_info.get('nvidia_base_url')}")
        print(f"API Key 설정 여부: {runtime_info.get('api_key_configured')}")
        print(f"Diagnostics: {runtime_info.get('diagnostics')}")
    except Exception as exc:
        print("Runtime 설정 확인 중 오류가 발생했습니다.")
        print(_safe_short_error(exc))
    print("주의: API Key 값은 화면에 표시하지 않습니다.")
    print("-" * 80)

    try:
        result = generate_cloud_llm_response(test_prompt)
        preview_length = 1000
        print(result[:preview_length])
        if len(result) > preview_length:
            print("\n... [응답 미리보기 생략]")
    except Exception as exc:
        print("Cloud LLM 호출을 완료하지 못했습니다.")
        print(_safe_short_error(exc))
