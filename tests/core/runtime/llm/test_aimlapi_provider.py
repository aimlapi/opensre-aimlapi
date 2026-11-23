"""AI/ML API provider: fixed endpoint, its own key, and no key leaks off-host."""

from __future__ import annotations

import contextlib
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import DefaultHttpxClient

from config.llm_models import AIMLAPI_BASE_URL
from config.llm_settings import LLMSettings
from core.llm.factory import build_llm_client
from core.llm.transports.sdk.llm_clients import OpenAILLMClient
from core.llm.types import ModelType
from surfaces.cli.wizard import llm_credential
from surfaces.shared.llm_setup.auth_profiles import api_key_setup_url
from surfaces.shared.llm_setup.catalog import PROVIDER_BY_VALUE

_TEST_KEY = "aimlapi-test-key"
_CHAT_OK = {
    "id": "test",
    "created": 0,
    "model": "openai/gpt-5.4-mini",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
}


def _aimlapi_client(monkeypatch: pytest.MonkeyPatch) -> OpenAILLMClient:
    settings = SimpleNamespace(provider="aimlapi", aimlapi_toolcall_model="openai/gpt-5.4-mini")
    monkeypatch.setattr("config.llm_settings.resolve_llm_settings", lambda: settings)
    monkeypatch.setattr(
        "core.llm.providers.provider_credentials.resolve_llm_api_key",
        lambda env_var: _TEST_KEY if env_var == "AIMLAPI_API_KEY" else "",
    )
    client = build_llm_client(ModelType.TOOLCALL)
    assert isinstance(client, OpenAILLMClient)
    return client


def _invoke_with_transport(
    client: OpenAILLMClient, handler: httpx.MockTransport
) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def _record(request: httpx.Request) -> None:
        seen.append(request)

    client._http_client = DefaultHttpxClient(transport=handler, event_hooks={"request": [_record]})
    client._client = None
    # Only the outgoing headers matter here, not how the call itself ends.
    with contextlib.suppress(RuntimeError):
        client.invoke("hi")
    return seen


def test_settings_defaults_use_namespaced_tool_capable_model() -> None:
    settings = LLMSettings.model_validate({"provider": "aimlapi"})
    choices = {row.value for row in PROVIDER_BY_VALUE["aimlapi"].models}

    for role in ("reasoning", "classification", "toolcall"):
        model = getattr(settings, f"aimlapi_{role}_model")
        assert model == "openai/gpt-5.4-mini"
        assert model in choices
    assert all("/" in value for value in choices)


def test_onboarding_entry_points_to_aimlapi_keys_page() -> None:
    provider = PROVIDER_BY_VALUE["aimlapi"]

    assert provider.label == "AI/ML API"
    assert provider.api_key_env == "AIMLAPI_API_KEY"
    assert api_key_setup_url("aimlapi") == "https://aimlapi.com/app/keys"


def test_key_is_sent_only_to_api_aimlapi_com(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _aimlapi_client(monkeypatch)
    assert client._base_url == AIMLAPI_BASE_URL == "https://api.aimlapi.com/v1"
    assert client._api_key_env == "AIMLAPI_API_KEY"

    requests = _invoke_with_transport(
        client, httpx.MockTransport(lambda _r: httpx.Response(HTTPStatus.OK, json=_CHAT_OK))
    )

    assert len(requests) == 1
    assert requests[0].url.scheme == "https"
    assert requests[0].url.host == "api.aimlapi.com"
    assert requests[0].url.path == "/v1/chat/completions"
    assert requests[0].headers["Authorization"] == f"Bearer {_TEST_KEY}"


def test_key_is_not_forwarded_on_redirect_to_another_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _aimlapi_client(monkeypatch)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.aimlapi.com":
            return httpx.Response(
                HTTPStatus.TEMPORARY_REDIRECT,
                headers={"Location": "https://collector.example.net/v1/chat/completions"},
            )
        return httpx.Response(HTTPStatus.OK, json=_CHAT_OK)

    requests = _invoke_with_transport(client, httpx.MockTransport(handle))

    assert requests[0].url.host == "api.aimlapi.com"
    assert requests[0].headers["Authorization"] == f"Bearer {_TEST_KEY}"
    off_host = [r for r in requests if r.url.host != "api.aimlapi.com"]
    assert off_host, "redirect target was never requested"
    for request in off_host:
        assert "Authorization" not in request.headers
        assert _TEST_KEY not in str(request.headers)


def test_wizard_key_prompt_shows_aimlapi_keys_link(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    blank_answer = MagicMock()
    blank_answer.ask.return_value = ""
    monkeypatch.setattr(
        "surfaces.cli.wizard.components.questionary.password", lambda *_a, **_k: blank_answer
    )

    outcome, _model = llm_credential._prompt_validated_llm_credential(
        PROVIDER_BY_VALUE["aimlapi"], model="openai/gpt-5.4-mini"
    )

    assert outcome == "deferred"
    assert "https://aimlapi.com/app/keys" in capsys.readouterr().out
