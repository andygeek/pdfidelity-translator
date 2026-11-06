import logging

import httpx
import openai
from babeldoc.utils.atomic_integer import AtomicInteger
from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.translator.base_rate_limiter import BaseRateLimiter
from pdf2zh_next.translator.base_translator import BaseTranslator
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

logger = logging.getLogger(__name__)


class IncompleteCompletionError(Exception):
    """The API answered, but the answer is empty or was cut by the token limit.

    Reasoning models (e.g. gpt-oss on Groq) spend completion tokens on reasoning
    first. When the budget runs out the visible ``content`` is empty or truncated.
    Accepting such answers produced blank or partial paragraphs downstream, so we
    raise and retry instead.
    """


# Transient failures worth a short, bounded retry. RateLimitError keeps its own
# long retry policy below.
_TRANSIENT_ERRORS = (
    IncompleteCompletionError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAITranslator(BaseTranslator):
    # https://github.com/openai/openai-python
    name = "openai"

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        super().__init__(settings, rate_limiter)
        self.timeout = settings.translate_engine_settings.openai_timeout
        self.client = openai.OpenAI(
            base_url=settings.translate_engine_settings.openai_base_url,
            api_key=settings.translate_engine_settings.openai_api_key,
            timeout=float(self.timeout) if self.timeout else openai.NOT_GIVEN,
            http_client=httpx.Client(
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=None
                )
            ),
        )
        self.options = {}
        self.temperature = settings.translate_engine_settings.openai_temperature
        self.reasoning_effort = (
            settings.translate_engine_settings.openai_reasoning_effort
        )
        self.send_temperature = (
            settings.translate_engine_settings.openai_send_temprature
        )
        self.send_reasoning_effort = (
            settings.translate_engine_settings.openai_send_reasoning_effort
        )
        self.extra_body = settings.translate_engine_settings._openai_extra_body
        self.max_completion_tokens = getattr(
            settings.translate_engine_settings, "openai_max_completion_tokens", None
        )

        if self.send_temperature and self.temperature:
            self.add_cache_impact_parameters("temperature", self.temperature)
            self.options["temperature"] = float(self.temperature)
        if self.send_reasoning_effort and self.reasoning_effort:
            self.add_cache_impact_parameters("reasoning_effort", self.reasoning_effort)
            self.options["reasoning_effort"] = self.reasoning_effort
        if self.max_completion_tokens:
            # Does not change the translation itself, only avoids truncation, so it
            # is intentionally not part of the cache key.
            self.options["max_completion_tokens"] = int(self.max_completion_tokens)
        if self.extra_body:
            self.add_cache_impact_parameters("extra_body", self.extra_body)
            self.options["extra_body"] = self.extra_body

        self.model = settings.translate_engine_settings.openai_model
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()
        self.cache_hit_prompt_token_count = AtomicInteger()
        self.incomplete_completion_count = AtomicInteger()

        self.enable_json_mode = (
            settings.translate_engine_settings.openai_enable_json_mode
        )
        if self.enable_json_mode:
            self.add_cache_impact_parameters("enable_json_mode", self.enable_json_mode)

    def _record_usage(self, response) -> None:
        try:
            if hasattr(response, "usage") and response.usage:
                if hasattr(response.usage, "total_tokens"):
                    self.token_count.inc(response.usage.total_tokens)
                if hasattr(response.usage, "prompt_tokens"):
                    self.prompt_token_count.inc(response.usage.prompt_tokens)
                if hasattr(response.usage, "completion_tokens"):
                    self.completion_token_count.inc(response.usage.completion_tokens)
                if hasattr(response.usage, "prompt_cache_hit_tokens"):
                    self.cache_hit_prompt_token_count.inc(
                        response.usage.prompt_cache_hit_tokens
                    )
                elif hasattr(response.usage, "prompt_tokens_details") and hasattr(
                    response.usage.prompt_tokens_details, "cached_tokens"
                ):
                    self.cache_hit_prompt_token_count.inc(
                        response.usage.prompt_tokens_details.cached_tokens
                    )
        except Exception as e:
            logger.error(f"Error getting token usage: {e}")

    def _extract_content(self, response) -> str:
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        content = (choice.message.content or "").strip()
        content = self._remove_cot_content(content).strip()
        if finish_reason == "length" or not content:
            self.incomplete_completion_count.inc(1)
            usage = getattr(response, "usage", None)
            raise IncompleteCompletionError(
                f"Incomplete completion from {self.model}: "
                f"finish_reason={finish_reason!r}, content_chars={len(content)}, "
                f"usage={usage}"
            )
        return content

    def _create(self, messages, rate_limit_params: dict | None):
        options = self.options.copy()
        if (
            self.enable_json_mode
            and rate_limit_params
            and rate_limit_params.get("request_json_mode", False)
        ):
            options["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(
            model=self.model,
            **options,
            messages=messages,
        )
        self._record_usage(response)
        return self._extract_content(response)

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    @retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        return self._create(self.prompt(text), rate_limit_params)

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    @retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None
        return self._create(
            [
                {
                    "role": "user",
                    "content": text,
                },
            ],
            rate_limit_params,
        )
