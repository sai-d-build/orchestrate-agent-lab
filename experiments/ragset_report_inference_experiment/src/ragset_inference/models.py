import json
import os
import re
import time
from openai import OpenAI
from pydantic import ValidationError
from .schemas import (
    ReportPrediction,
    ValidationResult,
    CritiqueResult,
    CritiqueIssue,
    CritiqueIssueType,
    CritiqueStatus,
    JudgeResult,
    JudgeAction,
    LABEL_KEYS,
)


def _is_api_overload_error(error: Exception) -> bool:
    """
    Detect API overload / server errors that warrant a long retry delay.

    Checks for common HTTP status codes and error patterns indicating
    the API is temporarily unavailable or overloaded:
    - 502 Bad Gateway
    - 503 Service Unavailable
    - 504 Gateway Timeout
    - 500 Internal Server Error (sometimes used for overload)
    - Custom codes like 5002, 5003 (provider-specific)
    - Rate limit errors (429)
    - Connection/timeout errors
    """
    error_str = str(error).lower()
    
    # Check for HTTP status codes in error message
    overload_codes = ["502", "503", "504", "5002", "5003", "500"]
    for code in overload_codes:
        if code in error_str:
            # Avoid false positive on "500" - only match if it looks like a status code
            if code == "500":
                # Match patterns like "status 500", "code 500", "error 500", "500 ", " 500"
                import re
                if re.search(r'(status|code|error)\s*[:=]?\s*500\b|\b500\s', error_str):
                    return True
            else:
                return True
    
    # Check for rate limiting
    if "429" in error_str or "rate limit" in error_str or "rate_limit" in error_str:
        return True
    
    # Check for connection/timeout errors
    if any(term in error_str for term in [
        "connection", "timeout", "timed out", "connect",
        "unavailable", "overload", "server error", "bad gateway",
        "gateway timeout", "service unavailable"
    ]):
        return True
    
    return False


def _get_retry_delay(error: Exception, attempt: int, base_delay: float = 2.0) -> float:
    """
    Calculate retry delay based on error type.
    
    For API overload errors (502, 503, 5002, 5003, etc.), use 100 seconds.
    For other transient errors, use exponential backoff.
    """
    if _is_api_overload_error(error):
        return 100.0
    return base_delay * (2 ** attempt)


def _flatten_nested_pf_oa(data: dict) -> dict:
    """
    Flatten nested PF_OA structure if the model incorrectly returns it as a parent
    category containing sub-fields (Effusion, Synovitis, Bakers, Contusion, Fracture).

    The schema expects all 12 labels to be flat at the top level of predictions.
    Some models occasionally nest PF_OA with sub-fields, which causes validation errors.
    """
    if "predictions" not in data:
        return data

    predictions = data["predictions"]
    if "PF_OA" not in predictions:
        return data

    pf_oa = predictions["PF_OA"]
    # Check if PF_OA is a dict with nested sub-fields (not just value/evidence)
    if not isinstance(pf_oa, dict):
        return data

    # Known sub-fields that the model incorrectly nests under PF_OA
    nested_subfields = {"Effusion", "Synovitis", "Bakers", "Contusion", "Fracture"}
    pf_oa_keys = set(pf_oa.keys())

    # If PF_OA contains any of the known sub-fields as keys, it's nested
    if not (pf_oa_keys & nested_subfields):
        return data

    # Extract the main PF_OA value/evidence
    main_pf_oa = {}
    if "value" in pf_oa:
        main_pf_oa["value"] = pf_oa["value"]
    if "evidence" in pf_oa:
        main_pf_oa["evidence"] = pf_oa["evidence"]

    # Replace PF_OA with the flattened version
    predictions["PF_OA"] = main_pf_oa if main_pf_oa else {"value": 0, "evidence": None}

    # Promote nested sub-fields to top level if they don't already exist
    for subfield in nested_subfields:
        if subfield in pf_oa and subfield not in predictions:
            predictions[subfield] = pf_oa[subfield]

    return data


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from text, handling markdown code fences."""
    text = text.strip()
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON from markdown code fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _create_client(provider: str, api_key: str | None = None) -> OpenAI:
    """Create an OpenAI-compatible client for the given provider."""
    if provider == "nvidia":
        return OpenAI(
            api_key=api_key or os.environ.get("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1",
        )
    elif provider == "openrouter":
        return OpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: nvidia, openrouter")


def _detect_provider(model: str) -> str:
    """Detect provider from model name."""
    if model.startswith("nvidia/") or "nemotron" in model.lower():
        return "nvidia"
    return "openrouter"


def _build_extra_body(provider: str, reasoning: bool | None) -> dict | None:
    """Build extra_body for provider-specific parameters."""
    if reasoning is None:
        return None
    if provider == "nvidia":
        # NVIDIA Nemotron supports reasoning parameter
        return {"reasoning": {"enabled": reasoning}}
    elif provider == "openrouter":
        # OpenRouter passes reasoning through to underlying provider
        return {"reasoning": {"enabled": reasoning}}
    return None


class InferenceModel:
    """Inference model using chat.completions API with JSON mode.

    Supports both OpenRouter and NVIDIA Nemotron endpoints.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 4000,
        provider: str | None = None,
        api_key: str | None = None,
        reasoning: bool | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.provider = provider or _detect_provider(model)
        self.reasoning = reasoning
        self.client = _create_client(self.provider, api_key)
        self.last_token_usage = {"input_tokens": 0, "output_tokens": 0}

    def run(
        self,
        system: str,
        user: str,
        validator_feedback: dict | None = None,
        attempt: int | None = None,
    ) -> ReportPrediction:
        """
        Run inference with optional validator feedback for retry.

        Args:
            system: System prompt
            user: User prompt with report and context
            validator_feedback: Optional feedback from Model 2 for retry.
                This is for Model 1's awareness of disagreements, NOT as evidence.
        """
        # Use chat.completions with JSON mode for better free model support
        extra_body = _build_extra_body(self.provider, self.reasoning)
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        # Retry on transient failures (e.g., None content from free tier models)
        # For API overload errors (502, 503, 5002, 5003, etc.), wait 100 seconds
        max_retries = 3
        base_delay = 2.0
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                # Capture actual model used (OpenRouter may route to different model)
                self.last_actual_model = getattr(response, 'model', self.model)
                # Capture token usage
                usage = getattr(response, 'usage', None)
                if usage:
                    self.last_token_usage = {
                        "input_tokens": getattr(usage, 'prompt_tokens', 0),
                        "output_tokens": getattr(usage, 'completion_tokens', 0),
                    }
                text = response.choices[0].message.content
                if text is None:
                    raise RuntimeError("Model returned None content")
                data = self._extract_json(text)
                if data is None:
                    raise RuntimeError(f"Could not extract JSON from inference response: {text[:200]}")
                # Flatten nested PF_OA structure if model incorrectly returns it
                data = _flatten_nested_pf_oa(data)
                return ReportPrediction.model_validate(data)
            except ValidationError as e:
                # Schema validation error - structural failure, don't retry with same input
                last_error = e
                if attempt < max_retries - 1:
                    delay = _get_retry_delay(e, attempt, base_delay)
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Schema validation failed after {max_retries} attempts: {e}\nRaw: {text[:500] if 'text' in locals() else 'N/A'}")
            except Exception as e:
                # Transient API errors (rate limit, None content, network)
                # For API overload errors (502, 503, 5002, 5003, etc.), wait 100 seconds
                last_error = e
                if attempt < max_retries - 1:
                    delay = _get_retry_delay(e, attempt, base_delay)
                    if _is_api_overload_error(e):
                        print(f"  API overload detected (attempt {attempt + 1}/{max_retries}), waiting {delay}s before retry...")
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Inference failed after {max_retries} attempts: {last_error}\nRaw: {text[:500] if 'text' in locals() else 'N/A'}")

    def _extract_json(self, text: str) -> dict | None:
        """Extract a JSON object from text, handling markdown code fences."""
        text = text.strip()
        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting JSON from markdown code fences
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try finding first JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None


class ValidatorModel:
    """Validator model using chat.completions API with JSON mode.

    Supports both OpenRouter and NVIDIA Nemotron endpoints.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 4000,
        provider: str | None = None,
        api_key: str | None = None,
        reasoning: bool | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.provider = provider or _detect_provider(model)
        self.reasoning = reasoning
        self.client = _create_client(self.provider, api_key)
        self.last_token_usage = {"input_tokens": 0, "output_tokens": 0}

    def run(self, system: str, user: str) -> ValidationResult:
        extra_body = _build_extra_body(self.provider, self.reasoning)
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        # Retry on transient failures (e.g., None content from free tier models)
        # For API overload errors (502, 503, 5002, 5003, etc.), wait 100 seconds
        max_retries = 3
        base_delay = 2.0
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                # Capture actual model used (OpenRouter may route to different model)
                self.last_actual_model = getattr(response, 'model', self.model)
                # Capture token usage
                usage = getattr(response, 'usage', None)
                if usage:
                    self.last_token_usage = {
                        "input_tokens": getattr(usage, 'prompt_tokens', 0),
                        "output_tokens": getattr(usage, 'completion_tokens', 0),
                    }
                text = response.choices[0].message.content
                if text is None:
                    raise RuntimeError("Model returned None content")
                data = self._extract_json(text)
                if data is None:
                    raise RuntimeError(f"Could not extract JSON from validation response: {text[:200]}")
                return ValidationResult.model_validate(data)
            except ValidationError as e:
                # Schema validation error - structural failure
                last_error = e
                if attempt < max_retries - 1:
                    delay = _get_retry_delay(e, attempt, base_delay)
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Validation schema failed after {max_retries} attempts: {e}\nRaw: {text[:500] if 'text' in locals() else 'N/A'}")
            except Exception as e:
                # Transient API errors (rate limit, None content, network)
                # For API overload errors (502, 503, 5002, 5003, etc.), wait 100 seconds
                last_error = e
                if attempt < max_retries - 1:
                    delay = _get_retry_delay(e, attempt, base_delay)
                    if _is_api_overload_error(e):
                        print(f"  API overload detected (attempt {attempt + 1}/{max_retries}), waiting {delay}s before retry...")
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Validation failed after {max_retries} attempts: {last_error}\nRaw: {text[:500] if 'text' in locals() else 'N/A'}")

    def _extract_json(self, text: str) -> dict | None:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None


class CriticModel:
    """Critic model (Model 2) using chat.completions API with JSON mode.

    Critiques Model 1's predictions against the report and canonical policy.
    Returns CritiqueResult with structured issues.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 4000,
        provider: str | None = None,
        api_key: str | None = None,
        reasoning: bool | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.provider = provider or _detect_provider(model)
        self.reasoning = reasoning
        self.client = _create_client(self.provider, api_key)
        self.last_token_usage = {"input_tokens": 0, "output_tokens": 0}

    def run(self, system: str, user: str) -> CritiqueResult:
        extra_body = _build_extra_body(self.provider, self.reasoning)
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        max_retries = 3
        base_delay = 2.0
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self.last_actual_model = getattr(response, 'model', self.model)
                # Capture token usage
                usage = getattr(response, 'usage', None)
                if usage:
                    self.last_token_usage = {
                        "input_tokens": getattr(usage, 'prompt_tokens', 0),
                        "output_tokens": getattr(usage, 'completion_tokens', 0),
                    }
                text = response.choices[0].message.content
                if text is None:
                    raise RuntimeError("Model returned None content")
                data = self._extract_json(text)
                if data is None:
                    raise RuntimeError(f"Could not extract JSON from critique response: {text[:200]}")
                return CritiqueResult.model_validate(data)
            except ValidationError as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = _get_retry_delay(e, attempt, base_delay)
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Critique schema failed after {max_retries} attempts: {e}\nRaw: {text[:500] if 'text' in locals() else 'N/A'}")
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = _get_retry_delay(e, attempt, base_delay)
                    if _is_api_overload_error(e):
                        print(f"  API overload detected (attempt {attempt + 1}/{max_retries}), waiting {delay}s before retry...")
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Critique failed after {max_retries} attempts: {last_error}\nRaw: {text[:500] if 'text' in locals() else 'N/A'}")

    def _extract_json(self, text: str) -> dict | None:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None


class JudgeModel:
    """Judge model (Model 3) using chat.completions API with JSON mode.

    Determines workflow action based on Model 1 prediction and Model 2 critique.
    Returns JudgeResult with workflow action only (no clinical labels).
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 4000,
        provider: str | None = None,
        api_key: str | None = None,
        reasoning: bool | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.provider = provider or _detect_provider(model)
        self.reasoning = reasoning
        self.client = _create_client(self.provider, api_key)
        self.last_token_usage = {"input_tokens": 0, "output_tokens": 0}

    def run(self, system: str, user: str) -> JudgeResult:
        extra_body = _build_extra_body(self.provider, self.reasoning)
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        max_retries = 3
        base_delay = 2.0
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self.last_actual_model = getattr(response, 'model', self.model)
                # Capture token usage
                usage = getattr(response, 'usage', None)
                if usage:
                    self.last_token_usage = {
                        "input_tokens": getattr(usage, 'prompt_tokens', 0),
                        "output_tokens": getattr(usage, 'completion_tokens', 0),
                    }
                text = response.choices[0].message.content
                if text is None:
                    raise RuntimeError("Model returned None content")
                data = self._extract_json(text)
                if data is None:
                    raise RuntimeError(f"Could not extract JSON from judge response: {text[:200]}")
                return JudgeResult.model_validate(data)
            except ValidationError as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = _get_retry_delay(e, attempt, base_delay)
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Judge schema failed after {max_retries} attempts: {e}\nRaw: {text[:500] if 'text' in locals() else 'N/A'}")
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = _get_retry_delay(e, attempt, base_delay)
                    if _is_api_overload_error(e):
                        print(f"  API overload detected (attempt {attempt + 1}/{max_retries}), waiting {delay}s before retry...")
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Judge failed after {max_retries} attempts: {last_error}\nRaw: {text[:500] if 'text' in locals() else 'N/A'}")

    def _extract_json(self, text: str) -> dict | None:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None