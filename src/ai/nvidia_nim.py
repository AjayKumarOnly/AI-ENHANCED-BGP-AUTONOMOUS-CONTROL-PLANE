"""
NVIDIA NIM (Inference Microservice) AI Advisor for BGP Autonomous Control Plane.
Integrates NVIDIA NIM foundation models (e.g. meta/llama-3.2-11b-vision-instruct,
nvidia/llama-3.1-nemotron, etc.) to perform deep route anomaly reasoning,
RFC-grounded root cause diagnosis, and autonomous mitigation recommendations.
"""

import os
import json
import logging
import urllib.request
import urllib.error
import ssl
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

logger = logging.getLogger("nvidia_nim_bgp")

DEFAULT_NVIDIA_API_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "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"
)
DEFAULT_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")

CLASS_MAPPING = {
    "Normal": 0,
    "Suspicious": 1,
    "Route Leak": 2,
    "Prefix Hijack": 3
}

ID_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}


class NvidiaNimBGPAdvisor:
    """
    NVIDIA NIM Client for Autonomous BGP Anomaly Classification, Root-Cause
    Analysis, and RFC Mitigation Advisory.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 12.0
    ):
        self.api_key = api_key or DEFAULT_NVIDIA_API_KEY
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.chat_url = f"{self.base_url}/chat/completions"
        self.ssl_context = ssl.create_default_context()

    def is_configured(self) -> bool:
        """Returns True if an API key is available."""
        return bool(self.api_key and not self.api_key.startswith("your_api_key"))

    def analyze_route(
        self,
        prefix: str,
        as_path: List[int],
        origin_as: int,
        expected_origin_as: Optional[int] = None,
        valley_free_violation: bool = False,
        announcements_per_minute: float = 0.0,
        flap_count_5min: int = 0,
        loc_pref_current: int = 100,
        ml_prediction: Optional[str] = None,
        ml_probabilities: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Queries NVIDIA NIM to classify the BGP route state, identify policy/RFC violations,
        and generate explainable mitigation guidance.
        """
        if not self.is_configured():
            return self._heuristic_fallback(
                prefix, as_path, origin_as, expected_origin_as,
                valley_free_violation, flap_count_5min,
                reason="NVIDIA API key not configured. Fallback heuristic active."
            )

        prompt = self._build_prompt(
            prefix=prefix,
            as_path=as_path,
            origin_as=origin_as,
            expected_origin_as=expected_origin_as,
            valley_free_violation=valley_free_violation,
            announcements_per_minute=announcements_per_minute,
            flap_count_5min=flap_count_5min,
            loc_pref_current=loc_pref_current,
            ml_prediction=ml_prediction,
            ml_probabilities=ml_probabilities
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert Autonomous BGP Control Plane Security Engine. "
                        "Analyze telemetry and evaluate Gao-Rexford commercial rules, RFC 1997, "
                        "RFC 6811 (RPKI ROV), and RFC 9234 (Route Leaks). "
                        "Output ONLY a raw, valid JSON object matching the requested schema without markdown backticks."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 300
        }

        try:
            req = urllib.request.Request(
                self.chat_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )

            with urllib.request.urlopen(req, context=self.ssl_context, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"].strip()

                parsed = self._extract_json(content)
                if parsed:
                    parsed["source"] = f"nvidia_nim:{self.model}"
                    parsed["status"] = "success"
                    # Ensure classification_id matches
                    c_name = parsed.get("classification", "Normal")
                    parsed["classification_id"] = CLASS_MAPPING.get(c_name, 0)
                    return parsed

                logger.warning(f"Failed to parse NVIDIA NIM JSON: {content}")
                return self._heuristic_fallback(
                    prefix, as_path, origin_as, expected_origin_as,
                    valley_free_violation, flap_count_5min,
                    reason=f"NVIDIA NIM raw response: {content[:100]}"
                )

        except Exception as e:
            logger.error(f"NVIDIA NIM request failed: {e}")
            return self._heuristic_fallback(
                prefix, as_path, origin_as, expected_origin_as,
                valley_free_violation, flap_count_5min,
                reason=f"NVIDIA NIM API connection error ({type(e).__name__}): {e}"
            )

    def _build_prompt(
        self,
        prefix: str,
        as_path: List[int],
        origin_as: int,
        expected_origin_as: Optional[int],
        valley_free_violation: bool,
        announcements_per_minute: float,
        flap_count_5min: int,
        loc_pref_current: int,
        ml_prediction: Optional[str],
        ml_probabilities: Optional[List[float]]
    ) -> str:
        return (
            f"Evaluate the following BGP Route Snapshot:\n"
            f"- Target Prefix: {prefix}\n"
            f"- Announced AS Path: {as_path}\n"
            f"- Origin AS in Route: AS{origin_as}\n"
            f"- Expected Origin AS: AS{expected_origin_as if expected_origin_as else 'Unknown'}\n"
            f"- Gao-Rexford Valley-Free Violation: {'YES' if valley_free_violation else 'NO'}\n"
            f"- Churn Rate: {announcements_per_minute:.1f} updates/min (Flaps in 5m: {flap_count_5min})\n"
            f"- Current LocalPref: {loc_pref_current}\n"
            f"- Classical ML Baseline: {ml_prediction or 'N/A'} (Probabilities: {ml_probabilities or []})\n\n"
            f"Classify into ONE of: ['Normal', 'Suspicious', 'Route Leak', 'Prefix Hijack'].\n"
            f"Provide recommended autonomous mitigation policy: ['NO_ACTION' (LocalPref 100), 'SOFT_DEPRIORITIZE' (LocalPref 80), 'HARD_DEPRIORITIZE' (LocalPref 50), 'QUARANTINE' (LocalPref 0, RFC 1997 no-export)].\n\n"
            f"Respond with JSON schema:\n"
            f"{{\n"
            f'  "classification": "Normal"|"Suspicious"|"Route Leak"|"Prefix Hijack",\n'
            f'  "confidence": 0.0-1.0,\n'
            f'  "trust_score": 0.0-1.0,\n'
            f'  "recommended_policy": "NO_ACTION"|"SOFT_DEPRIORITIZE"|"HARD_DEPRIORITIZE"|"QUARANTINE",\n'
            f'  "suggested_local_pref": 100|80|50|0,\n'
            f'  "rfc_reference": "RFC ...",\n'
            f'  "explanation": "Clear, concise technical justification"\n'
            f"}}"
        )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Cleans and extracts JSON object from LLM response text."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            clean_json = text[start_idx:end_idx + 1]
            try:
                return json.loads(clean_json)
            except json.JSONDecodeError:
                pass
        return None

    def _heuristic_fallback(
        self,
        prefix: str,
        as_path: List[int],
        origin_as: int,
        expected_origin_as: Optional[int],
        valley_free_violation: bool,
        flap_count_5min: int,
        reason: str
    ) -> Dict[str, Any]:
        """Provides guaranteed continuous operation if NIM service is unavailable."""
        if expected_origin_as and origin_as != expected_origin_as:
            classification = "Prefix Hijack"
            policy = "QUARANTINE"
            pref = 0
            trust = 0.05
            rfc = "RFC 6811 (RPKI ROV Origin Validation Mismatch)"
        elif valley_free_violation:
            classification = "Route Leak"
            policy = "HARD_DEPRIORITIZE"
            pref = 50
            trust = 0.35
            rfc = "RFC 9234 (BGP Route Leak Detection - Valley-Free Violation)"
        elif flap_count_5min > 10:
            classification = "Suspicious"
            policy = "SOFT_DEPRIORITIZE"
            pref = 80
            trust = 0.65
            rfc = "RFC 2439 (BGP Route Flap Damping)"
        else:
            classification = "Normal"
            policy = "NO_ACTION"
            pref = 100
            trust = 0.95
            rfc = "RFC 4271 (Standard BGP-4 Best Path)"

        return {
            "source": "fallback_heuristic",
            "status": "fallback",
            "classification": classification,
            "classification_id": CLASS_MAPPING[classification],
            "confidence": 0.85,
            "trust_score": trust,
            "recommended_policy": policy,
            "suggested_local_pref": pref,
            "rfc_reference": rfc,
            "explanation": f"Automated rule match ({reason})",
            "original_fallback_reason": reason
        }

    def predict_from_features(
        self,
        feature_vector: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[int, np.ndarray]:
        """
        Enables NvidiaNimBGPAdvisor to be used as a drop-in classifier compatible
        with BGPClassifier's predict() interface.
        Returns: (predicted_class_id, probability_distribution)
        """
        origin_changed = bool(feature_vector[2] > 0.5) if len(feature_vector) > 2 else False
        valley_free = bool(feature_vector[8] > 0.5) if len(feature_vector) > 8 else False
        churn = float(feature_vector[4]) if len(feature_vector) > 4 else 0.0

        if origin_changed:
            c_id = 3  # Prefix Hijack
            probs = np.array([0.02, 0.05, 0.03, 0.90])
        elif valley_free:
            c_id = 2  # Route Leak
            probs = np.array([0.05, 0.10, 0.80, 0.05])
        elif churn > 20:
            c_id = 1  # Suspicious
            probs = np.array([0.15, 0.70, 0.10, 0.05])
        else:
            c_id = 0  # Normal
            probs = np.array([0.92, 0.05, 0.02, 0.01])

        return c_id, probs
