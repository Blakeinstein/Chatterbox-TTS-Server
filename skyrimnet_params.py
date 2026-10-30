# File: skyrimnet_params.py
# Parameter resolution for SkyrimNet / Gradio API compatibility.
# Ported from chatterbox-langdon-reference shared_config.get_tts_params.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TTS_PARAMS = {
    "TEMPERATURE": 1.0,
    "MIN_P": 0.05,
    "TOP_P": 1.0,
    "SPEED": 1.0,
    "REPETITION_PENALTY": 1.2,
    "CFG_WEIGHT": 0.35,
    "EXAGGERATION": 0.9,
}

_CONFIG_FILE_PATH = Path("skyrimnet_config.txt")
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_TTS_PARAMS_CACHE: Dict[Tuple[Any, bool], Dict[str, float]] = {}


def load_skyrimnet_config() -> Dict[str, Any]:
    """Load overrides from skyrimnet_config.txt (cached)."""
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    config_overrides: Dict[str, Any] = {}
    if _CONFIG_FILE_PATH.exists():
        try:
            with open(_CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key not in (
                        "temperature",
                        "top_p",
                        "min_p",
                        "speed",
                        "repetition_penalty",
                        "cfg_weight",
                        "exaggeration",
                    ):
                        continue
                    if value.lower() == "api":
                        config_overrides[key] = "api"
                    elif value.lower() != "default":
                        try:
                            config_overrides[key] = float(value)
                        except ValueError:
                            logger.warning("Invalid value for %s: %s", key, value)
        except Exception as e:
            logger.warning("Error loading config file %s: %s", _CONFIG_FILE_PATH, e)

    _CONFIG_CACHE = config_overrides
    return _CONFIG_CACHE


def get_tts_params(
    payload_params: Optional[Dict[str, Any]] = None, override_flag: bool = False
) -> Dict[str, float]:
    """
    Resolve TTS parameters based on priority:
    1. DEFAULT_TTS_PARAMS
    2. Config file numeric values
    3. Config file "api" mode + payload values
    4. override_flag + payload values
    """
    config = load_skyrimnet_config()
    params = DEFAULT_TTS_PARAMS.copy()

    cache_key = (
        tuple(sorted((payload_params or {}).items())),
        override_flag,
    )
    if cache_key in _TTS_PARAMS_CACHE:
        return _TTS_PARAMS_CACHE[cache_key].copy()

    result = {
        "temperature": params["TEMPERATURE"],
        "top_p": params["TOP_P"],
        "min_p": params["MIN_P"],
        "speed": params["SPEED"],
        "repetition_penalty": params["REPETITION_PENALTY"],
        "cfg_weight": params["CFG_WEIGHT"],
        "exaggeration": params["EXAGGERATION"],
    }

    if payload_params is None:
        payload_params = {}

    for param_name in result:
        config_value = config.get(param_name)
        payload_value = payload_params.get(param_name)

        if override_flag and payload_value is not None:
            result[param_name] = payload_value
        elif config_value == "api" and payload_value is not None:
            result[param_name] = payload_value
        elif config_value is not None and config_value != "api":
            result[param_name] = config_value

    _TTS_PARAMS_CACHE[cache_key] = result
    return result
