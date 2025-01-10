"""Initialization file for the RAG system configuration module."""

from .config import Config, PathConfig, SplitterConfig, WaterLevelConfig, LLMConfig

# Global config instance
config = Config()

__all__ = [
    "Config",
    "PathConfig",
    "SplitterConfig",
    "WaterLevelConfig",
    "LLMConfig",
    "config",
]
