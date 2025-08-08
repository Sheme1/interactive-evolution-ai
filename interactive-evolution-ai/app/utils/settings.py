"""Settings utility class for working with the config/settings.ini file.

This class provides typed accessors and mutators for configuration values.
It wraps Python's built-in ``configparser`` while offering a convenient API
for reading and persisting changes. All PR-level configuration lives in
``interactive-evolution-ai/config``.

PEP-8 and type-hints are respected throughout.
"""
from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any, Union


class Settings:
    """Typed wrapper around ``configparser.ConfigParser``.

    Parameters
    ----------
    config_path:
        Optional explicit path to the ``settings.ini`` file. If *None*, a
        default path relative to the repository root is assumed:  ::

            <repo>/interactive-evolution-ai/config/settings.ini
    """

    def __init__(self, config_path: Union[str, Path, None] = None) -> None:
        self.config_path: Path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else Path(__file__).resolve().parent.parent.parent / "config" / "settings.ini"
        )

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")

        # Disable interpolation to allow literal '%' signs in values (e.g., "5%").
        # The default interpolation can cause an InterpolationSyntaxError.
        self._parser = configparser.ConfigParser(interpolation=None)
        # Preserve case of keys when writing back.
        self._parser.optionxform = str  # type: ignore[attr-defined]
        self._parser.read(self.config_path)

    # ---------------------------------------------------------------------
    # Generic getters
    # ---------------------------------------------------------------------
    def get_int(self, section: str, option: str) -> int:
        """Return *int* value from the config."""
        return self._parser.getint(section, option)

    def get_float(self, section: str, option: str) -> float:
        """Return *float* value from the config."""
        return self._parser.getfloat(section, option)

    def get_str(self, section: str, option: str) -> str:
        """Return *str* value from the config."""
        return self._parser.get(section, option)

    def get_bool(self, section: str, option: str) -> bool:  # noqa: D401
        """Return *bool* value (yes/no, true/false, 1/0)."""
        return self._parser.getboolean(section, option, fallback=False)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def set_value(self, section: str, option: str, value: Any) -> None:
        """Set *value* for a *section/option* pair and persist it to disk."""
        if section not in self._parser.sections():
            self._parser.add_section(section)

        self._parser.set(section, option, str(value))
        with self.config_path.open("w", encoding="utf-8") as fp:
            self._parser.write(fp)

    # Convenience wrappers ------------------------------------------------
    def set_int(self, section: str, option: str, value: int) -> None:  # noqa: D401
        """Shortcut for :py:meth:`set_value` accepting an *int*."""
        self.set_value(section, option, value)

    def set_float(self, section: str, option: str, value: float) -> None:  # noqa: D401
        """Shortcut for :py:meth:`set_value` accepting a *float*."""
        self.set_value(section, option, value)

    def set_str(self, section: str, option: str, value: str) -> None:  # noqa: D401
        """Shortcut for :py:meth:`set_value` accepting a *str*."""
        self.set_value(section, option, value)

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"<Settings path='{self.config_path}'>"