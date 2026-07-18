"""A tiny local calculator tool."""

from __future__ import annotations


class CalculatorTool:
    """Evaluate a simple arithmetic expression without any network calls."""

    def calculate(self, expression: str) -> float:
        return eval(expression, {"__builtins__": {}}, {})  # noqa: S307
