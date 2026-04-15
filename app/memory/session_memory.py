from typing import Dict, Any, List


class SessionMemory:
    """Lightweight in-memory session state for the CLI loop."""

    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}
        self.interactions: List[Dict[str, str]] = []

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def update(self, values: Dict[str, Any]) -> None:
        self.data.update(values)

    def add_interaction(self, role: str, content: str) -> None:
        self.interactions.append({"role": role, "content": content})
