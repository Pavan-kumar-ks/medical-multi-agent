"""Tool registry for MedOrchestrator.

Every external capability (web search, maps, diagnosis, scraper) is registered
here with its JSON-schema for both inputs and outputs.  Agents call tools via
`registry.call(tool_name, **kwargs)` which:
  1. Validates inputs against the input schema
  2. Calls the underlying function
  3. Validates outputs against the output schema
  4. Logs validation failures (non-blocking)
  5. Returns a standardised result dict

This makes tool use deterministic and debuggable.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from app.observability.logger import get_logger
    _log = get_logger("medorchestrator.tool_registry")
except Exception:
    _log = logging.getLogger("medorchestrator.tool_registry")

try:
    from app.tools.validator import validate_schema
except Exception:
    def validate_schema(data, schema):  # type: ignore
        return True, []


# ── Tool definition ───────────────────────────────────────────────────────────

@dataclass
class ToolDefinition:
    name:          str
    description:   str
    func:          Optional[Callable]
    input_schema:  Dict[str, Any]
    output_schema: Dict[str, Any]
    tags:          List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":          self.name,
            "description":   self.description,
            "input_schema":  self.input_schema,
            "output_schema": self.output_schema,
            "tags":          self.tags,
        }


# ── Registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    """Singleton registry of all MedOrchestrator tools."""

    _instance: Optional[ToolRegistry] = None

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._tools: Dict[str, ToolDefinition] = {}
            inst._register_defaults()
            cls._instance = inst
        return cls._instance

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        _log.debug(f"Tool registered: {tool.name}", extra={"tool": tool.name})

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    # ── Validated call ────────────────────────────────────────────────────

    def call(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a registered tool with input/output schema validation.

        Returns
        -------
        dict  {ok, result, error, tool, duration_ms}
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            _log.error(f"Unknown tool: {tool_name}")
            return {"ok": False, "result": None, "error": f"Unknown tool: {tool_name}", "tool": tool_name}

        if tool.func is None:
            return {"ok": False, "result": None, "error": f"Tool '{tool_name}' has no implementation", "tool": tool_name}

        # ── Validate input ────────────────────────────────────────────────
        ok, errors = validate_schema(kwargs, tool.input_schema)
        if not ok:
            _log.warning(
                f"Tool '{tool_name}' input validation failed",
                extra={"tool": tool_name, "errors": errors},
            )

        # ── Execute ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            result = tool.func(**kwargs)
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)

            # ── Validate output ───────────────────────────────────────────
            out_ok, out_errors = validate_schema(result, tool.output_schema)
            if not out_ok:
                _log.warning(
                    f"Tool '{tool_name}' output validation failed",
                    extra={"tool": tool_name, "errors": out_errors},
                )

            _log.debug(
                f"Tool '{tool_name}' OK ({duration_ms}ms)",
                extra={"tool": tool_name, "duration_ms": duration_ms},
            )
            return {"ok": True, "result": result, "error": None, "tool": tool_name, "duration_ms": duration_ms}

        except Exception as exc:
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            _log.error(
                f"Tool '{tool_name}' raised: {exc}",
                extra={"tool": tool_name, "error": str(exc)},
            )
            return {"ok": False, "result": None, "error": str(exc), "tool": tool_name, "duration_ms": duration_ms}

    # ── Default tool registrations ────────────────────────────────────────

    def _register_defaults(self) -> None:

        # ── web_search ────────────────────────────────────────────────────
        try:
            from app.tools.web_search import web_search as _web_search
            _ws_func = _web_search
        except Exception:
            _ws_func = None

        self.register(ToolDefinition(
            name        = "web_search",
            description = "Search the web using DuckDuckGo (no API key required).",
            func        = _ws_func,
            tags        = ["search", "internet"],
            input_schema = {
                "type": "object",
                "properties": {
                    "query":       {"type": "string",  "description": "Search query string"},
                    "max_results": {"type": "integer", "description": "Max results to return", "default": 5},
                },
                "required": ["query"],
            },
            output_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title":   {"type": "string"},
                        "url":     {"type": "string"},
                        "snippet": {"type": "string"},
                    },
                    "required": ["title", "url"],
                },
            },
        ))

        # ── find_nearby_hospitals ─────────────────────────────────────────
        try:
            from app.tools.mcp_maps import find_nearby_hospitals as _fnh
            _fnh_func = _fnh
        except Exception:
            _fnh_func = None

        self.register(ToolDefinition(
            name        = "find_nearby_hospitals",
            description = "Find hospitals within a radius using Overpass / Mappls.",
            func        = _fnh_func,
            tags        = ["maps", "hospital"],
            input_schema = {
                "type": "object",
                "properties": {
                    "lat":      {"type": "number"},
                    "lng":      {"type": "number"},
                    "radius_m": {"type": "integer", "default": 5000},
                },
                "required": ["lat", "lng"],
            },
            output_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":    {"type": "string"},
                        "address": {"type": ["string", "null"]},
                        "lat":     {"type": ["number", "null"]},
                        "lng":     {"type": ["number", "null"]},
                    },
                    "required": ["name"],
                },
            },
        ))

        # ── geocode_location ──────────────────────────────────────────────
        try:
            from app.tools.mcp_maps import geocode_location as _geo
            _geo_func = _geo
        except Exception:
            _geo_func = None

        self.register(ToolDefinition(
            name        = "geocode_location",
            description = "Convert a location text string to lat/lng coordinates.",
            func        = _geo_func,
            tags        = ["maps", "geocoding"],
            input_schema = {
                "type": "object",
                "properties": {
                    "location_text": {"type": "string"},
                },
                "required": ["location_text"],
            },
            output_schema = {
                "type": "object",
                "properties": {
                    "lat":       {"type": "number"},
                    "lng":       {"type": "number"},
                    "formatted": {"type": "string"},
                },
                "required": ["lat", "lng"],
            },
        ))

        # ── scrape_doctors ────────────────────────────────────────────────
        try:
            from app.scraper.runner import scrape_doctors as _sd
            _sd_func = _sd
        except Exception:
            _sd_func = None

        self.register(ToolDefinition(
            name        = "scrape_doctors",
            description = "Scrape doctor profiles from a hospital website.",
            func        = _sd_func,
            tags        = ["scraping", "doctors"],
            input_schema = {
                "type": "object",
                "properties": {
                    "hospital_name": {"type": "string"},
                    "specialty":     {"type": "string"},
                    "location":      {"type": "string", "default": ""},
                    "start_url":     {"type": ["string", "null"], "default": None},
                },
                "required": ["hospital_name", "specialty"],
            },
            output_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":           {"type": "string"},
                        "specialty":      {"type": "string"},
                        "clinic_hospital":{"type": "string"},
                        "source_url":     {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
        ))


# ── Module-level singleton ────────────────────────────────────────────────────
registry = ToolRegistry()
