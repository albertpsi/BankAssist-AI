"""HTTP surface. Later labs add routes here; the app factory stays the entry point."""

from bankassist.api.app import create_app

__all__ = ["create_app"]
