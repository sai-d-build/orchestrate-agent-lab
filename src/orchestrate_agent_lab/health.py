"""Minimal health check for Phase 0."""


def health_check() -> dict[str, str]:
    """Return a deterministic health response."""
    return {"status": "ok"}
