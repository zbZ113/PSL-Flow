from __future__ import annotations


ALLOWED_ROUTES = ("psl_flow",)


def validate_route(route: str | None) -> str:
    resolved = "psl_flow" if route in (None, "") else str(route)
    if resolved not in ALLOWED_ROUTES:
        raise ValueError(f"Unsupported route `{resolved}`. Allowed routes: {', '.join(ALLOWED_ROUTES)}")
    return resolved
