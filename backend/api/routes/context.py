"""
Backend context store — holds the user's active context flags in memory.
The replay loop reads these and injects them into every posted reading,
so risk_service.py can adjust thresholds accordingly.
"""
from fastapi import APIRouter

router = APIRouter()

_context_flags: list[str] = []


def get_context_flags() -> list[str]:
    """Called by the replay loop to include context in each reading payload."""
    return list(_context_flags)


@router.get("/context")
def get_context():
    return {"context_flags": _context_flags}


@router.post("/context")
def set_context(body: dict):
    global _context_flags
    _context_flags = [str(f) for f in body.get("context_flags", [])]
    return {"context_flags": _context_flags}
