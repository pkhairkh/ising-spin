"""Process memory measurement utilities."""

from __future__ import annotations


def get_rss_mb() -> int:
    """Return current process RSS in MiB (0 if unavailable).

    Tries ``resource.getrusage`` first (macOS / Linux), then falls back to
    reading ``/proc/<pid>/status`` on Linux.  Returns 0 when neither method
    is available or when both raise.
    """
    # Primary: resource module (works on most Unix)
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024  # KB -> MB
    except Exception:
        pass

    # Fallback: /proc on Linux
    try:
        import os

        with open(f"/proc/{os.getpid()}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024  # KB -> MB
    except Exception:
        pass

    return 0
