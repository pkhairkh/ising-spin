"""Utility functions for the Integer Language Model."""

import resource


def get_rss_mb() -> int:
    """Get current process RSS in MB."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except Exception:
        return 0
