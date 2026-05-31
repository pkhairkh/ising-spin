"""Utility functions for the Integer Language Model."""


def get_rss_mb() -> int:
    """Get current process RSS in MB."""
    try:
        import resource
        import sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return rss // (1024 * 1024)
        return rss // 1024
    except Exception:
        return 0
