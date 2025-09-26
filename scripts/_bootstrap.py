import os
import sys


def bootstrap():
    """Ensure project root in sys.path and enable HF Transfer.

    - Adds the repo root (parent of this file's dir) to sys.path so
      local packages like `models` can be imported when running scripts.
    - Sets HF_HUB_ENABLE_HF_TRANSFER=1 to enable faster downloads if
      `hf_transfer` is installed. Silently no-ops if unavailable.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.append(root)

    # Enable fast HF Hub transfers when the optional package is present
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    # Silence fork/parallelism warning from huggingface/tokenizers in workers
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import hf_transfer  # noqa: F401
    except Exception:
        # If not installed, the env var is harmless and downloads fall back
        pass
