"""Compatibility entry point for the multi-view BERT disaster-tweet pipeline.

The original repository stored a direct Colab-to-Python export in this file.
The maintained implementation now lives in ``train_ensemble.py``.
"""

from train_ensemble import main


if __name__ == "__main__":
    main()
