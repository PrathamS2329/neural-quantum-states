# src/ansatz/__init__.py
#
# Exposes the neural network ansätze at the package level so you can write:
#
#   from src.ansatz import RBM
#
# The CNN is imported conditionally since it's a stretch goal and may not
# always be present.

from .rbm import RBM

__all__ = ["RBM"]
