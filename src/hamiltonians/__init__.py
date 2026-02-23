# src/hamiltonians/__init__.py
#
# This makes `hamiltonians` a Python package and exposes the two Hamiltonian
# classes at the package level so you can write:
#
#   from src.hamiltonians import IsingHamiltonian, HeisenbergHamiltonian
#
# instead of the longer import path. Clean public API.

from .ising import IsingHamiltonian
from .heisenberg import HeisenbergHamiltonian

__all__ = ["IsingHamiltonian", "HeisenbergHamiltonian"]
