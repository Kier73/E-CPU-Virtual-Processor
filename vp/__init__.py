"""E-CPU Virtual Processor: a recoverable, externally taskable observer-host adapter.

The computation belongs to the installed E-CPU native package. This package
owns only the external command protocol and its portable journal.
"""
from .core import ProcessorRefused, bootstrap, load, run

__all__ = ["ProcessorRefused", "bootstrap", "load", "run"]
