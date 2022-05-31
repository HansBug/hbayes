from enum import IntEnum, auto

import enum_tools


@enum_tools.documentation.document_enum
class OptimizationEvent(IntEnum):
    """
    Overview:
        Events during optimization.
    """
    START = auto()  # doc: Start event.
    STEP = auto()  # doc: Step event.
    SKIP = auto()  # doc: Skipped event.
    END = auto()  # doc: End event.
