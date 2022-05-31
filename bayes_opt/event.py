from enum import IntEnum

import enum_tools


@enum_tools.documentation.document_enum
class OptimizationEvent(IntEnum):
    """
    Overview:
        Events during optimization.
    """
    START = 1  # doc: A person called Alice
    STEP = 2  # doc: Step event.
    END = 3  # doc: End event.
