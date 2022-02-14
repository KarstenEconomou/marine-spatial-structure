"""Defines the Zone class."""
from enum import IntEnum


class Zone(IntEnum):
    """Zones of genetic lineages."""
    SOUTH = 0
    NORTH_GULF = 1
    NORTH_NL = 2
    UNDEFINED = 3