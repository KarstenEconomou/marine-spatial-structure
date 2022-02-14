"""Defines the GeneticLineage class."""
import sys
from typing import Dict, Union
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from shapely.geometry import Polygon, Point

sys.path.insert(1, str(Path.cwd() / 'utils'))
from zone import Zone  # noqa: E402


class GeneticLineage:
    """Represents a polygon defining a genetic region."""

    def __init__(self, zone: Zone, prob: float, polygon: Polygon) -> None:
        """Initialize a genetic zone from a polygon with a threshold probability."""
        self.zone: Zone = zone
        self.prob: float = prob
        self.poly: Polygon = polygon

        self.sdm: Union[None, pd.DataFrame] = None  # SDM for particles originating from this genetic zone

    def __eq__(self, other: Union['GeneticLineage', Zone]) -> bool:
        """Compare zone equality."""
        if isinstance(other, GeneticLineage):
            return self.zone == other.zone
        elif isinstance(other, Zone):
            return self.zone == Zone

    def contains(self, lon: float, lat: float) -> bool:
        """Return if a query point is within the defining polygon."""
        return self.poly.contains(Point(lon, lat))

    def associate_sdm(self, codes: ArrayLike, lons: ArrayLike, lats: ArrayLike) -> None:
        """Associate a grid """
        self.sdm = pd.DataFrame({'code': codes, 'lon': lons, 'lat': lats})

    @classmethod
    def from_file(cls, zone: Zone, prob: float, coordinate_file: Path) -> 'GeneticLineage':
        """Initialize a genetic zone from a file containing coordinate points."""
        polygon = cls.make_polygon(coordinate_file)
        return cls(zone, prob, polygon)

    @staticmethod
    def make_polygon(file: Path) -> Polygon:
        """Create a polygon from a file containing coordinates."""
        coordinates = np.loadtxt(str(file))
        return Polygon([point for point in coordinates])

    @staticmethod
    def make_points(coords_1: ArrayLike, coords_2: ArrayLike) -> np.ndarray:
        """Make an array of coordinate points."""
        return np.array((coords_1, coords_2), dtype=tuple).T

    @staticmethod
    def get_region(zones: Dict[Zone, 'GeneticLineage'], lon: float, lat: float) -> Zone:
        """Get a region that a point belongs to"""
        for zone in zones.values():
            if zone.contains(lon, lat):
                return zone.zone
        return Zone.UNDEFINED
