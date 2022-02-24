"""Defines the Hexbin class."""
import sys
from typing import Optional, Dict, Tuple, Union, Sequence, List, TYPE_CHECKING
from pathlib import Path

import h3
from numpy.typing import ArrayLike
from shapely.geometry import Polygon, Point

sys.path.insert(1, str(Path.cwd() / 'utils'))
from constants import RESOLUTION, NUMBER_OF_SEEDS  # noqa: E402
from particle import Particle  # noqa: E402
if TYPE_CHECKING:
    from module import Module  # noqa: E402


class Hexbin:
    """Represents a hexagonal cell: the fundamental unit discretizing the domain."""

    def __init__(
        self,
        h3_index: str,
        label_map: Optional[Dict[str, int]] = None,
        parent_module: Optional['Module'] = None,
        flow: Optional['float'] = None
    ) -> None:
        """Initialize by H3 index and define integer index, parent module, and polygon representation."""
        self.h3: str = str(h3_index)
        self.hex: Tuple[Tuple[float, float], ...] = self.__make_hexagon()
        self.module: Module = parent_module
        self.flow: float = flow

        if label_map is not None:
            self.int = label_map[h3_index]
        else:
            self.int = None

        self.boundary_persistence: Union[None, float] = None

    def __repr__(self) -> str:
        """String representation."""
        label = {self.h3: self.int}
        return f'Hexbin({self.h3}, {label}, Module({self.module.index}))'

    def __str__(self) -> str:
        """Humananist string representation."""
        return f'Hexbin {self.h3}/{self.int} of module {self.module.index}'

    def __eq__(self, other: Union['Hexbin', str]):
        """Compare H3 index equality."""
        if isinstance(other, Hexbin):
            return self.h3 == other.h3
        elif isinstance(other, str):
            return self.h3 == other

    def __lt__(self, other: 'Hexbin') -> bool:
        """Compare h3 indices on less than call."""
        return self.h3 < other.h3

    def contains(self, lon: float, lat: float) -> bool:
        """Return if a query point is within the hexbin."""
        return Polygon(self.hex).contains(Point(lon, lat))

    def get_adjacent_bins(
        self,
        other_bins: Sequence['Hexbin'],
        hexbin_dict: Optional[Dict[str, 'Hexbin']] = None,
        k: int = 1
    ) -> Union[Tuple[str, ...], Tuple['Hexbin', ...]]:
        """Find adjacent bins that are in the given bin list."""
        other_bins = tuple(other_bin.h3 for other_bin in other_bins)

        ring = h3.k_ring(self.h3, k)
        adjacent_bins = tuple(hexbin for hexbin in ring if hexbin in other_bins and hexbin != self.h3)

        if hexbin_dict is not None:
            return tuple(hexbin_dict[hexbin] for hexbin in adjacent_bins)
        return adjacent_bins

    def is_boundary(self, domain: Sequence['Hexbin'], hexbin_dict: Dict[str, 'Hexbin']) -> bool:
        """Check if hexbin is acting as a boundary."""
        adjacent_bins = self.get_adjacent_bins(domain)
        for h3_index in adjacent_bins:
            hexbin = hexbin_dict[h3_index]
            if self.module.index != hexbin.module.index:
                return True
        return False

    def set_boundary_persistence(self, boundaries: Sequence['Hexbin']) -> None:
        """Calculate fraction of times acting as a boundary."""
        self.boundary_persistence = boundaries.count(self)/NUMBER_OF_SEEDS

    def __make_hexagon(self) -> Tuple[Tuple[float, float], ...]:
        """Define polygon representation by six vertices."""
        hexagon = tuple(h3.h3_to_geo_boundary(self.h3))

        # Reverse (lat, lon) to (lon, lat) — analogous to (x, y) — for plotting
        return tuple(point[::-1] for point in hexagon)

    @classmethod
    def from_integer(
        cls,
        integer_index: int,
        label_map: Dict[str, int],
        parent_module: Optional['Module'] = None,
        flow: Optional[float] = None,
    ) -> 'Hexbin':
        """Initialize by integer index."""
        h3_index = list(label_map.keys())[list(label_map.values()).index(integer_index)]
        return cls(h3_index, label_map, parent_module, flow)

    @classmethod
    def bin_particles(
        cls,
        particles: Sequence[Particle],
        res: int = RESOLUTION,
        allow_duplicates: bool = True,
    ) -> Tuple[List[str], List[str]]:
        lons, lats = Particle.get_positions(particles, time=0)
        source_nodes = cls.assign_h3_indices(lats, lons, res=res, allow_duplicates=allow_duplicates)

        lons, lats = Particle.get_positions(particles, time=-1)
        target_nodes = cls.assign_h3_indices(lats, lons, res=res, allow_duplicates=allow_duplicates)

        return source_nodes, target_nodes

    @staticmethod
    def create_label_mapping(h3_indices: ArrayLike) -> Dict[str, int]:
        """Create a label mapping that allows interchanging integer and H3 indices."""
        return dict((string, integer) for integer, string in enumerate(h3_indices))

    @staticmethod
    def assign_h3_indices(
        lats: ArrayLike,
        lons: ArrayLike,
        res: int = RESOLUTION,
        allow_duplicates: bool = True,
    ) -> List[str]:
        """Generate the H3 indices of bins that particle positions are in."""
        h3_indices = [h3.geo_to_h3(lat, lon, res) for lat, lon in zip(lats, lons)]
        if allow_duplicates:
            return h3_indices
        else:
            return list(dict.fromkeys(h3_indices))
