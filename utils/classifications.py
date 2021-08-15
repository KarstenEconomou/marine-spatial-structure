"""Classifies different characteristics."""
import sys
from abc import ABC, abstractmethod
from enum import Enum, IntEnum, auto
from itertools import permutations
from math import log10
from typing import Tuple, Union, List, Dict, Optional, Sequence
from pathlib import Path

import h3
import numpy as np
import numpy.ma as ma
import pandas as pd
from IPython.core.display import display
from numpy.typing import ArrayLike
from scipy.interpolate import griddata
from shapely.geometry import Polygon, Point

sys.path.insert(1, str(Path.cwd() / 'utils'))
from constants import NUMBER_OF_SEEDS, RESOLUTION, PLD, CP  # noqa: E402


class Season(Enum):
    """Spawning seasons."""
    fall = auto()  # September 15
    spring = auto()  # May 15


class Zone(IntEnum):
    """Genetic regions."""
    SOUTH = 0
    NORTH_GULF = 1
    NORTH_NL = 2
    UNDEFINED = 3


class GeneticZone:
    """Represents a polygon defining a genetic region."""

    def __init__(self, zone: Zone, prob: float, polygon: Polygon) -> None:
        """Initialize a genetic zone from a polygon with a threshold probability."""
        self.zone: Zone = zone
        self.prob: float = prob
        self.poly: Polygon = polygon

        self.sdm: Union[None, pd.DataFrame] = None  # SDM for particles originating from this genetic zone

    def __eq__(self, other: Union['GeneticZone', Zone]) -> bool:
        """Compare zone equality."""
        if isinstance(other, GeneticZone):
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
    def from_file(cls, zone: Zone, prob: float, coordinate_file: Path) -> 'GeneticZone':
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
        """Make array of coordinate points."""
        return np.array((coords_1, coords_2), dtype=tuple).T

    @staticmethod
    def get_region(zones: Dict[Zone, 'GeneticZone'], lon: float, lat: float) -> Zone:
        """Get a region that a point belongs to"""
        for zone in zones.values():
            if zone.contains(lon, lat):
                return zone.zone
        return Zone.UNDEFINED


class Particle:
    """Represents a particle."""

    def __init__(self, trajectory_lons: ArrayLike, trajectory_lats: ArrayLike, trajectory_temps: ArrayLike) -> None:
        """Initialize a particle by its trajectory."""
        self.lons: ArrayLike = trajectory_lons
        self.lats: ArrayLike = trajectory_lats
        self.temps: ArrayLike = trajectory_temps

        self.genetic_lineage: Union[None, Zone] = None
        self.final_position: Tuple[float, float] = self.get_position(-1)

    def assign_genetic_lineage(self, zones: Dict[Zone, GeneticZone], allow_undefined=True) -> None:
        """Assigns the particle to be of a genetic lineage based on its starting position."""
        genetic_lineage = GeneticZone.get_region(zones, *self.get_position(0))

        if not allow_undefined:
            if genetic_lineage is Zone.UNDEFINED:
                raise ValueError

        self.genetic_lineage = genetic_lineage

    def get_position(self, time: int) -> Tuple[float, float]:
        """Get the coordinates of a particle at the given time in its trajectory."""
        return self.lons[time], self.lats[time]

    @staticmethod
    def filter_unsuitable_particles(
            particles: Sequence['Particle'],
            full_lons: np.ndarray,
            full_lats: np.ndarray,
            zones: Dict[Zone, GeneticZone],
            competency_period: bool = False
    ) -> List['Particle']:
        """Find the particles that settled."""
        settled_particles_zone = dict((zone, None) for zone in zones.keys())

        # Find dimensional parameters of particle trajectories
        if competency_period:
            number_of_times = full_lats.shape[1]
        else:
            number_of_times = 1

        for zone in zones.values():
            # Get particles from zone
            particles_zone, indices_zone = (zip(*[(particle, i) for i, particle in enumerate(particles)
                                                  if particle.genetic_lineage is zone.zone]))
            indices_zone = np.array(indices_zone)

            # Use lons and lats of only particles from zone
            if number_of_times > 1:
                lats = np.copy(full_lats[indices_zone, :])
                lons = np.copy(full_lons[indices_zone, :])
            else:
                lats = np.copy(full_lats[indices_zone])
                lons = np.copy(full_lons[indices_zone])

            # Initialize settlement positions
            number_of_particles = len(particles_zone)
            settle_lats = np.zeros(number_of_particles)
            settle_lons = np.zeros(number_of_particles)

            # Define the grid points
            grid_points = GeneticZone.make_points(zone.sdm['lat'], zone.sdm['lon'])

            for time in range(number_of_times):
                # Make query points to check suitability at time
                if number_of_times > 1:
                    query_points = GeneticZone.make_points(lats[:, time], lons[:, time])
                else:
                    query_points = GeneticZone.make_points(lats[:], lons[:])

                # Find the particles that are presently able to settle and in suitable habitat
                settled_particles = np.nonzero(griddata(
                    grid_points,
                    zone.sdm['code'],
                    query_points,
                    method='nearest',
                ).flatten())

                # Record settlement positions
                if number_of_times > 1:
                    settle_lats[settled_particles] = lats[settled_particles, time]
                    settle_lons[settled_particles] = lons[settled_particles, time]

                    if time != number_of_times - 1:
                        # Mark settled particles for exclusion from further consideration
                        lats[settled_particles, (time + 1):] = 0
                        lons[settled_particles, (time + 1):] = 0
                else:
                    settle_lats[settled_particles] = lats[settled_particles]
                    settle_lons[settled_particles] = lons[settled_particles]

            # Identify particles that settled originating from zone
            suitable_particles = np.flatnonzero(settle_lats != 0)
            print(f'{len(suitable_particles)} from {zone.zone} settled')
            settled_particles_zone[zone.zone] = [particles_zone[particle] for particle in suitable_particles]

            # Record settlement positions
            if competency_period:
                for particle, lon, lat in zip(
                        settled_particles_zone[zone.zone],
                        settle_lons[suitable_particles],
                        settle_lats[suitable_particles]
                ):
                    particle.final_position = (lon, lat)

        settled_particles = [particle for zone in zones.keys() for particle in settled_particles_zone[zone]]
        return settled_particles

    @staticmethod
    def leaves_domain(mlds: ArrayLike) -> bool:
        """Determines if a particle left the domain."""
        if 0 in mlds or ma.is_masked(mlds):
            return True
        return False

    @staticmethod
    def get_particles_from_zone(particles: Sequence['Particle'], zone: Zone) -> List['Particle']:
        """Get the subset of particles belonging to a certain genetic lineage."""
        return [particle for particle in particles if particle.genetic_lineage is zone]

    @staticmethod
    def get_positions(
        particles: Sequence['Particle'],
        time: int = 0,
        zone: Optional[Zone] = None
    ) -> Tuple[List[float], List[float]]:
        """Get the positions of a list of particles at time, optionally originating from a zone."""
        if zone is None:
            if time == -1:
                lons, lats = zip(*[particle.final_position for particle in particles])
            else:
                lons, lats = zip(*[particle.get_position(time) for particle in particles])
        else:
            if time == -1:
                lons, lats = zip(
                    *[particle.final_position for particle in particles if particle.genetic_lineage is zone])
            else:
                lons, lats = zip(
                    *[particle.get_position(time) for particle in particles if particle.genetic_lineage is zone])
        return lons, lats


class ParticleType(ABC):
    """Abstract base class for a strategy filtering a particle type based on suitable habitat."""

    @staticmethod
    @abstractmethod
    def filter_initial_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticZone]) -> Sequence[Particle]:
        """Filter the initial positions of a particle set."""
        pass

    @staticmethod
    @abstractmethod
    def filter_final_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticZone]) -> Sequence[Particle]:
        """Filter the final positions of a particle set."""
        pass


class Unrestricted(ParticleType):
    """Class of particles that are free to start and end their trajectories anywhere."""
    name: str = 'unrestricted'

    @staticmethod
    def filter_initial_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticZone]) -> Sequence[Particle]:
        """Filter the initial positions of a particle set."""
        # Unrestricted spawn
        return particles

    @staticmethod
    def filter_final_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticZone]) -> Sequence[Particle]:
        """Filter the final positions of a particle set."""
        # Assign final positions at time PLD
        for particle in particles:
            particle.final_position = particle.get_position(PLD)

        return particles


class Restricted(ParticleType):
    """Class of particles that are restricted to suitable habitat spawn and settlement."""
    name: str = 'restricted'

    @staticmethod
    def filter_initial_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticZone]) -> Sequence[Particle]:
        """Filter the initial positions of a particle set."""
        # Get initial positions
        lons, lats = np.array(Particle.get_positions(particles, time=0))

        # Filter initial positions
        particles = Particle.filter_unsuitable_particles(particles, lons, lats, zones)
        return particles

    @staticmethod
    def filter_final_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticZone]) -> Sequence[Particle]:
        """Filter the final positions of a particle set."""
        # Get range of potential final positions
        lons = np.vstack([particle.lons[(PLD - CP):PLD] for particle in particles])
        lats = np.vstack([particle.lats[(PLD - CP):PLD] for particle in particles])

        # Filter final positions
        particles = Particle.filter_unsuitable_particles(particles, lons, lats, zones, competency_period=True)
        return particles


class Module:
    """Represents a detected community of hexbins."""

    def __init__(self, module_index: int) -> None:
        """Initialize index of module."""
        self.index: int = module_index

        self.coherence: Union[None, float] = None
        self.fortress: Union[None, float] = None
        self.mixing: Union[None, float] = None

    def __eq__(self, other: 'Module') -> bool:
        """Compare module index equality."""
        return self.index == other.index

    def calculate_coherence(self, trajectories: Sequence[Tuple['Hexbin', 'Hexbin']]) -> None:
        """Calculate the module coherence ratio."""
        start_particles = 0
        start_and_end_particles = 0
        for source, target in trajectories:
            if source.module.index == self.index:
                # Trajectory started in module
                start_particles += 1
                if target.module.index == self.index:
                    # Trajectory ended in module
                    start_and_end_particles += 1

        if start_particles == 0:
            self.coherence = 1
        else:
            self.coherence = start_and_end_particles / start_particles

    def calculate_fortress(self, trajectories: Sequence[Tuple['Hexbin', 'Hexbin']]) -> None:
        """Calculate the module fortress ratio."""
        end_and_start_particles = 0
        end_particles = 0
        for source, target in trajectories:
            if target.module.index == self.index:
                # Trajectory ended in module
                end_particles += 1
                if source.module.index == self.index:
                    # Trajectory started in module
                    end_and_start_particles += 1

        if end_particles == 0:
            self.fortress = 1
        else:
            self.fortress = end_and_start_particles / end_particles

    def calculate_mixing(self, transition_matrix: np.ndarray, all_hexbins: Sequence['Hexbin']) -> None:
        """Calculate the module mixing parameter."""
        # Get nodes in module
        module_hexbins = tuple(hexbin for hexbin in all_hexbins if hexbin.module == self)

        if len(module_hexbins) == 1:
            self.mixing = 1
        else:
            # Loop through all permutations of nodes in module
            entropy = 0
            for source, target in permutations(module_hexbins, 2):
                node_prob = transition_matrix[source.int, target.int]

                if node_prob != 0:
                    module_prob = sum(transition_matrix[source.int, hexbin.int] for hexbin in module_hexbins)
                    entropy += (node_prob / module_prob) * log10(node_prob / module_prob)

            self.mixing = -entropy / (len(module_hexbins) * log10(len(module_hexbins)))

    @staticmethod
    def read_clu(
        file: Union[str, Path],
        label_map: Dict[str, int],
        display_clu: bool = False
    ) -> Tuple[Tuple['Module', ...], Tuple['Hexbin', ...]]:
        """Parses the cluster output file of Infomap."""
        clu = pd.read_csv(file, sep=' ', comment='#', names=['node', 'module', 'flow'])

        module_map = dict((module, Module(module)) for module in clu['module'].unique())
        hexbins = tuple(Hexbin.from_integer(node, label_map, module_map[module])
                        for node, module in zip(clu['node'], clu['module']))

        if display_clu:
            display(clu)

        return tuple(module_map.values()), hexbins


class Hexbin:
    """Represents a hexagonal cell: the fundamental unit discretizing the domain."""

    def __init__(
        self,
        h3_index: str,
        label_map: Optional[Dict[str, int]] = None,
        parent_module: Optional[Module] = None
    ) -> None:
        """Initialize by H3 index and define integer index, parent module, and polygon representation."""
        self.h3: str = str(h3_index)
        self.hex: Tuple[Tuple[float, float], ...] = self.__make_hexagon()
        self.module: Module = parent_module

        if label_map is not None:
            self.int: int = label_map[h3_index]
        else:
            self.int: None = None

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

    def contains(self, lon: float, lat: float) -> bool:
        """Return if a query point is within the hexbin."""
        return Polygon(self.hex).contains(Point(lon, lat))

    def get_adjacent_bins(self, domain_bins: Sequence['Hexbin']) -> Tuple[str, ...]:
        """Find adjacent bins that are within the binned domain."""
        ring = h3.k_ring(self.h3, 1)
        return tuple(hexbin for hexbin in ring if hexbin in domain_bins and hexbin != self.h3)

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
        parent_module: Optional[Module] = None
    ) -> 'Hexbin':
        """Initialize by integer index."""
        h3_index = list(label_map.keys())[list(label_map.values()).index(integer_index)]
        return cls(h3_index, label_map, parent_module)

    @classmethod
    def bin_particles(
        cls,
        particles: Sequence[Particle],
        res: int = RESOLUTION,
        allow_duplicates: bool = True
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
        allow_duplicates: bool = True
    ) -> List[str]:
        """Generate the H3 indices of bins that particle positions are in."""
        h3_indices = [h3.geo_to_h3(lat, lon, res) for lat, lon in zip(lats, lons)]
        if allow_duplicates:
            return h3_indices
        else:
            return list(dict.fromkeys(h3_indices))
