"""Defines the ParticleType strategy design pattern."""
import sys
from abc import ABC, abstractmethod
from typing import Dict, Sequence, Optional, Union
from pathlib import Path

import h3
import numpy as np
from netCDF4 import Dataset

sys.path.insert(1, str(Path.cwd() / 'utils'))
from constants import PLD, CP  # noqa: E402
from geneticlineage import GeneticLineage  # noqa: E402
from module import Module  # noqa: E402
from particle import Particle  # noqa: E402
from plot import plot_particles, plot_subpopulations  # noqa: E402
from season import Season  # noqa: E402
from zone import Zone  # noqa: E402


class ParticleType(ABC):
    """Abstract base class for a strategy filtering a particle type based on suitable habitat."""
    @staticmethod
    @abstractmethod
    def get_simulation(season: Season) -> Dataset:
        """Get the simulation file."""
        pass

    @staticmethod
    @abstractmethod
    def plot(particles: Sequence['Particle'], time: int, zones: Optional[Dict[Zone, GeneticLineage]] = None, path: Optional[Union[str, Path]] = None) -> None:
        """Assign particle set final positions at PLD."""
        pass

    @staticmethod
    @abstractmethod
    def filter_initial_positions(particles: Sequence[Particle], zones: Optional[Dict[Zone, GeneticLineage]] = None) -> Sequence[Particle]:
        """Filter the initial positions of a particle set."""
        pass

    @staticmethod
    @abstractmethod
    def filter_final_positions(particles: Sequence[Particle], zones: Optional[Dict[Zone, GeneticLineage]] = None) -> Sequence[Particle]:
        """Filter the final positions of a particle set."""
        pass

    @staticmethod
    @abstractmethod
    def plot(particles: Sequence[Particle], zones: Optional[Dict[Zone, GeneticLineage]] = None, time: int = 0) -> Sequence[Particle]:
        """Plot the particle set."""
        pass

    @staticmethod
    @abstractmethod
    def sort_modules(modules: Sequence[Module], zones: Optional[Dict[Zone, GeneticLineage]] = None) -> Sequence[Module]:
        """Sort module indexing."""
        pass


class Unrestricted(ParticleType):
    """Class of particles that are free to start and end (at PLD) their trajectories anywhere."""
    name: str = 'unrestricted'
    genetics: bool = False
    seed: int = 1

    @staticmethod
    def get_simulation(season: Season):
        """Get the simulation file."""
        return Dataset(Path.cwd() / 'data' / 'simulations' / f'{season.name}.nc')

    @staticmethod
    def plot(particles: Sequence['Particle'], time: int, zones: Dict[Zone, GeneticLineage] = None, path: Optional[Union[str, Path]] = None) -> None:
        """Plot particles."""
        plot_particles(particles, time, path=path)

    @staticmethod
    def filter_initial_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticLineage] = None) -> Sequence[Particle]:
        """No initial filter required."""
        return particles

    @staticmethod
    def filter_final_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticLineage] = None) -> Sequence[Particle]:
        """Assign particle set final positions at PLD."""
        for particle in particles:
            particle.settlement_position = particle.get_position(PLD)
            particle.settlement_time = PLD

        return particles

    @staticmethod
    def sort_modules(modules: Sequence[Module], zones: Optional[Dict[Zone, GeneticLineage]] = None) -> Sequence[Module]:
        """Sort module indexing from south to north."""
        # Get southern points
        south_points = []
        for module in modules:
            south_point = min(h3.h3_to_geo(hexbin.h3)[0] for hexbin in module.hexbins)
            south_points.append((south_point, module.index))

        # Sort in ascending order
        south_points.sort(key=lambda x: x[0])

        # Re-index
        new_modules = []
        for new_index, (_, index) in enumerate(south_points, 1):
            for module in modules:
                if module.index == index:  # Module to be re-indexed found
                    modules.remove(module)  # Remove module from search
                    module.index = new_index  # Re-index module
                    new_modules.append(module)  # Mark module as re-indexed

        return new_modules


class UnrestrictedCP(ParticleType):
    """Class of particles that are free to start and end (at start of CP) their trajectories anywhere."""
    name: str = 'unrestricted_cp'
    genetics: bool = False
    seed: int = 0

    @staticmethod
    def get_simulation(season: Season):
        """Get the simulation file."""
        return Dataset(Path.cwd() / 'data' / 'simulations' / f'{season.name}.nc')

    @staticmethod
    def plot(particles: Sequence['Particle'], time: int, zones: Dict[Zone, GeneticLineage] = None, path: Optional[Union[str, Path]] = None) -> None:
        """Plot particles."""
        plot_particles(particles, time, path=path)

    @staticmethod
    def filter_initial_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticLineage] = None) -> Sequence[Particle]:
        """No initial filter required."""
        return particles

    @staticmethod
    def filter_final_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticLineage] = None) -> Sequence[Particle]:
        """Assign particle set final positions at start of CP."""
        for particle in particles:
            particle.settlement_position = particle.get_position(PLD - CP)
            particle.settlement_time = PLD - CP

        return particles

    @staticmethod
    def sort_modules(modules: Sequence[Module], zones: Optional[Dict[Zone, GeneticLineage]] = None) -> Sequence[Module]:
        """Sort module indexing from south to north."""
        # Get southern points
        south_points = []
        for module in modules:
            south_point = min(h3.h3_to_geo(hexbin.h3)[0] for hexbin in module.hexbins)
            south_points.append((south_point, module.index))

        # Sort in ascending order
        south_points.sort(key=lambda x: x[0])

        # Re-index
        new_modules = []
        for new_index, (_, index) in enumerate(south_points, 1):
            for module in modules:
                if module.index == index:  # Module to be re-indexed found
                    modules.remove(module)  # Remove module from search
                    module.index = new_index  # Re-index module
                    new_modules.append(module)  # Mark module as re-indexed

        return new_modules


class Fixed(ParticleType):
    """Class of particles that are free to start and end (at PLD) their trajectories anywhere."""
    name: str = 'fixed'
    genetics: bool = False
    seed: int = 2

    @staticmethod
    def get_simulation(season: Season):
        """Get the simulation file."""
        return Dataset(Path.cwd() / 'data' / 'simulations' / f'{season.name}_fixed.nc')

    @staticmethod
    def plot(particles: Sequence['Particle'], time: int, zones: Dict[Zone, GeneticLineage] = None, path: Optional[Union[str, Path]] = None) -> None:
        """Plot particles."""
        plot_particles(particles, time, path=path)

    @staticmethod
    def filter_initial_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticLineage] = None) -> Sequence[Particle]:
        """No initial filter required."""
        return particles

    @staticmethod
    def filter_final_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticLineage] = None) -> Sequence[Particle]:
        """Assign particle set final positions at PLD."""
        for particle in particles:
            particle.settlement_position = particle.get_position(PLD)
            particle.settlement_time = PLD

        return particles

    @staticmethod
    def sort_modules(modules: Sequence[Module], zones: Optional[Dict[Zone, GeneticLineage]] = None) -> Sequence[Module]:
        """Sort module indexing from south to north."""
        # Get southern points
        south_points = []
        for module in modules:
            south_point = min(h3.h3_to_geo(hexbin.h3)[0] for hexbin in module.hexbins)
            south_points.append((south_point, module.index))

        # Sort in ascending order
        south_points.sort(key=lambda x: x[0])

        # Re-index
        new_modules = []
        for new_index, (_, index) in enumerate(south_points, 1):
            for module in modules:
                if module.index == index:  # Module to be re-indexed found
                    modules.remove(module)  # Remove module from search
                    module.index = new_index  # Re-index module
                    new_modules.append(module)  # Mark module as re-indexed

        return new_modules


class Restricted(ParticleType):
    """Class of particles that are restricted to suitable habitat spawn and settlement."""
    name: str = 'restricted'
    genetics: bool = True
    seed: int = 0

    @staticmethod
    def get_simulation(season: Season):
        """Get the simulation file."""
        return Dataset(Path.cwd() / 'data' / 'simulations' / f'{season.name}.nc')

    @staticmethod
    def plot(particles: Sequence['Particle'], time: int, zones: Dict[Zone, GeneticLineage], path: Optional[Union[str, Path]] = None) -> None:
        """Plot subpopulations of particles."""
        plot_subpopulations(particles, time, zones=zones, path=path)
        
    @staticmethod
    def filter_initial_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticLineage]) -> Sequence[Particle]:
        """Filter the initial positions of a particle set."""
        # Get initial positions
        lons, lats = np.array(Particle.get_positions(particles, time=0))

        # Filter initial positions
        particles = Particle.filter_unsuitable_particles(particles, lons, lats, zones)
        return particles

    @staticmethod
    def filter_final_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticLineage]) -> Sequence[Particle]:
        """Filter the final positions of a particle set."""
        # Get range of potential final positions
        lons = np.vstack([particle.lons[(PLD - CP):PLD] for particle in particles])
        lats = np.vstack([particle.lats[(PLD - CP):PLD] for particle in particles])

        # Filter final positions
        particles = Particle.filter_unsuitable_particles(particles, lons, lats, zones, competency_period=True)
        return particles

    @staticmethod
    def sort_modules(modules: Sequence[Module], zones: Optional[Dict[Zone, GeneticLineage]] = None) -> Sequence[Module]:
        """Sort module indexing from south to north."""
        # Get southern points
        south_points = []
        for module in modules:
            points = [h3.h3_to_geo(hexbin.h3) for hexbin in module.hexbins]

            # Get genetic lineage of module
            g_zones = [GeneticLineage.get_region(zones, *point[::-1]) for point in points]
            zone = max(set(g_zones), key=g_zones.count)

            south_point = min(points, key=lambda x: x[0])
            south_points.append((zone, south_point[0], module.index))

        # Sort in ascending order
        south_points.sort(key=lambda x: (x[0].value, x[1]))

        # Re-index
        new_modules = []
        for new_index, (_, _, index) in enumerate(south_points, 1):
            for module in modules:
                if module.index == index:  # Module to be re-indexed found
                    modules.remove(module)  # Remove module from search
                    module.index = new_index  # Re-index module
                    new_modules.append(module)  # Mark module as re-indexed

        return new_modules
