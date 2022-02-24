"""Defines the ParticleType strategy design pattern."""
import sys
from abc import ABC, abstractmethod
from typing import Dict, Sequence, Optional
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

sys.path.insert(1, str(Path.cwd() / 'utils'))
from constants import PLD, CP  # noqa: E402
from geneticlineage import GeneticLineage  # noqa: E402
from particle import Particle  # noqa: E402
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


class Unrestricted(ParticleType):
    """Class of particles that are free to start and end their trajectories anywhere."""
    name: str = 'unrestricted'

    @staticmethod
    def get_simulation(season: Season):
        """Get the simulation file."""
        return Dataset(Path.cwd() / 'data' / 'simulations' / f'{season.name}.nc')

    @staticmethod
    def filter_initial_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticLineage] = None) -> Sequence[Particle]:
        """Filter the initial positions of a particle set."""
        # Unrestricted spawn
        return particles

    @staticmethod
    def filter_final_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticLineage] = None) -> Sequence[Particle]:
        """Filter the final positions of a particle set."""
        # Assign final positions at time PLD
        for particle in particles:
            particle.final_position = particle.get_position(PLD)

        return particles

class Fixed(ParticleType):
    """Class of particles that are free to start and end their trajectories anywhere."""
    name: str = 'fixed'

    @staticmethod
    def get_simulation(season: Season):
        """Get the simulation file."""
        return Dataset(Path.cwd() / 'data' / 'simulations' / f'{season.name}_fixed.nc')

    @staticmethod
    def filter_initial_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticLineage] = None) -> Sequence[Particle]:
        """Filter the initial positions of a particle set."""
        # Unrestricted spawn
        return particles

    @staticmethod
    def filter_final_positions(particles: Sequence[Particle], zones: Dict[Zone, GeneticLineage] = None) -> Sequence[Particle]:
        """Filter the final positions of a particle set."""
        # Assign final positions at time PLD
        for particle in particles:
            particle.final_position = particle.get_position(PLD)

        return particles


class Restricted(ParticleType):
    """Class of particles that are restricted to suitable habitat spawn and settlement."""
    name: str = 'restricted'

    @staticmethod
    def get_simulation(season: Season):
        """Get the simulation file."""
        return Dataset(Path.cwd() / 'data' / 'simulations' / f'{season.name}.nc')
        
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

