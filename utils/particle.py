"""Defines the Particle class."""
import sys
from typing import Union, Tuple, Dict, Sequence, List, Optional
from pathlib import Path

import numpy as np
import numpy.ma as ma
from numpy.typing import ArrayLike
from scipy.interpolate import griddata

sys.path.insert(1, str(Path.cwd() / 'utils'))
from constants import PLD, LEFT_BOUND, RIGHT_BOUND, TOP_BOUND, BOTTOM_BOUND
from geneticlineage import GeneticLineage
from zone import Zone


class Particle:
    """Represents a particle."""

    def __init__(self, lons: Sequence[float], lats: Sequence[float], mlds: Sequence[float]) -> None:
        """Initialize a particle by its trajectory."""
        self.lons: Sequence[float] = lons
        self.lats: Sequence[float] = lats
        self.mlds: Sequence[float] = mlds

        self.genetic_lineage: Optional[Zone] = None
        self.settlement_time: Optional[int] = None
        self.settlement_position: Optional[Tuple[float, float]] = None

    def assign_genetic_lineage(self, zones: Dict[Zone, GeneticLineage], allow_undefined=True) -> None:
        """Assigns the particle to be of a genetic lineage based on its starting position."""
        genetic_lineage = GeneticLineage.get_region(zones, *self.get_position(0))

        if not allow_undefined:
            if genetic_lineage is Zone.UNDEFINED:
                raise ValueError

        self.genetic_lineage = genetic_lineage

    def get_position(self, time: int) -> Tuple[float, float]:
        """Get the coordinates of a particle at the given time in its trajectory."""
        return (self.lons[time], self.lats[time])

    def is_beached(self) -> bool:
        """Determines if a particle was beached or out of field domain."""
        settle_mld = self.mlds[int(self.settlement_time)]
        if settle_mld == 0 or ma.is_masked(settle_mld):
            return True
        return False

    @staticmethod
    def filter_unsuitable_particles(
            particles: Sequence['Particle'],
            full_lons: np.ndarray,
            full_lats: np.ndarray,
            zones: Dict[Zone, GeneticLineage],
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

            # Initialize settlement positions & times
            number_of_particles = len(particles_zone)
            settle_lats = np.zeros(number_of_particles)
            settle_lons = np.zeros(number_of_particles)
            settle_times = np.zeros(number_of_particles)

            # Define the grid points
            grid_points = GeneticLineage.make_points(zone.sdm['lat'], zone.sdm['lon'])

            for time in range(number_of_times):
                # Make query points to check suitability at time
                if number_of_times > 1:
                    query_points = GeneticLineage.make_points(lats[:, time], lons[:, time])
                else:
                    query_points = GeneticLineage.make_points(lats[:], lons[:])

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
                settle_times[settled_particles] = (PLD - number_of_times) + (time + 1)

            # Identify particles that settled originating from zone
            suitable_particles = np.flatnonzero(settle_lats != 0)
            print(f'{len(suitable_particles)} from {zone.zone} settled')
            settled_particles_zone[zone.zone] = [particles_zone[particle] for particle in suitable_particles]

            # Record settlement positions
            if competency_period:
                for particle, lon, lat, time in zip(
                        settled_particles_zone[zone.zone],
                        settle_lons[suitable_particles],
                        settle_lats[suitable_particles],
                        settle_times[suitable_particles]
                ):
                    particle.settlement_position = (lon, lat)
                    particle.settlement_time = time

        settled_particles = [particle for zone in zones.keys() for particle in settled_particles_zone[zone]]
        return settled_particles

    @staticmethod
    def get_particles_from_zone(particles: Sequence['Particle'], zone: Zone) -> List['Particle']:
        """Get the subset of particles belonging to a certain genetic lineage."""
        return [particle for particle in particles if particle.genetic_lineage is zone]

    @staticmethod
    def get_positions(
        particles: Sequence['Particle'],
        time: int,
        zone: Optional[Zone] = None
    ) -> Tuple[List[float], List[float]]:
        """Get the positions of a list of particles at time, optionally originating from a zone."""
        if zone is None:
            if time == -1:
                lons, lats = zip(*[particle.settlement_position for particle in particles])
            else:
                lons, lats = zip(*[particle.get_position(time) for particle in particles])
        else:
            if time == -1:
                lons, lats = zip(
                    *[particle.settlement_position for particle in particles if particle.genetic_lineage is zone])
            else:
                lons, lats = zip(
                    *[particle.get_position(time) for particle in particles if particle.genetic_lineage is zone])
        return lons, lats