"""This module simulates the transport of particles that swim towards the mixed layer depth.

Written by Kira Krumhansl, Wendy Gentleman, and Karsten Economou.
"""
import sys
from datetime import timedelta as delta
from math import fabs, sqrt, copysign
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from parcels import FieldSet, Field, ParticleSet, JITParticle, ParticleFile, ErrorCode, Variable, ParcelsRandom
from scipy.io import loadmat

sys.path.insert(1, str(Path.cwd() / 'utils'))
from season import Season  # noqa: E402

SPAWN = Season.fall.name

SIMULATION_TIME = 45


def define_fields():
    """Read hydrodynamic data and generate physical field for simulation."""
    # Correspond data files and variables
    files = {
        'U': Path.cwd() / 'data' / 'fields' / SPAWN / '3D_U_SmallGrid_BNAM.nc',
        'V': Path.cwd() / 'data' / 'fields' / SPAWN / '3D_V_SmallGrid_BNAM.nc',
        'W': Path.cwd() / 'data' / 'fields' / SPAWN / '3D_W_SmallGrid_BNAM.nc',
        'mesh_mask': Path.cwd() / 'data' / 'fields' / '3D_Mask_SmallGrid_BNAM.nc',
        'MLD': Path.cwd() / 'data' / 'fields' / SPAWN / 'MLD_SmallGrid_BNAM.nc',
        'Bathy': Path.cwd() / 'data' / 'fields' / 'Bathy_SmallGrid_BNAM',
        'Temp': Path.cwd() / 'data' / 'fields' / SPAWN / 'Temp_SmallGrid_BNAM.nc',
        'Sal': Path.cwd() / 'data' / 'fields' / SPAWN / 'Sal_SmallGrid_BNAM.nc',
        'Mixing': Path.cwd() / 'data' / 'fields' / '3D_Mixing_SmallGrid_BNAM.nc',
    }

    # Map hydrodynamic data
    filenames = {
        'U': {'lon': files['mesh_mask'], 'lat': files['mesh_mask'], 'depth': files['W'], 'data': files['U']},
        'V': {'lon': files['mesh_mask'], 'lat': files['mesh_mask'], 'depth': files['W'], 'data': files['V']},
        'W': {'lon': files['mesh_mask'], 'lat': files['mesh_mask'], 'depth': files['W'], 'data': files['W']},
        'MLD': {'lon': files['mesh_mask'], 'lat': files['mesh_mask'], 'data': files['MLD']},
        'Temp': {'lon': files['mesh_mask'], 'lat': files['mesh_mask'], 'depth': files['W'], 'data': files['Temp']},
        'Sal': {'lon': files['mesh_mask'], 'lat': files['mesh_mask'], 'depth': files['W'], 'data': files['Sal']},
    }

    variables = {
        'U': 'U',
        'V': 'V',
        'W': 'W',
        'MLD': 'MLD',
        'Temp': 'Temp',
        'Sal': 'Sal',
    }

    dimensions = {
        'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
        'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
        'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
        'MLD': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
        'Temp': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
        'Sal': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
    }

    # Create field
    f_set = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=True)

    # Include mixing data
    mixing_data = Dataset(files['Mixing'])
    kh_zonal = mixing_data.variables['kh_zonal'][:]
    kh_meridional = mixing_data.variables['kh_meridional'][:]
    f_set.add_field(Field('Kh_zonal', data=kh_zonal, lon=f_set.W.lon, lat=f_set.W.lat, depth=f_set.W.depth,
                          mesh='spherical', allow_time_extrapolation=True))
    f_set.add_field(Field('Kh_meridional', data=kh_meridional, lon=f_set.W.lon, lat=f_set.W.lat, depth=f_set.W.depth,
                          mesh='spherical', allow_time_extrapolation=True))

    # Include bathymetry data
    bathy = loadmat(str(files['Bathy']))['Bathymetry']
    f_set.add_field(Field('Bathy', data=bathy, lon=f_set.W.lon, lat=f_set.W.lat, interp_method='cgrid_tracer',
                          allow_time_extrapolation=True))

    # Change dres to avoid potential for catastrophic cancellation
    f_set.add_constant('dres', 0.001)

    return f_set


def initialize_space_time_parameters():
    """Initialize particle positional, temporal, and tracking parameters."""
    # Set initial particle positions (latitudes and longitudes)
    initial_conditions_data = np.loadtxt(Path.cwd() / 'data' / 'initial_positions.txt')
    lon = initial_conditions_data[:, 0]
    lat = initial_conditions_data[:, 1]

    # Initialize depth and time arrays
    depth = np.ones(len(lon))  # Particles are initialized at depth 1 -- immediately replaced with calculation
    time = np.zeros(len(lon))

    return lon, lat, depth, time


def define_particles(f_set):
    """Define particles for simulation."""
    # Define particle movement constants in mm/s
    sink_speed_mean = 0.6
    sink_speed_sd = 0.3
    swim_speed_mean = 2.4
    swim_speed_sd = 2

    class SampleParticle(JITParticle):
        """Particle with behavioral attributes that can sample local field data."""
        MLD = Variable('MLD', initial=f_set.MLD)
        Bathy = Variable('Bathy', initial=f_set.Bathy)
        Temp = Variable('Temp', initial=f_set.Temp)
        Sal = Variable('Sal', initial=f_set.Sal)

        sink_pref = ParcelsRandom.normalvariate(sink_speed_mean, sink_speed_sd)
        if sink_pref > sink_speed_mean + sink_speed_sd:
            sink_pref = sink_speed_mean + sink_speed_sd * ParcelsRandom.uniform(0, 1)
        elif sink_pref < sink_speed_mean - sink_speed_sd:
            sink_pref = sink_speed_mean - sink_speed_sd * ParcelsRandom.uniform(0, 1)
        W_sink = Variable('W_sink', initial=sink_pref, to_write=False)

        swim_pref = ParcelsRandom.normalvariate(swim_speed_mean, swim_speed_sd)
        if swim_pref > swim_speed_mean + swim_speed_sd:
            swim_pref = swim_speed_mean + swim_speed_sd * ParcelsRandom.uniform(0, 1)
        elif swim_pref < swim_speed_mean - swim_speed_sd:
            swim_pref = swim_speed_mean - swim_speed_sd * ParcelsRandom.uniform(0, 1)
        W_swim = Variable('W_swim', initial=swim_pref, to_write=False)

    # Initialize particle parameters
    particle_lon, particle_lat, particle_depth, particle_time = initialize_space_time_parameters()

    # Return final simulation particles
    return ParticleSet.from_list(fieldset=f_set, pclass=SampleParticle, time=particle_time, lon=particle_lon,
                                 lat=particle_lat, depth=particle_depth, lonlatdepth_dtype=np.float64)


def characterize_advection_diffusion(particle, fieldset, time):
    """Characterize 3D advection and horizontal diffusion.

    Advection solved using fourth order-Runge-Kutta (RK4),
    3D horizontal diffusion the Milstein scheme at first order (M1).
    """

    # Current space-time location of particle
    lon_p = particle.lon
    lat_p = particle.lat
    depth_p = particle.depth
    dt = particle.dt

    # RK4 for Advection
    # Evaluate coefficients for linear interpolation of fields. Assumes starting mid-Sept and running for <=60 days

    # Start at current space-time point using those values to get slope
    (u1, v1) = fieldset.UV[time, depth_p, lat_p, lon_p]

    # Get estimate for slope at midpoint using previous estimate for slope
    lon1, lat1 = (lon_p + u1 * 0.5 * dt, lat_p + v1 * .5 * dt)
    (u2, v2) = fieldset.UV[time + 0.5 * dt, depth_p, lat1, lon1]

    # Get improved estimate for slope at midpoint using previous estimate for slope
    lon2, lat2 = (lon_p + u2 * 0.5 * dt, lat_p + v2 * .5 * dt)
    (u3, v3) = fieldset.UV[time + 0.5 * dt, depth_p, lat2, lon2]

    # Get estimate for slope at endpoint using previous estimate for slope at midpoint
    lon3, lat3 = (lon_p + u3 * dt, lat_p + v3 * dt)
    (u4, v4) = fieldset.UV[time + dt, depth_p, lat3, lon3]

    # Calculate particle displacement due to local advection
    # Assumes that FieldSet has already converted u,v,w to degrees
    advect_lon = ((u1 + 2 * u2 + 2 * u3 + u4) / 6.) * dt
    advect_lat = ((v1 + 2 * v2 + 2 * v3 + v4) / 6.) * dt

    # Milstein for horizontal diffusion
    kh_diff_dist = fieldset.dres  # in degrees
    # Note Kh is in m^2/s here, unlike built-in that has already converted to degrees
    # Conversion is done after computing displace ent to save on repeated conversions

    # Sample random number with zero mean and std of 1 from normal distribution
    Rx = ParcelsRandom.normalvariate(0., 1.)
    Ry = ParcelsRandom.normalvariate(0., 1.)

    # Get estimate of random kick in x-direction based on local diffusivity (neglects spatial variation)
    delx1 = sqrt(2 * fieldset.Kh_zonal[time, depth_p, lat_p, lon_p] * dt) * Rx

    # Get estimate of random kick in y-direction based on local diffusivity (neglects spatial variation)
    dely1 = sqrt(2 * fieldset.Kh_meridional[time, depth_p, lat_p, lon_p] * dt) * Ry

    # Get estimate of zonal diffusivity gradient at current location using finite centered difference
    # This derivative is used to correct basic random kick due to variable diffusivity
    Kxp1 = fieldset.Kh_zonal[time, depth_p, lat_p, lon_p + kh_diff_dist]
    Kxm1 = fieldset.Kh_zonal[time, depth_p, lat_p, lon_p - kh_diff_dist]
    dKdx = (Kxp1 - Kxm1) / (2 * kh_diff_dist)
    delx2 = 0.5 * dKdx * (Rx ** 2 + 1) * dt

    # Get estimate of meridional gradient at current location using finite centered difference
    # This derivative is used to correct basic random kick due to variable diffusivity
    Kyp1 = fieldset.Kh_meridional[time, depth_p, lat_p + kh_diff_dist, lon_p]
    Kym1 = fieldset.Kh_meridional[time, depth_p, lat_p - kh_diff_dist, lon_p]
    dKdy = (Kyp1 - Kym1) / (2 * kh_diff_dist)
    dely2 = 0.5 * dKdy * (Ry ** 2 + 1) * dt

    # Calculate particle horizontal displacement due to local diffusion and diffusivity gradient
    diffh_lon = delx1 + delx2
    diffh_lat = dely1 + dely2

    # Particle positions are updated only after evaluating all terms (i.e. advection + diffusion simultaneously)
    # Does not consider the interaction of adv and diff on estimated particle displacements within the time step)
    particle.lon += advect_lon + diffh_lon
    particle.lat += advect_lat + diffh_lat


def characterize_sinking(particle, fieldset, time):
    """Characterize the sinking of a particle."""
    particle.depth += particle.W_sink * 0.001 * particle.dt


def characterize_swimming(particle, fieldset, time):
    """Characterize the biological behavior of a particle."""
    min_depth = 0.51
    max_mean_depth = 20
    max_depth_range = 10
    squish_factor = 0.75

    # Mean particle depth is the minimum magnitude of three parameters
    depth_mean = min(min(particle.MLD, particle.Bathy / 2), max_mean_depth)

    # Range of acceptable depths about the mean is the minimum magnitude of three parameters
    depth_range = min(
        min(squish_factor * (depth_mean - min_depth), squish_factor * (particle.Bathy - depth_mean)), max_depth_range)

    # Preferred particle depth is a normal distribution approximately limited by the depth range around the mean depth
    depth_pref = depth_mean + ParcelsRandom.normalvariate(0, depth_range / 2)

    if depth_pref > depth_mean + depth_range:
        depth_pref = depth_mean + depth_range * ParcelsRandom.uniform(0, 1)
    elif depth_pref < depth_mean - depth_range:
        depth_pref = depth_mean - depth_range * ParcelsRandom.uniform(0, 1)

    # Assign swimming up or down towards preferred depth
    displacement_to_depth_pref = depth_pref - particle.depth  # Displacement of particle away from preferred depth
    swim_distance = min(particle.W_swim * 0.001 * particle.dt, fabs(displacement_to_depth_pref))
    depth_swam = copysign(swim_distance, displacement_to_depth_pref)  # Signed depth swam by particle

    # Update depth by amount swam
    particle.depth += depth_swam


def sample_particle_environment(particle, fieldset, time):
    """Sample the MLD, bathymetry, temperature, and salinity at a particle's location."""
    particle.MLD = fieldset.MLD[time, particle.depth, particle.lat, particle.lon]
    particle.Bathy = fieldset.Bathy[time, particle.depth, particle.lat, particle.lon]
    particle.Temp = fieldset.Temp[time, particle.depth, particle.lat, particle.lon]
    particle.Sal = fieldset.Sal[time, particle.depth, particle.lat, particle.lon]


def set_initial_particle_depth(particle, fieldset, time):
    """Determine initial particle depth using MLD, bathymetry, and defined minimum depth.

    MLD is assumed to be defined for every location.
    """
    min_depth = 0.51
    max_mean_depth = 20

    depth_pref = particle.MLD  # Preferred depth of particle is the MLD
    # depth_pref += random.normalvariate(0,5)

    particle.depth = min(max(depth_pref, min_depth), max(max_mean_depth, particle.Bathy - 1))


def assign_kernels(p_set):
    """Assign kernels for simulation."""
    # Define kernels for initialization
    initialization_kernels = (p_set.Kernel(set_initial_particle_depth) + p_set.Kernel(sample_particle_environment))
    p_set.execute(initialization_kernels, dt=0)  # Initialize particles with initial kernels

    # Define kernels to be used during transport
    return (p_set.Kernel(characterize_advection_diffusion)
            + p_set.Kernel(characterize_sinking)
            + p_set.Kernel(characterize_swimming)
            + p_set.Kernel(sample_particle_environment)
            )


def correct_out_of_bounds(particle, fieldset, time):
    """Correct the location of a particle that has left the domain (in a region of impossible interpolation)."""
    min_depth = 0.51

    if particle.depth > particle.Bathy - 1:
        # Below boundaries
        particle.depth = max(particle.Bathy - 1, min_depth)
        particle.time = time + particle.dt
    elif particle.depth < min_depth:
        # Above boundaries
        particle.depth = min_depth
        particle.time = time + particle.dt
    else:
        # To the side of boundaries
        particle.delete()


def run_simulation(p_set, kernels):
    """Execute defined particle tracking simulation."""
    p_set.execute(
        kernels,
        runtime=delta(days=SIMULATION_TIME),
        dt=delta(minutes=30),
        recovery={ErrorCode.ErrorOutOfBounds: correct_out_of_bounds},
        output_file=ParticleFile(
            Path.cwd() / 'data' / 'simulations' / f'{SPAWN}.nc',
            p_set,
            outputdt=delta(days=1),
        ),
    )


def main():
    """Prepare and run simulation."""
    field_set = define_fields()
    particle_set = define_particles(field_set)
    tracking_kernels = assign_kernels(particle_set)

    run_simulation(particle_set, tracking_kernels)


if __name__ == "__main__":
    main()
