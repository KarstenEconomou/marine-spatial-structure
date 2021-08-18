
# Assessing spatial structure in marine systems

![Spatial structure](https://karsteneconomou.com/projects/marine-spatial-structure/clusters.png)

This repository contains code used for assessing spatial structure in marine systems modelled using particle tracking. This research was conducted with [Dalhousie University](https://www.dal.ca)'s Department of Engineering Mathematics & Internetworking in collaboration with [Fisheries and Oceans Canada](https://www.dfo-mpo.gc.ca) as a [Natural Sciences and Engineering Research Council of Canada](https://www.nserc-crsng.gc.ca) Undergraduate Student Research Award recipient. Author is solely Karsten Economou.

## About the research

Particle tracking models, which portray the transport of particles in the ocean over a time interval, are used to assess the connectivity between established regions in terms of particle transfer between them. When the particles being modelled are larvae with behaviour, adjacent regions with low connectivity between them are said to have a natural biogeographic barrier to larval transport separating them. With particle tracking models, testing the validity of hypothesized barriers is trivial by comparing the spawn and settlement positions of particles; however, particle tracking models alone cannot synthesize the existence or locations of biogeographic barriers. We develop a framework utilizing the community detection algorithm *Infomap* to highlight the spatial structure demonstrated in particle tracking models, from which biogeographic barriers among other flow characteristics can be inferred. First, we partition our domain into discrete bins that particle positions at two discrete relative times are assigned to. Then, we represent particle flow as probabilities to directionally transition between bins in a network. Finally, we use *Infomap* to detect communities in the network, which provide information about flow dynamics in the system. We test the capabilities of this framework on a particle tracking model of three genetic lineages of *Placopecten magellanicus* with and without suitable habitat considerations specified by species distribution models to show this framework is robust to biological characteristics of modelled particles.

Read the full report, [Assessing spatial structure in marine systems: Developing a framework for synthesizing natural biogeographic barriers in particle tracking models](https://karsteneconomou.com/projects/marine-spatial-structure/report.pdf).

## Data used

The particle tracking simulation, contributed to by Dr. Kira Krumhansl, Dr. Wendy Gentleman, and Karsten Economou (not included in this repository), uses the Bedford Institute of Oceanography North Atlantic Ocean model (BNAM) for field data.

The species distribution model used was provided by Dr. Ben Lowen.

## Usage

See `LICENSE`.

### Installation

All `.py` and `.ipynb` files are written in Python 3.8.

Clone the repository with

```shell
git clone https://github.com/KarstenEconomou/marine-spatial-structure
```

or use [DownGit](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/KarstenEconomou/marine-spatial-structure) to download the files.

Install requirements with a virtual environment (venv) activated with

```shell
pip install -r requirements.txt
```

### File structure

Used data, temporary files, and most output is not included in this repository. `pathlib` is used for referencing files in code with the `cwd` assumed to be the top-level project directory.

## Main pipeline

### Pre-simulation

1. Create initial particle locations of a uniform density: `initial_positions.ipynb`
2. Convert polygons defining genetic regions expressed in `.shp` (and related files) format to `.txt` files of latitudes
   and longitudes: `make_polygon.m`
3. Write a grid over the domain depicting the suitability of habitat of each cell for each genetic lineage from a `.tif` probability-based species distribution model: `sdm_grid.ipynb`

### Post-simulation

1. Process simulated particle trajectories and create a flow network: `network.ipynb`
2. Run *Infomap*: `detect_communities.ipynb`
3. Plot and analyze detected communities stored in `.clu` format: `clusters.ipynb`

### Utilities

* `plot.py` offers tailored options for plotting particles and hexagons over the domain of interest
* `classifications.py` is a class framework used throughout post-processing
* `constants.py` is a centralized hub of constants
