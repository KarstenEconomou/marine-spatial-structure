"""Plot particles and hexagons within the domain of interest."""
import sys
from pathlib import Path
from typing import Union, Dict, Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

sys.path.insert(1, str(Path.cwd() / 'utils'))
from classifications import Zone, GeneticZone, Particle  # noqa: E402

HEX_FACE_ALPHA = 0.8
HEX_EDGE_ALPHA = 0.9
HEX_LINE_WIDTH = 0.05


def create_figure() -> plt.Figure:
    """Create figure."""
    return plt.figure(dpi=1200, facecolor='white')


def create_axis(fig: plt.Figure, title: Optional[str]) -> plt.Axes:
    """Create axis containing a land mask over the domain of interest."""
    ax = fig.add_subplot(projection=ccrs.Miller(), title=title)
    ax.set_extent([-77.3, -46.83, 34.19, 52.2])

    land = cfeature.NaturalEarthFeature(
        'physical',
        'land',
        '10m',
        edgecolor='black',
        linewidth=0.1,
        zorder=0,
        facecolor=cfeature.COLORS['land'],
    )
    ax.add_feature(land)

    return ax


def add_other_hexagons(ax: plt.Axes, domain_hexagons: ArrayLike) -> None:
    """Plot hexagons with transparent faces."""
    for hexagon in domain_hexagons:
        ax.add_patch(mpl.patches.Polygon(
            hexagon.hex,
            fc=(0, 0, 0, 0),
            ec=(0, 0, 0, HEX_EDGE_ALPHA),
            lw=HEX_LINE_WIDTH,
            zorder=1,
            transform=ccrs.PlateCarree(),
            )
        )


def add_boundaries(ax: plt.Axes, zones: Dict[Zone, GeneticZone]) -> None:
    """Draw boundaries."""
    for zone in zones.values():
        ax.plot(
            *zone.poly.exterior.xy,
            color='black',
            linewidth='0.25',
            transform=ccrs.PlateCarree(),
        )


def plot_hexbins(
    hexagons: ArrayLike,
    title: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
) -> None:
    fig = create_figure()
    ax = create_axis(fig, title)

    add_other_hexagons(ax, hexagons)

    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)


def plot_modules(
    hexagons: ArrayLike,
    colors: ArrayLike,
    other_hexagons: Optional[ArrayLike] = None,
    zones: Optional[Dict[Zone, GeneticZone]] = None,
    colorbar: bool = False,
    colorbar_label: str = 'Module',
    title: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
 ) -> None:
    """Plot hexagons coloured by the module they belong to."""
    fig = create_figure()
    ax = create_axis(fig, title)

    for hexagon in hexagons:
        ax.add_patch(mpl.patches.Polygon(
            hexagon.hex,
            fc=mpl.colors.to_rgba(colors[hexagon.module.index - 1], HEX_FACE_ALPHA),
            ec=(0, 0, 0, HEX_EDGE_ALPHA),
            lw=HEX_LINE_WIDTH,
            zorder=2,
            transform=ccrs.PlateCarree(),
            )
        )

    if other_hexagons is not None:
        add_other_hexagons(ax, other_hexagons)

    if zones is not None:
        add_boundaries(ax, zones)

    if colorbar:
        number_of_modules = max(hexagon.module.index for hexagon in hexagons)

        ticks = np.arange(1, (number_of_modules + 2))
        tick_locs = (np.arange(len(ticks)) + 0.5) * len(ticks) / len(ticks)

        cmap = mpl.colors.ListedColormap(colors[0:number_of_modules])
        norm = mpl.colors.BoundaryNorm(ticks, number_of_modules + 1)

        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
        cb.set_ticks(tick_locs[1::2])
        cb.set_ticklabels(ticks[0:number_of_modules:2].astype(int))
        cb.set_label(colorbar_label, labelpad=5)

    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)


def plot_quality(
    hexagons: ArrayLike,
    parameter: str = 'coherence',
    other_hexagons: Optional[ArrayLike] = None,
    cmap: str = 'magma',
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    title: Optional[str] = None,
    path: Optional[Union[str, Path]] = None
) -> None:
    """Plot hexagons coloured by the quality of their module."""
    fig = create_figure()
    ax = create_axis(fig, title)

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    color = mpl.cm.get_cmap(cmap)

    for hexagon in hexagons:
        if parameter == 'coherence':
            face_color = color(norm(hexagon.module.coherence))
        elif parameter == 'fortress':
            face_color = color(norm(hexagon.module.fortress))
        elif parameter == 'mixing':
            face_color = color(norm(hexagon.module.mixing))
        else:
            raise ValueError('Invalid quality parameter.')

        ax.add_patch(mpl.patches.Polygon(
            hexagon.hex,
            fc=mpl.colors.to_rgba(face_color, HEX_FACE_ALPHA),
            ec=(0, 0, 0, HEX_EDGE_ALPHA),
            lw=HEX_LINE_WIDTH,
            zorder=2,
            transform=ccrs.PlateCarree(),
            )
        )

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, label=colorbar_label, pad=0.02, fraction=0.046)

    if other_hexagons is not None:
        add_other_hexagons(ax, other_hexagons)

    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)


def plot_boundary_persistence(
    hexagons: ArrayLike,
    other_hexagons: Optional[ArrayLike] = None,
    cmap: str = 'cool',
    colorbar: bool = True,
    title: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot hexagons coloured by the percentage of solutions in which they act as boundaries."""
    fig = create_figure()
    ax = create_axis(fig, title)

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    color = mpl.cm.get_cmap(cmap)

    for hexagon in hexagons:
        face_color = color(norm(hexagon.boundary_persistence))
        ax.add_patch(mpl.patches.Polygon(
            hexagon.hex,
            fc=mpl.colors.to_rgba(face_color, HEX_FACE_ALPHA),
            ec=(0, 0, 0, HEX_EDGE_ALPHA),
            lw=HEX_LINE_WIDTH,
            zorder=2,
            transform=ccrs.PlateCarree(),
            )
        )

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, label='Boundary persistence', pad=0.02, fraction=0.046)

    if other_hexagons is not None:
        add_other_hexagons(ax, other_hexagons)

    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)


def plot_positions(
    lon: ArrayLike,
    lat: ArrayLike,
    color: str = 'lightblue',
    title: str = None,
    path: Union[str, Path] = None,
) -> None:
    """Plot particle positions."""
    fig = create_figure()
    ax = create_axis(fig, title)

    ax.scatter(
        lon,
        lat,
        alpha=0.7,
        s=0.25,
        marker='o',
        linewidth=0.0,
        color=color,
        transform=ccrs.PlateCarree(),
    )

    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)


def plot_particles(
    particles: ArrayLike,
    time: int = 0,
    color: str = 'lightblue',
    title: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot particle positions."""
    fig = create_figure()
    ax = create_axis(fig, title)

    if time == -1:
        lons, lats = zip(*[particle.final_position for particle in particles])
    else:
        lons, lats = Particle.get_positions(particles, time=time)

    ax.scatter(
        lons,
        lats,
        alpha=0.7,
        s=0.2,
        marker='o',
        linewidth=0.0,
        color=color,
        transform=ccrs.PlateCarree(),
    )

    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)


def plot_subpopulations(
    particles: ArrayLike,
    zones: Dict[Zone, GeneticZone],
    time: int = 0,
    colors: Optional[ArrayLike] = None,
    title: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot particle positions coloured by genetic lineage."""
    fig = create_figure()
    ax = create_axis(fig, title)

    if colors is None:
        colors = ['red', 'green', 'blue']

    add_boundaries(ax, zones)

    for i, zone in enumerate(zones.keys()):
        if time == -1:
            lons, lats = zip(*[particle.final_position for particle in particles if particle.genetic_lineage is zone])
        else:
            lons, lats = Particle.get_positions(particles, time=time, zone=zone)

        ax.scatter(
            lons,
            lats,
            alpha=0.7,
            s=0.2,
            marker='o',
            linewidth=0.0,
            color=colors[i],
            transform=ccrs.PlateCarree(),
        )

    if path is not None:
        plt.savefig(path)
