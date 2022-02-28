"""Plot particles and hexagons within the domain of interest."""
import sys
from pathlib import Path
from typing import Union, Dict, Optional, Sequence, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from numpy.typing import ArrayLike

sys.path.insert(1, str(Path.cwd() / 'utils'))
from constants import LEFT_BOUND, RIGHT_BOUND, TOP_BOUND, BOTTOM_BOUND
from geneticlineage import GeneticLineage  # noqa: E402
from hexbin import Hexbin  # noqa: E402
from module import Module  # noqa: E402
from particle import Particle  # noqa: E402
from zone import Zone  # noqa: E402

PROJECTION = ccrs.PlateCarree()

OBJ_ZORDER = 0

HEX_FACE_ALPHA = 1
HEX_EDGE_ALPHA = 1
HEX_LINE_WIDTH = 0.04

TICK_PAD = 2
LABEL_PAD = 4

TICK_FONT_SIZE = 7
LABEL_FONT_SIZE = TICK_FONT_SIZE


def create_figure() -> plt.Figure:
    """Create figure."""
    return plt.figure(dpi=900, facecolor='white')


def create_axis(fig: plt.Figure, title: Optional[str] = None, ticks: bool = True) -> plt.Axes:
    """Create axis containing a land mask over the domain of interest."""
    ax = fig.add_subplot(projection=PROJECTION, title=title)
    pad = 0.25
    ax.set_extent([LEFT_BOUND - pad, RIGHT_BOUND + pad, BOTTOM_BOUND - pad, TOP_BOUND + pad])

    if ticks:
        ax.set_xticks([-75, -70, -65, -60, -55, -50], crs=PROJECTION)
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax.xaxis.set_tick_params(labelsize=TICK_FONT_SIZE, pad=TICK_PAD)

        ax.set_yticks([35, 40, 45, 50], crs=PROJECTION)
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        ax.yaxis.set_tick_params(labelsize=TICK_FONT_SIZE, pad=TICK_PAD)

    land = cfeature.NaturalEarthFeature(
        'physical',
        'land',
        '10m',
        edgecolor='black',
        linewidth=0.1,
        zorder=OBJ_ZORDER + 2,
        facecolor=cfeature.COLORS['land'],
    )
    ax.add_feature(land)

    return ax


def add_other_hexagons(ax: plt.Axes, domain_hexagons: Sequence[Hexbin]) -> None:
    """Plot hexagons with transparent faces."""
    for hexagon in domain_hexagons:
        ax.add_patch(mpl.patches.Polygon(
            hexagon.hex,
            fc=(0, 0, 0, 0),
            ec=(0, 0, 0, HEX_EDGE_ALPHA),
            lw=HEX_LINE_WIDTH,
            zorder=OBJ_ZORDER,
            transform=ccrs.PlateCarree(),
            )
        )


def add_boundaries(ax: plt.Axes, zones: Dict[Zone, GeneticLineage]) -> None:
    """Draw boundaries."""
    for zone in zones.values():
        ax.plot(
            *zone.poly.exterior.xy,
            color='black',
            linewidth='0.25',
            zorder=OBJ_ZORDER + 1,
            transform=ccrs.PlateCarree(),
        )


def save_plot(path: Union[str, Path]):
    """Save a plot."""
    pad_inches = 0.05
    plt.savefig(path, bbox_inches='tight', pad_inches=pad_inches)


def plot_hexbins(
    hexagons: Sequence[Hexbin],
    title: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
) -> None:
    fig = create_figure()
    ax = create_axis(fig, title)

    add_other_hexagons(ax, hexagons)

    if path is not None:
        save_plot(path)


def plot_modules(
    modules: Sequence[Module],
    other_hexagons: Optional[Sequence[Hexbin]] = None,
    zones: Optional[Dict[Zone, GeneticLineage]] = None,
    colorbar: bool = False,
    colorbar_label: Optional[str] = None,
    title: Optional[str] = None,
    ticks: bool = True,
    path: Optional[Union[str, Path]] = None,
 ) -> None:
    """Plot hexagons coloured by the module they belong to."""
    fig = create_figure()
    ax = create_axis(fig, title, ticks)

    for module in modules:
        for hexagon in module.hexbins:
            ax.add_patch(mpl.patches.Polygon(
                hexagon.hex,
                fc=mpl.colors.to_rgba(module.color, HEX_FACE_ALPHA),
                ec=(0, 0, 0, HEX_EDGE_ALPHA),
                lw=HEX_LINE_WIDTH,
                zorder=OBJ_ZORDER,
                transform=ccrs.PlateCarree(),
                )
            )

    if other_hexagons is not None:
        add_other_hexagons(ax, other_hexagons)

    if zones is not None:
        add_boundaries(ax, zones)

    if colorbar:
        number_of_modules = len(modules)
        module_colors = [module.color for module in modules]

        ticks = np.arange(1, (number_of_modules + 2))
        tick_locs = (np.arange(len(ticks)) + 0.5) * len(ticks) / len(ticks)

        cmap = mpl.colors.ListedColormap(module_colors)
        norm = mpl.colors.BoundaryNorm(ticks, number_of_modules + 1)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)

        cb.set_ticks(tick_locs[1::2])
        cb.set_ticklabels(ticks[0:number_of_modules:2].astype(int))
        cb.ax.tick_params(labelsize=TICK_FONT_SIZE, pad=TICK_PAD)
        
        cb.set_label(colorbar_label, size=LABEL_FONT_SIZE, labelpad=LABEL_PAD)

    if path is not None:
        save_plot(path)


def plot_quality(
    modules: Sequence[Module],
    parameter: str = 'coherence',
    other_hexagons: Optional[Sequence[Hexbin]] = None,
    cmap: str = 'cet_rainbow',
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

    for module in modules:
        # Module should be colored according to the desire parameter
        if parameter == 'coherence':
            parameter_value = module.coherence
        elif parameter == 'fortress':
            parameter_value = module.fortress
        elif parameter == 'mixing':
            parameter_value = module.mixing

        face_color = color(norm(parameter_value))

        for hexagon in module.hexbins:
            ax.add_patch(mpl.patches.Polygon(
                hexagon.hex,
                fc=mpl.colors.to_rgba(face_color, HEX_FACE_ALPHA),
                ec=(0, 0, 0, HEX_EDGE_ALPHA),
                lw=HEX_LINE_WIDTH,
                zorder=OBJ_ZORDER,
                transform=ccrs.PlateCarree(),
                )
            )

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(colorbar_label, size=LABEL_FONT_SIZE, labelpad=LABEL_PAD)
        cb.ax.tick_params(labelsize=TICK_FONT_SIZE, pad=TICK_PAD)

    if other_hexagons is not None:
        add_other_hexagons(ax, other_hexagons)

    if path is not None:
        save_plot(path)


def plot_flow(
    hexbins: Sequence[Hexbin],
    other_hexagons: Optional[Sequence[Hexbin]] = None,
    cmap: str = 'cet_rainbow',
    colorbar: bool = True,
    title: Optional[str] = None,
    path: Optional[Union[str, Path]] = None
) -> None:
    """Plot hexagons coloured by their flow."""
    fig = create_figure()
    ax = create_axis(fig, title)

    flows = tuple(hexbin.flow for hexbin in hexbins)

    norm = mpl.colors.Normalize(vmin=min(flows), vmax=max(flows))
    color = mpl.cm.get_cmap(cmap)

    for hexagon in hexbins:
        ax.add_patch(mpl.patches.Polygon(
            hexagon.hex,
            fc=mpl.colors.to_rgba(color(norm(hexagon.flow)), HEX_FACE_ALPHA),
            ec=(0, 0, 0, HEX_EDGE_ALPHA),
            lw=HEX_LINE_WIDTH,
            zorder=OBJ_ZORDER,
            transform=ccrs.PlateCarree(),
            )
        )

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label('Flow', size=LABEL_FONT_SIZE, labelpad=LABEL_PAD)
        cb.ax.tick_params(labelsize=TICK_FONT_SIZE, pad=TICK_PAD)

    if other_hexagons is not None:
        add_other_hexagons(ax, other_hexagons)

    if path is not None:
        save_plot(path)


def plot_positions(
    lon: Sequence[float],
    lat: Sequence[float],
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
        zorder=OBJ_ZORDER,
        transform=ccrs.PlateCarree(),
    )

    if path is not None:
        save_plot(path)


def plot_particles(
    particles: Sequence[Particle],
    time: int,
    color: str = 'lightblue',
    title: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot particle positions."""
    fig = create_figure()
    ax = create_axis(fig, title)

    if time == -1:
        lons, lats = zip(*[particle.settlement_position for particle in particles])
    else:
        lons, lats = Particle.get_positions(particles, time)

    ax.scatter(
        lons,
        lats,
        alpha=0.7,
        s=0.2,
        marker='o',
        linewidth=0.0,
        color=color,
        zorder=OBJ_ZORDER,
        transform=ccrs.PlateCarree(),
    )

    if path is not None:
        save_plot(path)


def plot_subpopulations(
    particles: Sequence[Particle],
    time: int,
    zones: Dict[Zone, GeneticLineage],
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

    for color, zone in zip(colors, zones.keys()):
        if time == -1:
            lons, lats = zip(*[particle.settlement_position for particle in particles if particle.genetic_lineage is zone])
        else:
            lons, lats = Particle.get_positions(particles, time, zone=zone)

        ax.scatter(
            lons,
            lats,
            alpha=0.7,
            s=0.2,
            marker='o',
            linewidth=0.0,
            color=color,
            zorder=OBJ_ZORDER,
            transform=ccrs.PlateCarree(),
        )

    if path is not None:
        save_plot(path)


def plot_contourf(
    positions: Sequence[Tuple[float, float]],
    values: Sequence[float],
    title: Optional[str] = None,
    cmap: str = 'viridis',
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
) -> None:
    fig = create_figure()
    ax = create_axis(fig, title)

    min_value = min(values)
    max_value = max(values)
    norm = plt.Normalize(vmin=min_value, vmax=max_value)

    lats, lons = list(map(list, zip(*positions)))

    ax.tricontourf(
        lats,
        lons,
        list(values), 
        alpha=1,
        cmap=cmap,
        norm=norm,
        zorder=OBJ_ZORDER,
        transform=ccrs.PlateCarree(),
    )

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, pad=0.02, fraction=0.046)
        cb.set_label(colorbar_label, size=LABEL_FONT_SIZE, labelpad=LABEL_PAD)
        cb.ax.tick_params(labelsize=TICK_FONT_SIZE, pad=TICK_PAD)

    if path is not None:
        save_plot(path)