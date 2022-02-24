"""Defines the Module class."""
import sys
from itertools import permutations
from math import log10
from typing import List, Union, Sequence, Tuple, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.core.display import display

sys.path.insert(1, str(Path.cwd() / 'utils'))
from hexbin import Hexbin  # noqa: E402


class Module:
    """Represents a detected community of hexbins."""

    def __init__(self, module_index: int) -> None:
        """Initialize index of module."""
        self.index: int = module_index
        self.hexbins: Optional[List['Hexbin']] = None

        # Quality parameters
        self.coherence: Optional[float] = None
        self.fortress: Optional[float] = None
        self.mixing: Optional[float] = None

        # Color information
        self.color: Optional[str] = None

    def __eq__(self, other: 'Module') -> bool:
        """Compare module indices on equal to call."""
        return self.index == other.index

    def __lt__(self, other: 'Module') -> bool:
        """Compare module indices on less than call."""
        return self.index < other.index

    def is_small(self) -> bool:
        """Return whether the module is small."""
        small_threshold = 3
        return len(self.hexbins) <= small_threshold

    def associate_hexbins(self, hexbins: List['Hexbin']) -> None:
        """Associate the constituent hexbins with the module."""
        self.hexbins = [hexbin for hexbin in hexbins if hexbin.module.index == self.index]

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

    def is_null(self) -> bool:
        """Returns whether a module is the null module."""
        return self.index == 0

    @classmethod
    def make_null_module(cls) -> 'Module':
        """Make a null module."""
        null_module = cls(0)
        null_module.color = '#c7c7c7'
        return null_module

    @staticmethod
    def read_clu(
        file: Union[str, Path],
        label_map: Dict[str, int],
        display_clu: bool = False,
    ) -> Tuple[Tuple['Module', ...], Tuple['Hexbin', ...]]:
        """Parses the clu output file of Infomap."""
        clu = pd.read_csv(file, sep=' ', comment='#', names=['node', 'module', 'flow'])

        module_map = dict((module, Module(module)) for module in clu['module'].unique())
        hexbins = [Hexbin.from_integer(node, label_map, module_map[module], flow)
                        for node, module, flow  in zip(clu['node'], clu['module'], clu['flow'])]

        if display_clu:
            display(clu)

        return list(module_map.values()), hexbins

    @staticmethod
    def remove_noise(
        modules: List['Module'],
        null_module: 'Module',
        hexbin_dict: Dict[str, 'Hexbin'],
        colors: List[str],
    ) -> List['Module']:
        """Cycles through modules to set colors, grey bad hexbins, and re-index."""

        def grey_hexbin(hexagon: 'Hexbin') -> None:
            """Removes a hexbin from a module and sets its module to the null module."""
            hexagon.module = null_module  # Set module to null module
            null_module.hexbins.append(hexagon)  # Add to null module hexbins

        null_module.hexbins = []
        new_modules = [null_module]
        module_index = 1
        for module in modules:
            for hexbin in module.hexbins:
                adjacent_bins = hexbin.get_adjacent_bins(module.hexbins, hexbin_dict=hexbin_dict)
                number_of_adjacent_bins = len(adjacent_bins)
                if number_of_adjacent_bins == 0:
                    # Isolated hexbin
                    grey_hexbin(hexbin)
                elif number_of_adjacent_bins == 1:
                    other_bin = adjacent_bins[0]
                    if len(other_bin.get_adjacent_bins(module.hexbins, hexbin_dict=hexbin_dict)) == 1:
                        # Isolated two hexbins
                        grey_hexbin(hexbin)
                        grey_hexbin(other_bin)
            
            # Grey all hexbins if module is small
            if module.is_small():
                for hexbin in module.hexbins:
                    grey_hexbin(hexbin)

            # Color and re-index module if it still contains any hexbins
            module.associate_hexbins(hexbin_dict.values())
            if len(module.hexbins) != 0:
                module.index = module_index
                module.color = colors[module_index - 1]
                new_modules.append(module)

                module_index += 1
        
        return new_modules