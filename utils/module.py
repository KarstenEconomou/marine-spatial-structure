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
