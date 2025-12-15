from collections import defaultdict
from typing import List, Dict, Any
import numpy as np


class ContextManager:
    """
    Groups time series by labelset.
    """

    @staticmethod
    def group_by_labelset(data: List[Dict[str, Any]]):
        """
        Input:
            [
              {"labelset": {...}, "values": np.ndarray},
              ...
            ]

        Output:
            {
              frozenset(labelset.items()): [np.ndarray, ...]
            }
        """
        grouped = defaultdict(list)

        for item in data:
            labelset = frozenset(item["labelset"].items())
            grouped[labelset].append(item["values"])

        return grouped
