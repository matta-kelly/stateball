"""Run Expectancy lookup table.

Source: 2010-2015 MLB averages (Tango/Lichtman).
Index: RE_TABLE[runners_bitmask][outs]
  - runners_bitmask: 1B=1, 2B=2, 3B=4 (0-7)
  - outs: 0, 1, 2
"""

from __future__ import annotations

import numpy as np

RE_TABLE = np.array([
    # 0 out    1 out    2 out
    [0.481,   0.254,   0.098],  # 0: bases empty
    [0.859,   0.509,   0.224],  # 1: 1B
    [1.100,   0.664,   0.319],  # 2: 2B
    [1.437,   0.884,   0.429],  # 3: 1B+2B
    [1.350,   0.950,   0.353],  # 4: 3B
    [1.784,   1.130,   0.478],  # 5: 1B+3B
    [1.920,   1.352,   0.570],  # 6: 2B+3B
    [2.282,   1.520,   0.669],  # 7: loaded
], dtype=np.float64)
