"""Recreate entities collapsed by the bad fuzzy merges.

Restores "HOKI Rewards" (Product) and "VelaTavern" (Project) when they were
absorbed into a longer name or stored only as an alias. Reattaches episodes
that mention the short name on its own. Does not delete episodes.

If a merge log exists for either name, that merge is undone first.

    python scripts/split_wrong_merges.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.maintenance import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "repair-merges", *sys.argv[1:]]
    main()
