"""Re-extract entities for existing episodes.

Drops generic names and any entity that would be attached to more than 30%
of episodes. Prints counts and entity names only, never episode text.

    python scripts/backfill_entities.py
    python scripts/backfill_entities.py --limit 20
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.maintenance import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "backfill-entities", *sys.argv[1:]]
    main()
