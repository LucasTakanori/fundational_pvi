# Make parent modules accessible as src.X
import sys
from pathlib import Path

# Add parent to path FIRST
parent = Path(__file__).parent.parent
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))

# Register this module location for submodules
__path__ = [str(parent)]


