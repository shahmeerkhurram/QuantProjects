"""Art Gallery Problem — Fisk's constructive proof, implemented.

The Art Gallery Theorem (Chvátal 1975) states that ``floor(n/3)`` guards are
always sufficient, and sometimes necessary, to see the whole interior of a
simple polygon with ``n`` vertices.

Fisk's 1978 proof is constructive and is what this package implements:

1. Triangulate the polygon (ear clipping, O(n^2)).
2. The triangulation's vertex graph is 3-colourable — a fact that follows from
   the dual tree being acyclic, so a greedy colouring along it never conflicts.
3. Every triangle has one vertex of each colour, so placing guards on the
   *least frequent* colour class covers every triangle, hence the whole polygon,
   using at most ``floor(n/3)`` guards.

The output is verified rather than asserted: :func:`verify_coverage` checks the
guard set against every triangle independently of the construction.

This is the geometry counterpart to the risk engine in :mod:`risk_engine` — a
separate problem, kept in a separate package, sharing only the repository.
"""

from .geometry import (
    Point,
    Polygon,
    is_convex_vertex,
    point_in_triangle,
    polygon_area,
    segment_intersects,
)
from .solver import (
    GuardSolution,
    place_guards,
    three_colour,
    triangulate,
    verify_coverage,
)

__version__ = "0.1.0"

__all__ = [
    "GuardSolution",
    "Point",
    "Polygon",
    "__version__",
    "is_convex_vertex",
    "place_guards",
    "point_in_triangle",
    "polygon_area",
    "segment_intersects",
    "three_colour",
    "triangulate",
    "verify_coverage",
]
