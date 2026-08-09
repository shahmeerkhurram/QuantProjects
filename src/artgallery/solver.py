"""Fisk's constructive proof: triangulate, 3-colour, place guards on a colour class."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .geometry import Point, Polygon, is_convex_vertex, point_in_triangle

__all__ = [
    "GuardSolution",
    "Triangle",
    "place_guards",
    "three_colour",
    "triangulate",
    "verify_coverage",
]

Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class GuardSolution:
    """A verified guard placement for one polygon."""

    polygon: Polygon
    triangles: list[Triangle]
    colouring: dict[int, int]
    guards: list[int]
    guard_colour: int

    @property
    def n_guards(self) -> int:
        return len(self.guards)

    @property
    def bound(self) -> int:
        return self.polygon.theoretical_bound

    @property
    def guard_points(self) -> list[Point]:
        return [self.polygon.vertices[i] for i in self.guards]

    def summary(self) -> str:
        return (
            f"n={self.polygon.n} vertices, {len(self.polygon.reflex_vertices())} reflex  |  "
            f"{len(self.triangles)} triangles  |  {self.n_guards} guards "
            f"(bound floor(n/3) = {self.bound})"
        )


def triangulate(polygon: Polygon) -> list[Triangle]:
    """Ear-clipping triangulation, returning vertex-index triples.

    A vertex is an *ear tip* when it is convex and the triangle it forms with its
    neighbours contains no other vertex of the polygon. Clipping ears one at a
    time terminates because every simple polygon with more than three vertices
    has at least two (the Two Ears Theorem).

    O(n^2). Adequate here: the point is to demonstrate the theorem
    constructively, not to triangulate a million-vertex mesh.
    """
    n = polygon.n
    if n == 3:
        return [(0, 1, 2)]

    remaining = list(range(n))
    verts = polygon.vertices
    triangles: list[Triangle] = []
    # Guard against a pathological input looping forever rather than hanging.
    attempts_without_progress = 0

    while len(remaining) > 3:
        if attempts_without_progress > len(remaining):
            raise RuntimeError(
                "ear clipping stalled — the polygon is likely not simple or is degenerate"
            )
        progressed = False
        for k in range(len(remaining)):
            i_prev = remaining[(k - 1) % len(remaining)]
            i_curr = remaining[k]
            i_next = remaining[(k + 1) % len(remaining)]

            if not is_convex_vertex(verts[i_prev], verts[i_curr], verts[i_next]):
                continue

            others = [
                idx for idx in remaining if idx not in (i_prev, i_curr, i_next)
            ]
            if any(
                point_in_triangle(verts[idx], verts[i_prev], verts[i_curr], verts[i_next])
                for idx in others
            ):
                continue

            triangles.append((i_prev, i_curr, i_next))
            remaining.pop(k)
            progressed = True
            attempts_without_progress = 0
            break

        if not progressed:
            attempts_without_progress += 1

    triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def three_colour(triangles: list[Triangle]) -> dict[int, int]:
    """Colour the triangulation's vertices with three colours.

    The dual graph of a polygon triangulation is a tree, so a breadth-first walk
    over adjacent triangles never revisits a triangle by two different routes —
    which is exactly why the greedy assignment below cannot conflict. Each new
    triangle shares an edge (two already-coloured vertices) with its parent, so
    the third vertex has exactly one colour available.
    """
    if not triangles:
        return {}

    # Adjacency over shared edges.
    edge_to_tris: dict[frozenset[int], list[int]] = {}
    for t_idx, tri in enumerate(triangles):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_tris.setdefault(frozenset((a, b)), []).append(t_idx)

    neighbours: dict[int, list[int]] = {i: [] for i in range(len(triangles))}
    for shared in edge_to_tris.values():
        if len(shared) == 2:
            x, y = shared
            neighbours[x].append(y)
            neighbours[y].append(x)

    colouring: dict[int, int] = {}
    seen: set[int] = set()

    # The dual of a simple polygon's triangulation is connected, but iterate over
    # every component anyway so a caller passing a partial fan still gets a
    # valid colouring.
    for root in range(len(triangles)):
        if root in seen:
            continue
        for vertex, colour in zip(triangles[root], (0, 1, 2), strict=True):
            colouring.setdefault(vertex, colour)
        seen.add(root)
        queue = deque([root])

        while queue:
            current = queue.popleft()
            for nxt in neighbours[current]:
                if nxt in seen:
                    continue
                seen.add(nxt)
                tri = triangles[nxt]
                known = {v: colouring[v] for v in tri if v in colouring}
                missing = [v for v in tri if v not in colouring]
                if len(missing) == 1:
                    available = {0, 1, 2} - set(known.values())
                    colouring[missing[0]] = available.pop()
                else:
                    # Only reachable for a disconnected dual; assign greedily.
                    free = [c for c in (0, 1, 2) if c not in set(known.values())]
                    # strict=False: there may be more free colours than
                    # uncoloured vertices, which is fine.
                    for v, c in zip(missing, free, strict=False):
                        colouring[v] = c
                queue.append(nxt)

    return colouring


def place_guards(polygon: Polygon) -> GuardSolution:
    """Solve the Art Gallery instance and return a verified placement.

    Guards go on the *least frequent* colour class. Since every triangle has one
    vertex of each colour, that class alone covers every triangle, and the
    smallest of three disjoint classes over ``n`` vertices has at most
    ``floor(n/3)`` members — which is Fisk's proof of the bound.

    Raises
    ------
    RuntimeError
        If the resulting placement fails independent coverage verification. The
        construction is believed correct, so this is a genuine assertion rather
        than an expected path.
    """
    triangles = triangulate(polygon)
    colouring = three_colour(triangles)

    classes: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for vertex, colour in colouring.items():
        classes[colour].append(vertex)

    guard_colour = min(classes, key=lambda c: len(classes[c]))
    guards = sorted(classes[guard_colour])

    solution = GuardSolution(polygon, triangles, colouring, guards, guard_colour)
    if not verify_coverage(solution):
        raise RuntimeError("guard placement failed coverage verification")
    return solution


def verify_coverage(solution: GuardSolution) -> bool:
    """Check the guard set independently of how it was produced.

    Every triangle of the triangulation must contain at least one guard vertex.
    A guard sitting on a triangle's corner sees that entire triangle (triangles
    are convex), so covering all triangles covers the polygon.
    """
    guards = set(solution.guards)
    return all(guards & set(tri) for tri in solution.triangles)
