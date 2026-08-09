"""Exact-ish primitives for simple polygons.

All predicates are written around the signed area (cross product) of a vertex
triple, which keeps the orientation logic in one place and avoids the sign bugs
that plague hand-rolled point-in-polygon code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "Point",
    "Polygon",
    "cross",
    "is_convex_vertex",
    "on_segment",
    "point_in_triangle",
    "polygon_area",
    "segment_intersects",
]

EPS = 1e-12

Point = tuple[float, float]


def cross(o: Point, a: Point, b: Point) -> float:
    """Twice the signed area of triangle ``(o, a, b)``.

    Positive when the turn ``o -> a -> b`` is counter-clockwise. Every other
    predicate in this module is expressed in terms of this one.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def polygon_area(vertices: list[Point]) -> float:
    """Signed area via the shoelace formula. Positive means counter-clockwise."""
    n = len(vertices)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def on_segment(p: Point, a: Point, b: Point) -> bool:
    """True when ``p`` lies on the closed segment ``ab`` (assumes collinearity)."""
    return (
        min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS
    )


def is_convex_vertex(prev: Point, curr: Point, nxt: Point, ccw: bool = True) -> bool:
    """True when ``curr`` is a convex corner for the given orientation."""
    turn = cross(prev, curr, nxt)
    return turn > EPS if ccw else turn < -EPS


def point_in_triangle(p: Point, a: Point, b: Point, c: Point, strict: bool = False) -> bool:
    """Point-in-triangle by consistent sign of the three sub-triangle areas.

    With ``strict=False`` the boundary counts as inside, which is what ear
    clipping needs (a vertex touching an ear's edge still blocks the ear).
    """
    d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
    # With strict=False a zero cross product (p exactly on an edge) must count
    # as neither negative nor positive, so the point stays "inside". With
    # strict=True the comparison flips and a boundary point falls outside.
    tol = -EPS if strict else EPS
    has_neg = (d1 < -tol) or (d2 < -tol) or (d3 < -tol)
    has_pos = (d1 > tol) or (d2 > tol) or (d3 > tol)
    return not (has_neg and has_pos)


def segment_intersects(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True when segments ``p1p2`` and ``p3p4`` properly or improperly cross."""
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS)) and (
        (d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS)
    ):
        return True

    # Collinear touching cases.
    if abs(d1) <= EPS and on_segment(p1, p3, p4):
        return True
    if abs(d2) <= EPS and on_segment(p2, p3, p4):
        return True
    if abs(d3) <= EPS and on_segment(p3, p1, p2):
        return True
    return bool(abs(d4) <= EPS and on_segment(p4, p1, p2))


@dataclass(frozen=True)
class Polygon:
    """A simple polygon, stored counter-clockwise.

    Construction normalises orientation so every downstream predicate can assume
    CCW, and validates simplicity — a self-intersecting input would silently
    break both the triangulation and the theorem it demonstrates.
    """

    vertices: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError(f"a polygon needs at least 3 vertices, got {len(self.vertices)}")
        # Simplicity is checked first: a symmetric bow-tie has exactly zero
        # shoelace area, and "edges intersect" is the more useful diagnosis.
        if not self._is_simple():
            raise ValueError("polygon is not simple (edges intersect)")
        if abs(polygon_area(list(self.vertices))) < EPS:
            raise ValueError("polygon is degenerate (zero area)")
        if polygon_area(list(self.vertices)) < 0:
            object.__setattr__(self, "vertices", tuple(reversed(self.vertices)))

    @classmethod
    def from_list(cls, points) -> Polygon:
        return cls(tuple((float(x), float(y)) for x, y in points))

    def _is_simple(self) -> bool:
        n = len(self.vertices)
        for i in range(n):
            a1, a2 = self.vertices[i], self.vertices[(i + 1) % n]
            for j in range(i + 1, n):
                # Skip the shared-endpoint pairs: adjacent edges always touch.
                if j == i or (j + 1) % n == i or j == (i + 1) % n:
                    continue
                b1, b2 = self.vertices[j], self.vertices[(j + 1) % n]
                if segment_intersects(a1, a2, b1, b2):
                    return False
        return True

    @property
    def n(self) -> int:
        return len(self.vertices)

    @property
    def area(self) -> float:
        return abs(polygon_area(list(self.vertices)))

    @property
    def theoretical_bound(self) -> int:
        """``floor(n / 3)`` — the Art Gallery Theorem's sufficiency bound."""
        return self.n // 3

    def reflex_vertices(self) -> list[int]:
        """Indices of the concave corners.

        A convex polygon has none and needs exactly one guard; the count of
        reflex vertices is the honest measure of how hard an instance is.
        """
        n = self.n
        return [
            i
            for i in range(n)
            if not is_convex_vertex(
                self.vertices[(i - 1) % n], self.vertices[i], self.vertices[(i + 1) % n]
            )
        ]

    @property
    def is_convex(self) -> bool:
        return not self.reflex_vertices()

    def contains(self, p: Point) -> bool:
        """Ray-casting point-in-polygon test, boundary counted as inside."""
        n = self.n
        for i in range(n):
            a, b = self.vertices[i], self.vertices[(i + 1) % n]
            if abs(cross(a, b, p)) <= EPS and on_segment(p, a, b):
                return True

        inside = False
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            if (y1 > p[1]) != (y2 > p[1]):
                x_cross = x1 + (p[1] - y1) * (x2 - x1) / (y2 - y1)
                if p[0] < x_cross:
                    inside = not inside
        return inside

    def sees(self, guard: Point, target: Point) -> bool:
        """True when ``target`` is visible from ``guard`` inside the polygon.

        Visibility means the open segment stays strictly inside: it must not
        cross any edge, and its midpoint must lie in the interior (which rules
        out a segment that runs outside a reflex notch while touching only
        vertices).
        """
        n = self.n
        for i in range(n):
            a, b = self.vertices[i], self.vertices[(i + 1) % n]
            if _shares_endpoint(guard, target, a, b):
                continue
            if _properly_crosses(guard, target, a, b):
                return False
        mid = ((guard[0] + target[0]) / 2.0, (guard[1] + target[1]) / 2.0)
        return self.contains(mid)

    def bounding_box(self) -> tuple[float, float, float, float]:
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        return min(xs), min(ys), max(xs), max(ys)


def _shares_endpoint(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    return any(
        math.isclose(p[0], q[0], abs_tol=1e-9) and math.isclose(p[1], q[1], abs_tol=1e-9)
        for p in (p1, p2)
        for q in (q1, q2)
    )


def _properly_crosses(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Strict crossing — touching at a shared endpoint does not count."""
    d1, d2 = cross(q1, q2, p1), cross(q1, q2, p2)
    d3, d4 = cross(p1, p2, q1), cross(p1, p2, q2)
    return ((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS)) and (
        (d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS)
    )
