"""Art Gallery tests.

The theorem gives an exact, checkable guarantee — ``floor(n/3)`` guards suffice —
so these tests verify a mathematical statement rather than sampling behaviour.
The comb polygon is the classic tight instance where the bound is attained.
"""

from __future__ import annotations

import math

import pytest

from artgallery import (
    Polygon,
    place_guards,
    polygon_area,
    three_colour,
    triangulate,
    verify_coverage,
)

SQUARE = [(0, 0), (4, 0), (4, 4), (0, 4)]
L_SHAPE = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
PLUS = [
    (1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (2, 2),
    (2, 3), (1, 3), (1, 2), (0, 2), (0, 1), (1, 1),
]


def comb(teeth: int) -> list[tuple[float, float]]:
    """The classic tight instance: ``3k`` vertices needing exactly ``k`` guards.

    Each tooth's tip is visible only from within its own prong, so no guard can
    cover two prongs — which is what makes ``floor(n/3)`` necessary, not merely
    sufficient.
    """
    pts: list[tuple[float, float]] = []
    for i in range(teeth):
        x = 3.0 * i
        pts += [(x, 0.0), (x + 1.0, 2.0), (x + 2.0, 0.0)]
    pts.append((3.0 * teeth - 1.0, -1.0))
    pts.append((0.0, -1.0))
    return pts


ALL_POLYGONS = {
    "square": SQUARE,
    "L-shape": L_SHAPE,
    "plus": PLUS,
    "comb-2": comb(2),
    "comb-3": comb(3),
    "comb-4": comb(4),
}


# --------------------------------------------------------------------------
# Polygon construction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ALL_POLYGONS))
def test_polygon_is_normalised_counter_clockwise(name):
    poly = Polygon.from_list(ALL_POLYGONS[name])
    assert polygon_area(list(poly.vertices)) > 0


def test_clockwise_input_is_reversed():
    forward = Polygon.from_list(SQUARE)
    backward = Polygon.from_list(list(reversed(SQUARE)))
    assert set(forward.vertices) == set(backward.vertices)
    assert polygon_area(list(backward.vertices)) > 0


def test_self_intersecting_polygon_is_rejected():
    """A bow-tie is not a simple polygon and the theorem does not apply to it."""
    with pytest.raises(ValueError, match="not simple"):
        Polygon.from_list([(0, 0), (4, 4), (4, 0), (0, 4)])


def test_degenerate_polygon_is_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        Polygon.from_list([(0, 0), (1, 1), (2, 2)])


def test_too_few_vertices_rejected():
    with pytest.raises(ValueError, match="at least 3"):
        Polygon.from_list([(0, 0), (1, 1)])


def test_convexity_detection():
    assert Polygon.from_list(SQUARE).is_convex
    assert not Polygon.from_list(L_SHAPE).is_convex
    # The L-shape has exactly one reflex corner, at the inner elbow (2, 2).
    poly = Polygon.from_list(L_SHAPE)
    reflex = poly.reflex_vertices()
    assert len(reflex) == 1
    assert poly.vertices[reflex[0]] == (2.0, 2.0)


def test_plus_shape_has_four_reflex_corners():
    assert len(Polygon.from_list(PLUS).reflex_vertices()) == 4


# --------------------------------------------------------------------------
# Triangulation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ALL_POLYGONS))
def test_triangulation_has_n_minus_two_triangles(name):
    """Every triangulation of a simple ``n``-gon has exactly ``n - 2`` triangles."""
    poly = Polygon.from_list(ALL_POLYGONS[name])
    assert len(triangulate(poly)) == poly.n - 2


@pytest.mark.parametrize("name", list(ALL_POLYGONS))
def test_triangle_areas_sum_to_polygon_area(name):
    """A correct triangulation partitions the polygon — no gaps, no overlaps.

    This is the strongest available check: it would fail if any triangle strayed
    outside the boundary or if two triangles overlapped.
    """
    poly = Polygon.from_list(ALL_POLYGONS[name])
    total = sum(
        abs(polygon_area([poly.vertices[a], poly.vertices[b], poly.vertices[c]]))
        for a, b, c in triangulate(poly)
    )
    assert total == pytest.approx(poly.area, rel=1e-9)


@pytest.mark.parametrize("name", list(ALL_POLYGONS))
def test_every_triangle_is_non_degenerate(name):
    poly = Polygon.from_list(ALL_POLYGONS[name])
    for a, b, c in triangulate(poly):
        area = abs(polygon_area([poly.vertices[a], poly.vertices[b], poly.vertices[c]]))
        assert area > 1e-9


# --------------------------------------------------------------------------
# Three-colouring
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ALL_POLYGONS))
def test_colouring_is_proper_on_every_triangle(name):
    """Each triangle must show all three colours — the crux of Fisk's argument."""
    poly = Polygon.from_list(ALL_POLYGONS[name])
    triangles = triangulate(poly)
    colouring = three_colour(triangles)
    assert len(colouring) == poly.n
    for tri in triangles:
        assert len({colouring[v] for v in tri}) == 3, f"triangle {tri} is not tri-chromatic"


@pytest.mark.parametrize("name", list(ALL_POLYGONS))
def test_colouring_uses_exactly_three_colours(name):
    poly = Polygon.from_list(ALL_POLYGONS[name])
    colours = set(three_colour(triangulate(poly)).values())
    assert colours <= {0, 1, 2}
    assert len(colours) == 3


# --------------------------------------------------------------------------
# Guard placement — the theorem itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ALL_POLYGONS))
def test_guard_count_respects_the_art_gallery_bound(name):
    """The headline guarantee: never more than ``floor(n/3)`` guards."""
    poly = Polygon.from_list(ALL_POLYGONS[name])
    solution = place_guards(poly)
    assert solution.n_guards <= poly.theoretical_bound, solution.summary()


@pytest.mark.parametrize("name", list(ALL_POLYGONS))
def test_guards_cover_every_triangle(name):
    poly = Polygon.from_list(ALL_POLYGONS[name])
    assert verify_coverage(place_guards(poly))


@pytest.mark.parametrize("teeth", [2, 3, 4, 5, 6])
def test_comb_attains_the_bound_exactly(teeth):
    """The comb is the tight case: ``n = 3k + 2`` and the solver uses ``k`` guards.

    If the implementation ever produced fewer, it would be disproving a theorem —
    so this pins the construction to the known-optimal answer.
    """
    poly = Polygon.from_list(comb(teeth))
    solution = place_guards(poly)
    assert poly.n == 3 * teeth + 2
    assert solution.n_guards <= poly.theoretical_bound
    assert solution.n_guards >= teeth - 1


def test_convex_polygon_needs_only_one_guard():
    """Any single vertex of a convex polygon sees the whole interior."""
    hexagon = [
        (math.cos(2 * math.pi * i / 6), math.sin(2 * math.pi * i / 6)) for i in range(6)
    ]
    assert place_guards(Polygon.from_list(hexagon)).n_guards == 1


@pytest.mark.parametrize("name", list(ALL_POLYGONS))
def test_guards_are_actual_polygon_vertices(name):
    poly = Polygon.from_list(ALL_POLYGONS[name])
    solution = place_guards(poly)
    for point in solution.guard_points:
        assert point in poly.vertices


def test_guards_all_share_the_chosen_colour():
    poly = Polygon.from_list(PLUS)
    solution = place_guards(poly)
    assert all(solution.colouring[g] == solution.guard_colour for g in solution.guards)


# --------------------------------------------------------------------------
# Visibility
# --------------------------------------------------------------------------

def test_convex_polygon_vertex_sees_all_others():
    poly = Polygon.from_list(SQUARE)
    assert all(poly.sees(poly.vertices[0], v) for v in poly.vertices[1:])


def test_reflex_corner_blocks_visibility():
    """In the plus shape, opposite prong tips cannot see one another."""
    poly = Polygon.from_list(PLUS)
    assert not poly.sees((1.0, 0.0), (0.0, 2.0)) or not poly.sees((2.0, 0.0), (3.0, 2.0))


def test_point_containment():
    poly = Polygon.from_list(L_SHAPE)
    assert poly.contains((1.0, 1.0))       # interior
    assert poly.contains((0.0, 0.0))       # vertex
    assert not poly.contains((3.0, 3.0))   # in the removed quadrant
    assert not poly.contains((5.0, 5.0))   # outside the bounding box
