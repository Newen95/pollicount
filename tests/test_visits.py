from pollicount.visits import VisitRule, find_visits

from samples import track

FPS = 10
FLOWER = track("f1", "flower", {f: (100, 100, 200, 200) for f in range(100)})


def perched(frames, x=140, y=140, size=20):
    return {f: (x, y, x + size, y + size) for f in frames}


def test_insect_resting_long_enough_is_one_visit():
    insect = track("i1", "insect", perched(range(10, 15)))
    visits = find_visits([insect], [FLOWER], FPS)

    assert len(visits) == 1
    assert (visits[0].flower_id, visits[0].start_frame, visits[0].end_frame) == ("f1", 10, 14)
    assert visits[0].duration_s == 0.5


def test_touch_shorter_than_threshold_is_ignored():
    insect = track("i1", "insect", perched(range(10, 12)))
    assert find_visits([insect], [FLOWER], FPS) == []


def test_short_tracking_dropout_does_not_split_a_visit():
    insect = track("i1", "insect", perched([*range(10, 14), *range(16, 20)]))
    visits = find_visits([insect], [FLOWER], FPS)

    assert [(v.start_frame, v.end_frame) for v in visits] == [(10, 19)]


def test_long_absence_makes_two_visits():
    insect = track("i1", "insect", perched([*range(10, 14), *range(30, 34)]))
    assert len(find_visits([insect], [FLOWER], FPS)) == 2


def test_insect_flying_across_flower_is_not_a_visit():
    flight = {f: (100 + 15 * (f - 10), 140, 120 + 15 * (f - 10), 160) for f in range(10, 17)}
    assert find_visits([track("i1", "insect", flight)], [FLOWER], FPS) == []


def test_insect_next_to_flower_is_not_a_visit():
    insect = track("i1", "insect", perched(range(10, 20), x=195))
    assert find_visits([insect], [FLOWER], FPS) == []


def test_moving_between_two_flowers_gives_one_visit_each():
    second = track("f2", "flower", {f: (300, 100, 400, 200) for f in range(100)})
    insect = track("i1", "insect", {**perched(range(10, 15)), **perched(range(20, 25), x=340)})
    visits = find_visits([insect], [FLOWER, second], FPS)

    assert [v.flower_id for v in visits] == ["f1", "f2"]


def test_threshold_is_configurable():
    insect = track("i1", "insect", perched(range(10, 12)))
    assert len(find_visits([insect], [FLOWER], FPS, VisitRule(min_duration_s=0.2))) == 1


def test_flower_missing_for_a_few_frames_keeps_the_visit():
    flower = track("f1", "flower", {f: (100, 100, 200, 200) for f in range(100) if f not in (12, 13)})
    insect = track("i1", "insect", perched(range(10, 15)))
    assert [(v.start_frame, v.end_frame) for v in find_visits([insect], [flower], FPS)] == [(10, 14)]


def test_large_close_up_insect_counts_when_its_centre_is_on_the_flower():
    wings_out = {f: (110, 60, 270, 220) for f in range(10, 15)}
    visits = find_visits([track("i1", "insect", wings_out)], [FLOWER], FPS)

    assert [(v.start_frame, v.end_frame) for v in visits] == [(10, 14)]


def test_flower_holding_the_centre_wins():
    big = track("big", "flower", {f: (0, 0, 300, 300) for f in range(100)})
    small = track("small", "flower", {f: (140, 140, 170, 170) for f in range(100)})
    visits = find_visits([track("i1", "insect", perched(range(10, 15), x=145, y=145))], [big, small], FPS)

    assert [v.flower_id for v in visits] == ["small"]


def test_insect_standing_on_top_of_a_flower_counts_by_its_legs():
    side_view = {f: (120, 20, 180, 120) for f in range(10, 15)}  # body above, legs on the top of the flower
    visits = find_visits([track("i1", "insect", side_view)], [FLOWER], FPS)

    assert [(v.start_frame, v.end_frame) for v in visits] == [(10, 14)]


def test_butterfly_with_wings_folded_up_counts_by_its_feet():
    # A tall narrow box whose bottom edge just reaches into the top of the flower.
    butterfly = track("b", "insect", {f: (130, 20, 170, 105) for f in range(10)})

    assert [v.duration_s for v in find_visits([butterfly], [FLOWER], FPS)] == [1.0]
    assert find_visits([butterfly], [FLOWER], FPS, VisitRule(feet_on_top=False)) == []


def test_insect_hovering_beside_a_flower_does_not_count_by_its_feet():
    beside = track("h", "insect", {f: (210, 120, 240, 150) for f in range(10)})
    assert find_visits([beside], [FLOWER], FPS) == []
