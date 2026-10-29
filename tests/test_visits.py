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
