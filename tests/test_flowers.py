from pollicount.flowers import group_touching

from samples import track


def static(object_id, box, frames=range(3)):
    return track(object_id, "flower", {f: box for f in frames})


def test_touching_florets_become_one_head():
    florets = [static("1", (0, 0, 100, 100)), static("2", (110, 0, 210, 100)), static("3", (50, 90, 150, 190))]
    heads = group_touching(florets)

    assert len(heads) == 1
    b = heads[0].boxes[0]
    assert (heads[0].object_id, b.left, b.top, b.right, b.bottom) == ("1", 0, 0, 210, 190)


def test_separate_flowers_stay_separate():
    flowers = [static("1", (0, 0, 100, 100)), static("2", (300, 0, 400, 100))]
    assert sorted(h.object_id for h in group_touching(flowers)) == ["1", "2"]


def test_head_box_follows_members_frame_by_frame():
    a = track("1", "flower", {0: (0, 0, 100, 100), 1: (5, 0, 105, 100)})
    b = track("2", "flower", {1: (100, 0, 200, 100)})
    head = group_touching([a, b])[0]

    assert (head.boxes[0].right, head.boxes[1].left, head.boxes[1].right) == (100, 5, 200)


def test_missing_floret_is_filled_from_its_nearest_frame_following_the_camera():
    offsets = [(10.0 * f, 0.0) for f in range(10)]  # the scene slides 10 px right per frame
    first = track("1", "flower", {f: (100 + 10 * f, 0, 200 + 10 * f, 100) for f in range(10)})
    late = track("2", "flower", {f: (210 + 10 * f, 0, 310 + 10 * f, 100) for f in range(6, 10)})
    head = group_touching([first, late], offsets=offsets, size=(1000, 500), max_fill=10)[0]

    b = head.boxes[2]
    assert (b.left, b.right) == (120, 330)


def test_flowers_seen_at_different_times_are_compared_in_the_scene():
    offsets = [(-100.0 * f, 0.0) for f in range(3)]  # panning right: the scene slides left
    a = track("1", "flower", {0: (100, 0, 200, 100)})
    b = track("2", "flower", {2: (100, 0, 200, 100)})  # same place on screen, 200 px further in the scene

    assert len(group_touching([a, b], offsets=offsets)) == 2
    assert len(group_touching([a, b])) == 1


def test_no_filling_across_a_camera_move():
    a = track("1", "flower", {0: (0, 0, 100, 100), 9: (0, 0, 100, 100)})
    head = group_touching([a], max_fill=10, camera_moves=[(3, 5)])[0]

    assert 2 in head.boxes and 6 in head.boxes and 4 not in head.boxes
