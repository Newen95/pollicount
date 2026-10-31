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
