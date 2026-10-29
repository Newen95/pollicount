import json

from pollicount import export
from pollicount.classify import NOT_AN_INSECT, Classification, parse_reply
from pollicount.visits import Visit


def test_parse_reply_reads_json_inside_fences():
    reply = (
        'Thinking...\n```json\n{"group": "Hoverflies", "confidence": 0.65, "visible_cues": ["one pair of wings"]}\n```'
    )
    result = parse_reply(reply)

    assert (result.group, result.confidence, result.cues) == ("hoverflies", 0.65, ["one pair of wings"])


def test_parse_reply_falls_back_on_unknown_group():
    result = parse_reply('{"group": "dragonfly", "confidence": 0.9}')
    assert (result.group, result.confidence) == ("other insects", 0.0)


def test_parse_reply_survives_garbage():
    assert parse_reply("no json here").group == "other insects"


def test_parse_reply_clamps_confidence():
    assert parse_reply('{"group": "wasps", "confidence": 7}').confidence == 1.0


def test_fit_count_drops_non_insects_and_counts_per_group(tmp_path):
    visits = [
        Visit("1", "f", 0, 9, 10),
        Visit("1", "f", 30, 39, 10),
        Visit("2", "f", 5, 9, 10),
        Visit("3", "f", 0, 9, 10),
    ]
    groups = {
        "1": Classification("honeybees", 0.9),
        "2": Classification("hoverflies", 0.7),
        "3": Classification(NOT_AN_INSECT, 0.8),
    }
    record = export.fit_count(visits, groups, duration_s=600, flowers=4)

    assert record["counts"]["honeybees"] == 2
    assert record["counts"]["hoverflies"] == 1
    assert record["total_visits"] == 3
    assert record["insects"] == 2
    assert record["duration_min"] == 10

    export.write(record, tmp_path)
    assert json.loads((tmp_path / "fit_count.json").read_text())["total_visits"] == 3
    assert (tmp_path / "fit_count.csv").read_text().splitlines()[1].endswith(",3")
