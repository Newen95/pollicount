# PolliCount

Zero-shot pollinator counting from an ordinary phone video, for citizen science.

Built during the Meta Global AI Developer Hackathon (Oct 26 - Nov 4, 2026). Everything in this repository was written during the event.

## Idea

A volunteer films a flower patch with a phone for 10 minutes. PolliCount:

1. tracks every insect and every flower with **SAM 3.1** (Meta Model API);
2. checks and classifies each insect track into FIT Count groups with **Muse Spark** (bumblebee, honeybee, solitary bee, hoverfly, other fly, butterfly/moth, wasp, beetle, small insect);
3. counts a **visit** when an insect stays on a flower for at least 0.3 s;
4. exports an annotated video and a FIT Count / Spipoll record the volunteer can correct.

## Status

Work in progress.

## License

MIT
