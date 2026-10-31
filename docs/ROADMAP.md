# Roadmap

Submission: Wed 4 Nov, 22:00 Paris (official deadline Thu 5 Nov, 03:00). Judged: 3-minute demo video, code, live product.

Goal: a responsive web app, on desktop and phone, that a volunteer genuinely enjoys using. Film or pick a video,
watch it being analysed, get an annotated video and a FIT Count record to check and send.

Each level must be shippable on its own: if time runs out, the previous level is what gets submitted.

## Level 1, MVP (Sat 31 Oct)

- [x] Real pipeline run: SAM 3.1 tracking + Muse Spark Open classification (close-up video), costs logged
- [x] Annotated video: boxes, group labels, visit counter (drawn in the browser)
- [x] FastAPI backend: upload, background analysis, progress, results
- [x] Web app: upload or film, live progress, annotated video, FIT Count summary, examples
- [x] Deployed on Modal with a public URL
- [x] Checked on a phone

## Level 2 (Mon 2 Nov)

- [ ] Far-field mode: motion candidates verified by Muse Spark Open (T1: visit F1 0.62 vs 0 with SAM alone), in the pipeline
- [ ] Accuracy measured on the Monash annotated videos (visit precision and recall, classification accuracy)
- [ ] Backend comparison table: Muse Spark, Muse Spark Open, Muse Glimmer (accuracy, cost per video, time)
- [ ] Volunteer review: correct a group, discard a false visit, totals update instantly
- [ ] Optional FIT Count record, following the official PoMS form field by field: date, time and place filled
  automatically, weather from Open-Meteo, target flower and flower count proposed by the analysis, habitat,
  patch context and identification level remembered on the phone; ready to copy into the PoMS app
- [ ] Polished mobile experience: one-hand flow, PWA install, clear states for slow networks

## Level 3 (Tue 3 Nov)

- [ ] Review agent: re-checks low-confidence tracks (re-prompts SAM with points, asks for more crops, explains its decision)
- [ ] Offline mode: Muse Glimmer served on Modal, then on a laptop in the field
- [ ] FIT Count submission-ready export
- [ ] Group illustrations generated with Muse Image

## Freeze (Wed 4 Nov)

- [ ] No new features
- [ ] Demo video: script, screen recordings of the real app, voice-over, edit (3 min)
- [ ] README: problem, how it works, results table, credits
- [ ] Repository public (Tue 3 Nov), submission form sent

## UI and UX principles

- Beautiful first: a distinct visual identity (nature, field notebook), not a generic dashboard.
- Fluid: no blocking spinner. Progress is shown as it happens (frames tracked, insects found, visits counted).
- Immediate value: the first result (annotated video) appears before the full report.
- Mobile first: large touch targets, one-hand use outdoors, readable in sunlight.
- Honest: every count shows its confidence, and the volunteer can correct it.

## Other deadlines

- Fireworks credits: redeem before Tue 3 Nov
- Workshops 17:00 Paris: Fri 30 Oct (Fireworks, Muse open models)
