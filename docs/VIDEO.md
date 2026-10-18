# Video workflow

Generate style keyframes with the adapter, then send images and motion prompts to FLUX 3 Video. The image adapter is not loaded by the video model.

## Working montage

| Shot | Planned final seconds |
| --- | --- |
| Shrek entrance | 7 |
| Joker stairs | 8 |
| Squid Game doll | 8 |
| Anton Ego tastes | 6 |
| Ego childhood memory | 6 |
| Simba presentation on a skyscraper | 8 |
| Giant Joi hologram | 9 |

The core target is 52 seconds. These are planned shots, not a guarantee all are generated or accepted. The original castle opening experiment is excluded from the batch runner; retain service moderation boundaries and do not retry blocked requests through substitutions.

## Controlled generation

1. Select a start image with correct identity, anatomy, composition and style.
2. For controlled endpoints, edit that image into a related final pose. Do not ask the model to morph unrelated franchises into each other.
3. Place the selected paths and motion prompts in `films/<slug>/video-selection.json`; the example shows the schema.
4. Prepare locally, then submit ten-second silent HD drafts only when ready to incur API costs.
5. Review full clips for drift, faces, hands, contact, motion and architecture. Preserve job metadata and draft caches outside Git.
6. Enhance selected drafts through BFL's documented `draft_enhance` workflow. Enhancement is a separate step; the batch script currently generates drafts, not final enhanced clips.
7. Edit on movement, shape or sound rather than relying on generated cross-scene morphing.

For the project-specific seven-shot draft assembler, place completed clips at `films/pop-culture-recreation/video/<shot-id>/draft.mp4` and the corresponding seven entries in `video-selection.json`, with IDs and `target_seconds` matching the table in `merge_pop_videos.py`.

```bash
python merge_pop_videos.py
```

This makes a silent **1280×720, 24 fps draft**, not a Full HD master. It requires all seven source clips unless `--allow-partial` is explicitly requested; partial previews are not submissions. It does not synthesize missing video from stills. The output uses hard cuts and records source hashes and actual durations.

## Final delivery checklist

- At least 45 seconds of actual film; check the final duration rather than only summing the shot list.
- One licensed/original soundtrack and intentional sound effects; do not assume film music is cleared.
- Consistent image treatment, correct identities and readable actions across shots.
- No unintended logos, corrupted text, flash frames or missing clips.
- Export a broadly playable H.264/AAC MP4 and test the public viewing link.
- Keep a provenance record and distinguish draft renders from enhanced/final output.

The cleaned repository does not include API credentials, accepted-job polling URLs, signed media URLs or raw job responses. Never publish those artifacts when adding the final film link.
