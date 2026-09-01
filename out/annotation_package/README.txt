ANNOTATION PACKAGE — nucleus segmentation (retina)
===================================================
For: Vincent · From: Denish · ~40 tiles, est. 4-6 hours total, any pace

WHAT THIS IS
  Each image is a small 512px tile from our retina scans, already enhanced for
  visibility. A model has pre-drawn DRAFT outlines around what it thinks are
  nuclei. Your job is to CORRECT the drafts, not draw from scratch:
    - delete outlines around things that are not nuclei
    - add outlines around nuclei it missed
    - fix outlines that merge two nuclei into one, or split one into two
    - roughly right outline shape is fine; exact pixel edges are not the goal

GETTING STARTED (no installation)
  1. Double-click  via.html  — it opens in your browser and works offline.
  2. Project -> Add local files -> select ALL images in the tiles/ folder.
  3. Annotation -> Import Annotations (from json) -> drafts_annotations_via.json
     (the draft outlines appear on every tile)
  4. Edit: click an outline to select. Press 'd' to delete. To add: choose the
     polygon tool (left toolbar), click around the nucleus, press Enter.
  5. SAVE OFTEN: Annotation -> Export Annotations (as json). Keep the file name,
     send it back when done (or per session — partial files are fine).

START WITH THE 3 PILOT TILES (files 001-003)
  We will do these together on a short call first, agree the rules below,
  and only then continue with the rest.

THE RULES (draft — we finalise together on the call)
  R1  Mark a nucleus if you are CONFIDENT it is one. Skip anything you would
      only be guessing at — consistency matters more than completeness.
  R2  Two touching nuclei = two outlines. This is the error that matters most.
  R3  A nucleus cut by the tile edge: outline the visible part only if more
      than about half of it is inside the tile; otherwise skip it.
  R4  Blurry blob that is clearly out-of-focus depth bleed: skip.
  R5  When torn between one-or-two nuclei, choose two only if you can see a
      dimmer line between them; otherwise one.
  (We will adjust these based on what we see together on the pilot tiles.)

WHAT WE DO WITH IT
  Your corrected outlines become (a) the training examples for the model and
  (b) the answer key we score every method against. Files 001-028 teach the
  model; the Region-2 files are the exam and never touch training.

Questions / anything odd in an image -> just note the file name and ask.
