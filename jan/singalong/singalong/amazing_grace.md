# Amazing Grace: the demo song

Two sources of truth, in this order of preference:

1. **`data/damp100/reference.npz`** (made by `jan/prep/precompute_damp.py` from the DAMP-S-AG backing-track MIDI):
   frame-wise melody, chord pitch classes, key. This is what the SNN was trained against and what the live
   demo should follow when the singer sings along to that arrangement (Eb major, two key changes).
2. **`amazing_grace.py` lead sheet** (this folder): the common hymnal version, 3/4, written in scale degrees, so
   it works in any key. Used when the reference is absent (this cloud session, synthetic data, teammates'
   laptops without the DAMP download) and as the `song` mode prior for the HMM selector.

The two can differ in passing chords. For the live demo the selector only needs the *progression*
(I I IV I I I V V | I I IV I I V I I), which both share.
