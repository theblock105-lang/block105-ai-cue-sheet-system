# THE BLOCK 105 RADIO — AI Cue Sheet System

Production web wrapper around the proven Block 105 audio-identification workflow.

## Railway variable
Set `AUDD_API_TOKEN` to the station's AudD Enterprise API token.

## Flow
1. Broadcaster uploads the full show and enters the show title.
2. Full-audio AudD identification runs using accurate offsets.
3. Ambiguous scan regions are automatically converted into precision windows.
4. Precision AudD scans resolve candidates.
5. Sparse + precision evidence is merged into the working timeline.
6. Missing album metadata is filled as `Song Title (Single)`.
7. Live365 marker-density rules are applied.
8. Broadcaster downloads the final Live365 CSV.
