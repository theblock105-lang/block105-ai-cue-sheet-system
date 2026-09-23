import csv
import os
import re
import statistics
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request, send_file


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024

AUDD_URL = "https://enterprise.audd.io/"
JOBS = {}
LOCK = threading.Lock()


def parse_seconds(value):
    if value is None:
        return None

    value = str(value).strip()
    parts = value.split(":")

    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)

        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)

        return float(value)

    except (ValueError, TypeError):
        return None


def format_time(seconds):
    ms = max(0, round(float(seconds) * 1000))

    h = ms // 3600000
    ms %= 3600000

    m = ms // 60000
    ms %= 60000

    s = ms // 1000
    ms %= 1000

    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def normalize(value):
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def song_key(artist, title):
    return normalize(artist), normalize(title)


def ffmpeg_extract(source, start, duration, output):
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(max(0, start)),
        "-i",
        str(source),
        "-t",
        str(duration),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "44100",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "4",
        str(output),
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def audd_scan(audio_file, token):
    """
    Main scan.

    One 12-second recognition sample approximately every
    60 seconds of show audio.

    This is our proven working baseline.
    """

    data = {
        "api_token": token,
        "accurate_offsets": "true",
        "skip": "4",
        "every": "1",
        "limit": "60",
    }

    with open(audio_file, "rb") as f:
        response = requests.post(
            AUDD_URL,
            data=data,
            files={"file": f},
            timeout=1200,
        )

    response.raise_for_status()

    payload = response.json()

    if payload.get("status") == "error":
        raise RuntimeError(
            str(payload.get("error") or payload)
        )

    return payload


def collect_candidates(payload, base_offset=0):
    songs = {}
    scans = 0
    candidates = 0

    for scan in payload.get("result", []):
        scans += 1

        scan_offset = parse_seconds(
            scan.get("offset")
        )

        if scan_offset is None:
            continue

        for candidate in scan.get("songs", []):

            if not isinstance(candidate, dict):
                continue

            artist = str(
                candidate.get("artist") or ""
            ).strip()

            title = str(
                candidate.get("title") or ""
            ).strip()

            if not artist or not title:
                continue

            timecode = parse_seconds(
                candidate.get("timecode")
            )

            if timecode is None:
                continue

            try:
                start_offset = (
                    float(
                        candidate.get(
                            "start_offset",
                            0,
                        )
                    )
                    / 1000.0
                )
            except (ValueError, TypeError):
                start_offset = 0

            estimated_start = (
                base_offset
                + scan_offset
                + start_offset
                - timecode
            )

            key = song_key(
                artist,
                title,
            )

            songs.setdefault(
                key,
                {
                    "artist": artist,
                    "title": title,
                    "album": str(
                        candidate.get("album") or ""
                    ).strip(),
                    "year": str(
                        candidate.get(
                            "release_date"
                        ) or ""
                    )[:4],
                    "estimates": [],
                },
            )["estimates"].append(
                estimated_start
            )

            candidates += 1

    return songs, scans, candidates


def best_cluster(values, window):
    if not values:
        return []

    values = sorted(values)
    best = []

    for i, start in enumerate(values):

        cluster = [
            v
            for v in values[i:]
            if v - start <= window
        ]

        if len(cluster) > len(best):
            best = cluster

    return best


def build_sparse_timeline(songs):
    """
    Working song-detection logic.

    A valid single AudD identification is retained.
    Multiple detections are clustered when available.
    """

    out = []

    for song in songs.values():

        estimates = song.get(
            "estimates",
            [],
        )

        if not estimates:
            continue

        cluster = best_cluster(
            estimates,
            35.0,
        )

        if not cluster:
            continue

        out.append(
            {
                "artist": song["artist"],
                "title": song["title"],
                "album": song["album"],
                "year": song["year"],
                "start": statistics.median(
                    cluster
                ),
                "detections": len(cluster),
                "spread": (
                    max(cluster) - min(cluster)
                    if len(cluster) > 1
                    else 0
                ),
                "source": "SPARSE",
            }
        )

    return sorted(
        out,
        key=lambda x: x["start"],
    )


def find_ambiguous_times(payload):
    times = []

    for scan in payload.get(
        "result",
        [],
    ):

        keys = set()

        for song in scan.get(
            "songs",
            [],
        ):

            artist = str(
                song.get("artist") or ""
            ).strip()

            title = str(
                song.get("title") or ""
            ).strip()

            if artist and title:
                keys.add(
                    song_key(
                        artist,
                        title,
                    )
                )

        if len(keys) > 1:

            t = parse_seconds(
                scan.get("offset")
            )

            if t is not None:
                times.append(t)

    return sorted(set(times))


def build_precision_windows(times):
    windows = []

    for center in times:

        start = max(
            0,
            center - 60,
        )

        end = center + 90

        if (
            windows
            and start <= windows[-1]["end"]
        ):

            windows[-1]["end"] = max(
                windows[-1]["end"],
                end,
            )

        else:

            windows.append(
                {
                    "start": start,
                    "end": end,
                }
            )

    return windows


def precision_scan(
    audio,
    windows,
    token,
    workdir,
):
    """
    Precision scan.

    COST OPTIMIZATION:
    We now scan every other 12-second chunk.

    skip=1 / every=1
    = scan 12 seconds
    = skip 12 seconds
    = repeat

    This cuts precision recognition usage
    approximately in half compared with skip=0.
    """

    merged = {}
    total_scans = 0

    for i, window in enumerate(
        windows,
        1,
    ):

        out = (
            Path(workdir)
            / f"precision_{i}.mp3"
        )

        ffmpeg_extract(
            audio,
            window["start"],
            window["end"]
            - window["start"],
            out,
        )

        data = {
            "api_token": token,
            "accurate_offsets": "true",

            # COST OPTIMIZATION
            "skip": "1",
            "every": "1",

            # Safety ceiling for each precision window.
            "limit": "12",
        }

        with open(out, "rb") as f:

            response = requests.post(
                AUDD_URL,
                data=data,
                files={"file": f},
                timeout=1200,
            )

        response.raise_for_status()

        payload = response.json()

        if payload.get("status") == "error":
            continue

        for scan in payload.get(
            "result",
            [],
        ):

            total_scans += 1

            scan_offset = parse_seconds(
                scan.get("offset")
            )

            if scan_offset is None:
                continue

            for candidate in scan.get(
                "songs",
                [],
            ):

                if not isinstance(
                    candidate,
                    dict,
                ):
                    continue

                artist = str(
                    candidate.get(
                        "artist"
                    )
                    or ""
                ).strip()

                title = str(
                    candidate.get(
                        "title"
                    )
                    or ""
                ).strip()

                if not artist or not title:
                    continue

                tc = parse_seconds(
                    candidate.get(
                        "timecode"
                    )
                )

                if tc is None:
                    continue

                try:
                    so = (
                        float(
                            candidate.get(
                                "start_offset",
                                0,
                            )
                        )
                        / 1000
                    )

                except (
                    ValueError,
                    TypeError,
                ):
                    so = 0

                start = (
                    window["start"]
                    + scan_offset
                    + so
                    - tc
                )

                key = song_key(
                    artist,
                    title,
                )

                merged.setdefault(
                    key,
                    {
                        "artist": artist,
                        "title": title,
                        "album": str(
                            candidate.get(
                                "album"
                            )
                            or ""
                        ).strip(),
                        "year": str(
                            candidate.get(
                                "release_date"
                            )
                            or ""
                        )[:4],
                        "estimates": [],
                    },
                )["estimates"].append(
                    start
                )

    overrides = []

    for song in merged.values():

        cluster = best_cluster(
            song["estimates"],
            8.0,
        )

        if len(cluster) < 2:
            continue

        overrides.append(
            {
                "artist": song["artist"],
                "title": song["title"],
                "album": song["album"],
                "year": song["year"],
                "start": statistics.median(
                    cluster
                ),
                "detections": len(cluster),
                "spread": (
                    max(cluster)
                    - min(cluster)
                ),
                "source": "PRECISION",
            }
        )

    return (
        sorted(
            overrides,
            key=lambda x: x["start"],
        ),
        total_scans,
    )


def merge_timelines(
    sparse,
    precision,
):
    combined = []
    used = set()

    for item in sparse:

        matches = [
            (i, p)
            for i, p in enumerate(
                precision
            )
            if (
                i not in used
                and song_key(
                    item["artist"],
                    item["title"],
                )
                == song_key(
                    p["artist"],
                    p["title"],
                )
            )
        ]

        if matches:

            i, p = min(
                matches,
                key=lambda x: abs(
                    item["start"]
                    - x[1]["start"]
                ),
            )

            used.add(i)
            combined.append(
                p.copy()
            )

        else:

            combined.append(
                item.copy()
            )

    for i, p in enumerate(
        precision
    ):

        if i not in used:
            combined.append(
                p.copy()
            )

    combined.sort(
        key=lambda x: x["start"]
    )

    final = []

    for item in combined:

        if (
            final
            and song_key(
                item["artist"],
                item["title"],
            )
            == song_key(
                final[-1]["artist"],
                final[-1]["title"],
            )
            and abs(
                item["start"]
                - final[-1]["start"]
            )
            <= 20
        ):

            if (
                item["source"]
                == "PRECISION"
            ):
                final[-1] = item

        else:

            final.append(
                item
            )

    return final


def build_live365(
    timeline,
    show_title,
):
    markers = [
        {
            "offset": "00:00:00.000",
            "media_type": "talk",
            "title": show_title,
            "artist": show_title,
            "album": "",
            "year": "",
        }
    ]

    for item in timeline:

        ts = format_time(
            item["start"]
        )

        if ts == "00:00:00.000":
            ts = "00:00:00.001"

        album = (
            item.get("album")
            or f"{item['title']} (Single)"
        )

        markers.append(
            {
                "offset": ts,
                "media_type": "music",
                "title": item["title"],
                "artist": item["artist"],
                "album": album,
                "year": item.get(
                    "year",
                    "",
                ),
            }
        )

    def ms(ts):
        h, m, s = ts.split(":")
        sec, milli = s.split(".")

        return (
            int(h) * 3600000
            + int(m) * 60000
            + int(sec) * 1000
            + int(milli)
        )

    limits = [
        (600000, 5),
        (900000, 7),
        (1800000, 13),
        (4680000, 33),
    ]

    accepted = []
    corrections = 0

    for marker in markers:

        if marker["media_type"] == "talk":
            accepted.append(marker)
            continue

        t = ms(
            marker["offset"]
        )

        violates = any(
            sum(
                1
                for x in accepted
                if (
                    x["media_type"]
                    == "music"
                    and 0
                    <= t - ms(
                        x["offset"]
                    )
                    <= window
                )
            )
            >= maximum
            for window, maximum in limits
        )

        if not violates:

            accepted.append(
                marker
            )

        else:

            for prev in reversed(
                accepted
            ):

                if (
                    prev["media_type"]
                    == "music"
                ):

                    prev.update(
                        {
                            "media_type": "talk",
                            "title": show_title,
                            "artist": show_title,
                            "album": "",
                            "year": "",
                        }
                    )

                    corrections += 1
                    break

    return accepted, corrections


def write_csv(
    markers,
    path,
):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.writer(f)

        w.writerow(
            [
                "offset",
                "media_type",
                "title",
                "artist",
                "album",
                "year",
            ]
        )

        for m in markers:

            w.writerow(
                [
                    m["offset"],
                    m["media_type"],
                    m["title"],
                    m["artist"],
                    m["album"],
                    m["year"],
                ]
            )


def process_job(
    job_id,
    audio_path,
    show_title,
    workdir,
):
    try:

        token = os.environ.get(
            "AUDD_API_TOKEN"
        )

        if not token:
            raise RuntimeError(
                "AUDD_API_TOKEN is not configured on the server."
            )

        JOBS[job_id]["status"] = "running"

        JOBS[job_id]["message"] = (
            "Running full-audio identification…"
        )

        sparse_payload = audd_scan(
            audio_path,
            token,
        )

        (
            sparse_songs,
            sparse_scans,
            sparse_candidates,
        ) = collect_candidates(
            sparse_payload
        )

        sparse = build_sparse_timeline(
            sparse_songs
        )

        ambiguous = find_ambiguous_times(
            sparse_payload
        )

        windows = build_precision_windows(
            ambiguous
        )

        JOBS[job_id]["message"] = (
            f"Precision-checking "
            f"{len(windows)} ambiguous region(s)…"
        )

        if windows:

            (
                precision,
                precision_scans,
            ) = precision_scan(
                audio_path,
                windows,
                token,
                workdir,
            )

        else:

            precision = []
            precision_scans = 0

        timeline = merge_timelines(
            sparse,
            precision,
        )

        markers, corrections = build_live365(
            timeline,
            show_title,
        )

        csv_path = (
            Path(workdir)
            / "live365_cue_sheet.csv"
        )

        write_csv(
            markers,
            csv_path,
        )

        JOBS[job_id].update(
            {
                "status": "complete",
                "message": "Complete.",
                "source_markers": len(
                    timeline
                ),
                "final_markers": len(
                    markers
                ),
                "corrections": corrections,
                "warnings": len(
                    ambiguous
                ),
                "sparse_scans": sparse_scans,
                "sparse_candidates": sparse_candidates,
                "precision_windows": len(
                    windows
                ),
                "precision_scans": precision_scans,
                "download": (
                    f"/download/{job_id}"
                ),
                "timeline": [
                    {
                        "time": format_time(
                            x["start"]
                        ),
                        "artist": x["artist"],
                        "title": x["title"],
                        "album": (
                            x.get("album")
                            or f"{x['title']} (Single)"
                        ),
                        "source": x["source"],
                    }
                    for x in timeline
                ],
            }
        )

    except Exception as exc:

        JOBS[job_id].update(
            {
                "status": "error",
                "message": str(exc),
            }
        )


@app.get("/")
def index():
    return render_template(
        "index.html"
    )


@app.post("/analyze")
def analyze():

    upload = request.files.get(
        "audio"
    )

    show_title = (
        request.form.get(
            "show_title"
        )
        or ""
    ).strip()

    if not upload or not upload.filename:

        return jsonify(
            {
                "error": (
                    "Please select the full "
                    "radio show audio file."
                )
            }
        ), 400

    if not show_title:

        return jsonify(
            {
                "error": (
                    "Please enter the show title."
                )
            }
        ), 400

    job_id = uuid.uuid4().hex

    workdir = (
        Path(tempfile.gettempdir())
        / f"block105_{job_id}"
    )

    workdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ext = (
        Path(upload.filename)
        .suffix
        .lower()
        or ".mp3"
    )

    audio_path = (
        workdir
        / f"show{ext}"
    )

    upload.save(
        audio_path
    )

    JOBS[job_id] = {
        "status": "queued",
        "message": "Queued…",
    }

    threading.Thread(
        target=process_job,
        args=(
            job_id,
            str(audio_path),
            show_title,
            str(workdir),
        ),
        daemon=True,
    ).start()

    return jsonify(
        {
            "job_id": job_id
        }
    )


@app.get("/status/<job_id>")
def status(job_id):

    job = JOBS.get(job_id)

    if not job:

        return jsonify(
            {
                "error": "Job not found."
            }
        ), 404

    return jsonify(job)


@app.get("/download/<job_id>")
def download(job_id):

    job = JOBS.get(job_id)

    if (
        not job
        or job.get("status")
        != "complete"
    ):

        return jsonify(
            {
                "error": (
                    "The CSV is not ready."
                )
            }
        ), 404

    path = (
        Path(tempfile.gettempdir())
        / f"block105_{job_id}"
        / "live365_cue_sheet.csv"
    )

    if not path.exists():

        return jsonify(
            {
                "error": (
                    "CSV file not found."
                )
            }
        ), 404

    return send_file(
        path,
        as_attachment=True,
        download_name=(
            "BLOCK105_Live365_Cue_Sheet.csv"
        ),
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                8080,
            )
        ),
        debug=False,
    )
