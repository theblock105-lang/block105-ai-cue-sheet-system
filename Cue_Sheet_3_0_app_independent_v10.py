import csv
import json
import os
import smtplib
import ssl
import re
import statistics
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from email.message import EmailMessage

import requests
from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)
# Allow 2-hour shows plus a generous upload buffer, including high-bitrate MP3 files.
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

AUDD_URL = 'https://enterprise.audd.io/'
JOBS = {}
LOCK = threading.Lock()


def parse_seconds(value):
    if value is None:
        return None
    value = str(value).strip()
    parts = value.split(':')
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
    return f'{h:02d}:{m:02d}:{s:02d}.{ms:03d}'


def normalize(value):
    return re.sub(r'\s+', ' ', str(value or '').casefold()).strip()


def format_show_title(value):
    """Format user-entered show names without altering recognized music metadata."""
    text = re.sub(r'\s+', ' ', str(value or '')).strip()
    if not text:
        return text
    text = text.title()
    text = re.sub(r'\bDj\b', 'DJ', text)
    return text


def song_key(artist, title):
    return normalize(artist), normalize(title)


def ffmpeg_extract(source, start, duration, output):
    cmd = [
        'ffmpeg', '-y', '-ss', str(max(0, start)), '-i', str(source),
        '-t', str(duration), '-vn', '-ac', '1', '-ar', '44100',
        '-codec:a', 'libmp3lame', '-q:a', '4', str(output)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def audd_scan(audio_file, token):
    data = {
        'api_token': token,
        'accurate_offsets': 'true',
        'skip': '4',
        'every': '1',
    }
    with open(audio_file, 'rb') as f:
        response = requests.post(AUDD_URL, data=data, files={'file': f}, timeout=1200)
    response.raise_for_status()
    payload = response.json()
    if payload.get('status') == 'error':
        raise RuntimeError(str(payload.get('error') or payload))
    return payload


def collect_candidates(payload, base_offset=0):
    songs = {}
    scans = 0
    candidates = 0
    for scan in payload.get('result', []):
        scans += 1
        scan_offset = parse_seconds(scan.get('offset'))
        if scan_offset is None:
            continue
        for candidate in scan.get('songs', []):
            if not isinstance(candidate, dict):
                continue
            artist = str(candidate.get('artist') or '').strip()
            title = str(candidate.get('title') or '').strip()
            if not artist or not title:
                continue
            timecode = parse_seconds(candidate.get('timecode'))
            if timecode is None:
                continue
            try:
                start_offset = float(candidate.get('start_offset', 0)) / 1000.0
            except (ValueError, TypeError):
                start_offset = 0
            estimated_start = base_offset + scan_offset + start_offset - timecode
            key = song_key(artist, title)
            songs.setdefault(key, {
                'artist': artist,
                'title': title,
                'album': str(candidate.get('album') or '').strip(),
                'year': str(candidate.get('release_date') or '')[:4],
                'estimates': [],
            })['estimates'].append(estimated_start)
            candidates += 1
    return songs, scans, candidates


def best_cluster(values, window):
    if not values:
        return []
    values = sorted(values)
    best = []
    for i, start in enumerate(values):
        cluster = [v for v in values[i:] if v - start <= window]
        if len(cluster) > len(best):
            best = cluster
    return best


def build_sparse_timeline(songs):
    out = []
    for song in songs.values():
        cluster = best_cluster(song['estimates'], 25.0)
        if len(cluster) < 2:
            continue
        out.append({
            'artist': song['artist'], 'title': song['title'],
            'album': song['album'], 'year': song['year'],
            'start': statistics.median(cluster),
            'detections': len(cluster), 'spread': max(cluster) - min(cluster),
            'source': 'SPARSE'
        })
    return sorted(out, key=lambda x: x['start'])


def find_ambiguous_times(payload):
    times = []
    for scan in payload.get('result', []):
        keys = set()
        for song in scan.get('songs', []):
            artist = str(song.get('artist') or '').strip()
            title = str(song.get('title') or '').strip()
            if artist and title:
                keys.add(song_key(artist, title))
        if len(keys) > 1:
            t = parse_seconds(scan.get('offset'))
            if t is not None:
                times.append(t)
    return sorted(set(times))


def build_precision_windows(times):
    windows = []
    for center in times:
        start = max(0, center - 60)
        end = center + 90
        if windows and start <= windows[-1]['end']:
            windows[-1]['end'] = max(windows[-1]['end'], end)
        else:
            windows.append({'start': start, 'end': end})
    return windows


def precision_scan(audio, windows, token, workdir):
    merged = {}
    total_scans = 0
    for i, window in enumerate(windows, 1):
        out = Path(workdir) / f'precision_{i}.mp3'
        ffmpeg_extract(audio, window['start'], window['end'] - window['start'], out)
        data = {
            'api_token': token, 'accurate_offsets': 'true', 'skip': '0', 'every': '1'
        }
        with open(out, 'rb') as f:
            response = requests.post(AUDD_URL, data=data, files={'file': f}, timeout=1200)
        response.raise_for_status()
        payload = response.json()
        if payload.get('status') == 'error':
            continue
        for scan in payload.get('result', []):
            total_scans += 1
            scan_offset = parse_seconds(scan.get('offset'))
            if scan_offset is None:
                continue
            for candidate in scan.get('songs', []):
                if not isinstance(candidate, dict):
                    continue
                artist = str(candidate.get('artist') or '').strip()
                title = str(candidate.get('title') or '').strip()
                if not artist or not title:
                    continue
                tc = parse_seconds(candidate.get('timecode'))
                if tc is None:
                    continue
                try:
                    so = float(candidate.get('start_offset', 0)) / 1000
                except (ValueError, TypeError):
                    so = 0
                start = window['start'] + scan_offset + so - tc
                key = song_key(artist, title)
                merged.setdefault(key, {
                    'artist': artist, 'title': title,
                    'album': str(candidate.get('album') or '').strip(),
                    'year': str(candidate.get('release_date') or '')[:4],
                    'estimates': []
                })['estimates'].append(start)
    overrides = []
    for song in merged.values():
        cluster = best_cluster(song['estimates'], 8.0)
        if len(cluster) < 3:
            continue
        overrides.append({
            'artist': song['artist'], 'title': song['title'],
            'album': song['album'], 'year': song['year'],
            'start': statistics.median(cluster), 'detections': len(cluster),
            'spread': max(cluster) - min(cluster), 'source': 'PRECISION'
        })
    return sorted(overrides, key=lambda x: x['start']), total_scans


def merge_timelines(sparse, precision):
    combined = []
    used = set()
    for item in sparse:
        matches = [
            (i, p) for i, p in enumerate(precision)
            if i not in used and song_key(item['artist'], item['title']) == song_key(p['artist'], p['title'])
        ]
        if matches:
            i, p = min(matches, key=lambda x: abs(item['start'] - x[1]['start']))
            used.add(i)
            combined.append(p.copy())
        else:
            combined.append(item.copy())
    for i, p in enumerate(precision):
        if i not in used:
            combined.append(p.copy())
    combined.sort(key=lambda x: x['start'])
    final = []
    for item in combined:
        if final and song_key(item['artist'], item['title']) == song_key(final[-1]['artist'], final[-1]['title']) and abs(item['start'] - final[-1]['start']) <= 20:
            if item['source'] == 'PRECISION':
                final[-1] = item
        else:
            final.append(item)
    return final


def get_audio_duration(audio_path):
    """Return audio duration in seconds using ffprobe, or None if unavailable."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(audio_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def marker_ms(marker):
    h, m, s = marker['offset'].split(':')
    sec, milli = s.split('.')
    return int(h) * 3600000 + int(m) * 60000 + int(sec) * 1000 + int(milli)


def marker_density_violations(markers):
    limits = [
        (600000, 5),
        (900000, 7),
        (1800000, 13),
        (4680000, 33),
    ]
    timestamps = [marker_ms(m) for m in markers]
    violations = []
    for window, maximum in limits:
        for start in timestamps:
            count = sum(1 for t in timestamps if start <= t < start + window)
            if count > maximum:
                violations.append((start, window, maximum, count))
                break
    return violations


def remove_near_touching_markers(markers, corrections, min_gap_ms=2000):
    """Remove recognition collisions that occur less than 1 second apart.

    The protected opening marker always wins. For other collisions, preserve the
    first marker and convert the removed collision into a correction rather than
    allowing stacked/near-stacked Live365 markers.
    """
    if not markers:
        return markers, corrections
    markers = sorted(markers, key=marker_ms)
    result = [markers[0]]
    for marker in markers[1:]:
        if marker_ms(marker) - marker_ms(result[-1]) < min_gap_ms:
            corrections += 1
            continue
        result.append(marker)
    return result, corrections


def normalize_final_markers(markers, show_title, duration_seconds, corrections):
    """Final QC pass before CSV creation.

    Final order: normalize -> deduplicate -> classify metadata -> remove near
    collisions -> enforce density -> restore the five-marker minimum with
    safely spaced TALK markers -> verify again.
    """
    show_title = (show_title or 'BLOCK 105 SHOW').strip()
    cutoff_ms = 2 * 60 * 60 * 1000
    cleaned = []
    for marker in markers:
        t = marker_ms(marker)
        if t > cutoff_ms:
            corrections += 1
            continue
        marker['offset'] = format_time(t / 1000)
        cleaned.append(marker)
    cleaned.sort(key=marker_ms)

    protected = {
        'offset': '00:00:00.000', 'media_type': 'talk',
        'title': show_title, 'artist': show_title, 'album': '', 'year': '',
        '_protected': True,
    }
    deduped = [protected]
    seen = {0}
    for marker in cleaned:
        t = marker_ms(marker)
        if t == 0 or t in seen:
            corrections += 1
            continue
        seen.add(t)
        marker['title'] = str(marker.get('title') or '').strip()
        marker['artist'] = str(marker.get('artist') or '').strip()
        marker['album'] = str(marker.get('album') or '').strip()
        marker['year'] = str(marker.get('year') or '').strip()
        is_independent = (
            marker.get('media_type') == 'independent'
            or marker['artist'].casefold() == 'independent artist'
        )
        if marker.get('media_type') == 'talk':
            marker.update({'title': show_title, 'artist': show_title, 'album': '', 'year': ''})
        elif is_independent:
            marker.update({'media_type': 'talk', 'title': show_title,
                           'artist': 'Independent Artist', 'album': 'Non-Copy-Right', 'year': ''})
        else:
            if not marker['title']:
                corrections += 1
                continue
            if not marker['artist']:
                marker['artist'] = 'Independent Artist'
                marker['media_type'] = 'talk'
                marker['title'] = show_title
                marker['album'] = 'Non-Copy-Right'
            elif not marker['album']:
                marker['album'] = f"{marker['title']} (Single)"
        deduped.append(marker)

    # Remove near-touching collisions before density calculations.
    deduped, corrections = remove_near_touching_markers(deduped, corrections, 1000)
    deduped.sort(key=marker_ms)

    # Final density gate: remove the non-protected marker contributing to the
    # most violations until the timeline is compliant. Never rename a marker
    # to TALK as a substitute for removing it; that would still count as a marker.
    while True:
        violations = marker_density_violations(deduped)
        if not violations:
            break
        candidates = [m for m in deduped if not m.get('_protected')]
        if not candidates:
            break
        best = None
        best_score = -1
        for candidate in candidates:
            ct = marker_ms(candidate)
            score = 0
            for start, window, maximum, count in violations:
                if start <= ct < start + window:
                    score += count - maximum
            if score > best_score:
                best_score = score
                best = candidate
        deduped.remove(best)
        corrections += 1

    # Restore the five-marker minimum using evenly spaced TALK markers that
    # are individually tested against the density gate.
    target = 5
    duration = min(float(duration_seconds or 0), 7200.0)
    if duration <= 0:
        duration = 7200.0
    preferred = [duration * i / target for i in range(1, target)]
    fallback = [duration * i / 100.0 for i in range(1, 100)]
    for t in preferred + fallback:
        if len(deduped) >= target:
            break
        ms = int(round(t * 1000))
        if ms <= 0 or ms >= int(duration * 1000) or ms in {marker_ms(x) for x in deduped}:
            continue
        candidate = {'offset': format_time(t), 'media_type': 'talk',
                     'title': show_title, 'artist': show_title, 'album': '', 'year': ''}
        trial = deduped + [candidate]
        if not marker_density_violations(trial):
            deduped.append(candidate)
            corrections += 1

    deduped.sort(key=marker_ms)

    # Absolute final gate. If adding the minimum created a violation, remove
    # the least-important non-protected marker and then try the minimum pass again.
    while marker_density_violations(deduped):
        candidates = [m for m in deduped if not m.get('_protected')]
        if not candidates:
            break
        deduped.remove(candidates[-1])
        corrections += 1

    return deduped, corrections


def classify_unknown_candidate(item, show_title):
    """Conservative 3.0 classification hook.

    We only promote an item to Independent Artist when an upstream stage has
    explicitly marked it independent. We intentionally do not guess from
    speech/music energy alone; real broadcaster audio showed that approach can
    misclassify programs.
    """
    media_type = str(item.get('media_type') or '').strip().casefold()
    artist = str(item.get('artist') or '').strip()
    explicit = media_type == 'independent' or artist.casefold() == 'independent artist'
    if explicit:
        return {
            **item,
            'media_type': 'independent',
            'title': show_title,
            'artist': 'Independent Artist',
            'album': 'Non-Copy-Right',
            'year': '',
        }
    return item



def build_live365(timeline, show_title, duration_seconds=None):
    markers = [{
        'offset': '00:00:00.000',
        'media_type': 'talk',
        'title': show_title,
        'artist': show_title,
        'album': '',
        'year': '',
        '_protected': True,
    }]

    for raw_item in timeline:
        item = classify_unknown_candidate(raw_item, show_title)
        start = float(item.get('start', 0) or 0)
        if start < 0 or start > 7200:
            continue
        ts = format_time(start)
        if ts == '00:00:00.000':
            ts = '00:00:00.001'
        media_type = str(item.get('media_type') or 'music').strip().casefold()
        title = str(item.get('title') or '').strip()
        artist = str(item.get('artist') or '').strip()
        album = str(item.get('album') or '').strip()
        if not title and media_type != 'independent':
            continue
        if media_type == 'independent' or artist.casefold() == 'independent artist':
            media_type = 'independent'
            title = show_title
            artist = 'Independent Artist'
            album = 'Non-Copy-Right'
        elif media_type == 'talk':
            title = show_title
            artist = show_title
            album = ''
        else:
            album = album or f"{title} (Single)"
        markers.append({
            'offset': ts,
            'media_type': media_type,
            'title': title,
            'artist': artist,
            'album': album,
            'year': '' if media_type in ('talk', 'independent') else str(item.get('year') or ''),
        })

    corrections = 0

    # Pass 1: normalize metadata, remove exact duplicates, and remove
    # near-touching recognition collisions BEFORE enforcing the marker-count
    # rules. This ordering matters: otherwise the collision cleanup can take
    # a previously valid five-marker sheet back below the minimum.
    markers, corrections = normalize_final_markers(
        markers, show_title, duration_seconds, corrections
    )
    markers, corrections = remove_near_touching_markers(markers, corrections, 2000)

    # Pass 2: enforce Live365 density limits. Prefer removing the marker that
    # contributes to the most violations; if tied, remove the newest unprotected
    # marker. This is safer than blindly deleting the last row.
    while True:
        violations = marker_density_violations(markers)
        if not violations:
            break
        if len(markers) <= 5:
            # Five markers is the hard floor, but five markers are not allowed
            # to violate Live365 density limits. Rebuild the non-protected
            # markers at safe positions instead of shipping a violating sheet.
            protected_marker = markers[0]
            duration = min(float(duration_seconds or 0), 7200.0)
            if duration <= 0:
                duration = 7200.0
            rebuilt = [protected_marker]
            for i in range(1, 5):
                t = duration * i / 5.0
                candidate = {
                    'offset': format_time(t),
                    'media_type': 'talk',
                    'title': show_title,
                    'artist': show_title,
                    'album': '',
                    'year': '',
                }
                if not marker_density_violations(rebuilt + [candidate]):
                    rebuilt.append(candidate)
            markers = rebuilt
            corrections += 1
            break

        violation_counts = [0] * len(markers)
        for start, window, maximum, count in violations:
            for i, marker in enumerate(markers):
                t = marker_ms(marker)
                if start <= t < start + window:
                    violation_counts[i] += 1

        candidates = [i for i in range(1, len(markers)) if violation_counts[i] > 0]
        if not candidates:
            break
        idx = max(candidates, key=lambda i: (violation_counts[i], marker_ms(markers[i])))
        markers.pop(idx)
        corrections += 1

    # Pass 3: if density cleanup reduced the sheet below five, rebuild the
    # missing TALK markers at safe, evenly distributed positions.
    if len(markers) < 5:
        markers, corrections = normalize_final_markers(
            markers, show_title, duration_seconds, corrections
        )
        markers, corrections = remove_near_touching_markers(markers, corrections, 2000)

    # Pass 4: hard final verification. Never export a sheet that still has
    # duplicate offsets, sub-second collisions, or Live365 density violations.
    markers.sort(key=marker_ms)
    if not markers or marker_ms(markers[0]) != 0:
        raise RuntimeError('Final QC failed: protected opening marker is missing.')
    offsets = [marker_ms(m) for m in markers]
    if len(offsets) != len(set(offsets)):
        raise RuntimeError('Final QC failed: duplicate marker offsets remain.')
    if any(b - a < 2000 for a, b in zip(offsets, offsets[1:])):
        raise RuntimeError('Final QC failed: markers remain less than 1 second apart.')
    final_violations = marker_density_violations(markers)
    if final_violations:
        raise RuntimeError(f'Final QC failed: Live365 density violations remain: {final_violations[:3]}')
    if duration_seconds and float(duration_seconds) > 4 and len(markers) < 5:
        raise RuntimeError('Final QC failed: fewer than five markers were produced.')

    # Final metadata contract: every exported row must have the exact fields
    # and semantics expected by the Live365 CSV writer. This is deliberately
    # checked after every other transformation so no later pass can corrupt it.
    for m in markers:
        if set(m.keys()) < {'offset', 'media_type', 'title', 'artist', 'album', 'year'}:
            raise RuntimeError('Final QC failed: incomplete marker metadata.')
        if not m['offset'] or not re.fullmatch(r'\d{2}:\d{2}:\d{2}\.\d{3}', m['offset']):
            raise RuntimeError('Final QC failed: invalid marker offset format.')
        if m['media_type'] == 'talk':
            is_independent_talk = (
                m['title'] == show_title
                and m['artist'] == 'Independent Artist'
                and m['album'] == 'Non-Copy-Right'
            )
            is_normal_talk = (
                m['title'] == show_title
                and m['artist'] == show_title
                and m['album'] == ''
            )
            if not (is_normal_talk or is_independent_talk):
                raise RuntimeError('Final QC failed: TALK metadata is inconsistent.')
        elif m['media_type'] == 'independent':
            raise RuntimeError('Final QC failed: Independent Artist marker was not normalized to TALK.')
        elif not m['title'] or not m['artist'] or not m['album']:
            raise RuntimeError('Final QC failed: recognized music marker has incomplete metadata.')

    return markers, corrections

def write_csv(markers, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['offset','media_type','title','artist','album','year'])
        for m in markers:
            w.writerow([m['offset'],m['media_type'],m['title'],m['artist'],m['album'],m['year']])


def send_cue_sheet_email(csv_path, show_title):
    sender = os.environ.get('BLOCK105_EMAIL_ADDRESS', 'theblock105@yahoo.com').strip()
    recipient = os.environ.get('BLOCK105_EMAIL_TO', 'theblock105@yahoo.com').strip()
    app_password = os.environ.get('BLOCK105_EMAIL_APP_PASSWORD', '').strip()
    if not app_password:
        raise RuntimeError('Yahoo email app password is not configured.')

    message = EmailMessage()
    message['From'] = sender
    message['To'] = recipient
    message['Subject'] = f'GENERATED CUE SHEET {show_title.upper()}'
    message.set_content(
        f'THE BLOCK 105 RADIO\n\n'
        f'Generated cue sheet for: {show_title}\n\n'
        f'The Live365 cue sheet is attached.'
    )
    with open(csv_path, 'rb') as f:
        message.add_attachment(
            f.read(),
            maintype='text',
            subtype='csv',
            filename='BLOCK105_Live365_Cue_Sheet.csv'
        )

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.mail.yahoo.com', 465, timeout=30, context=context) as smtp:
        smtp.login(sender, app_password)
        smtp.send_message(message)


def process_job(job_id, audio_path, show_title, workdir):
    try:
        token = os.environ.get('AUDD_API_TOKEN')
        if not token:
            raise RuntimeError('AUDD_API_TOKEN is not configured on the server.')
        JOBS[job_id]['status'] = 'running'
        JOBS[job_id]['message'] = 'Running full-audio identification…'
        sparse_payload = audd_scan(audio_path, token)
        sparse_songs, sparse_scans, sparse_candidates = collect_candidates(sparse_payload)
        sparse = build_sparse_timeline(sparse_songs)

        ambiguous = find_ambiguous_times(sparse_payload)
        windows = build_precision_windows(ambiguous)
        JOBS[job_id]['message'] = f'Precision-checking {len(windows)} ambiguous region(s)…'
        precision, precision_scans = precision_scan(audio_path, windows, token, workdir) if windows else ([], 0)

        timeline = merge_timelines(sparse, precision)
        duration_seconds = get_audio_duration(audio_path)
        markers, corrections = build_live365(timeline, show_title, duration_seconds)
        csv_path = Path(workdir) / 'live365_cue_sheet.csv'
        write_csv(markers, csv_path)

        JOBS[job_id].update({
            'status':'complete', 'message':'Complete.',
            'show_title': show_title,
            'email_status': 'not_sent',
            'source_markers': len(timeline), 'final_markers': len(markers),
            'corrections': corrections, 'warnings': len(ambiguous), 'generation_date': __import__('datetime').datetime.now().strftime('%m/%d/%y'),
            'sparse_scans': sparse_scans, 'precision_windows': len(windows),
            'precision_scans': precision_scans,
            'download': f'/download/{job_id}',
            'timeline': [
                {'time': format_time(x['start']), 'artist':x['artist'], 'title':x['title'], 'album':x.get('album') or f"{x['title']} (Single)", 'source':x['source']}
                for x in timeline
            ]
        })
    except Exception as exc:
        JOBS[job_id].update({'status':'error','message':str(exc)})


@app.get('/')
def index():
    return render_template('index.html')


@app.post('/analyze')
def analyze():
    upload = request.files.get('audio')
    show_title = format_show_title(request.form.get('show_title') or '')
    if not upload or not upload.filename:
        return jsonify({'error':'Please select the full radio show audio file.'}), 400
    if not show_title:
        return jsonify({'error':'Please enter the show title.'}), 400
    job_id = uuid.uuid4().hex
    workdir = Path(tempfile.gettempdir()) / f'block105_{job_id}'
    workdir.mkdir(parents=True, exist_ok=True)
    ext = Path(upload.filename).suffix.lower() or '.mp3'
    audio_path = workdir / f'show{ext}'
    upload.save(audio_path)
    JOBS[job_id] = {'status':'queued','message':'Queued…'}
    threading.Thread(target=process_job, args=(job_id,str(audio_path),show_title,str(workdir)), daemon=True).start()
    return jsonify({'job_id':job_id})


@app.get('/status/<job_id>')
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error':'Job not found.'}), 404
    return jsonify(job)


@app.get('/download/<job_id>')
def download(job_id):
    job = JOBS.get(job_id)
    if not job or job.get('status') != 'complete':
        return jsonify({'error':'The CSV is not ready.'}), 404
    path = Path(tempfile.gettempdir()) / f'block105_{job_id}' / 'live365_cue_sheet.csv'
    if not path.exists():
        return jsonify({'error':'CSV file not found.'}), 404
    return send_file(path, as_attachment=True, download_name='BLOCK105_Live365_Cue_Sheet.csv')


@app.post('/email/<job_id>')
def email_job(job_id):
    job = JOBS.get(job_id)
    if not job or job.get('status') != 'complete':
        return jsonify({'error': 'The CSV is not ready.'}), 404
    path = Path(tempfile.gettempdir()) / f'block105_{job_id}' / 'live365_cue_sheet.csv'
    if not path.exists():
        return jsonify({'error': 'CSV file not found.'}), 404
    show_title = job.get('show_title') or ''
    try:
        send_cue_sheet_email(path, show_title)
        job['email_status'] = 'sent'
        return jsonify({'status': 'sent', 'message': 'EMAIL SENT ✓'})
    except Exception as exc:
        job['email_status'] = 'failed'
        print(f'[EMAIL] Failed to send job {job_id}: {exc}', flush=True)
        return jsonify({
            'error': 'EMAIL SERVER DOWN',
            'detail': 'PLEASE DOWNLOAD AND SEND CUE SHEET MANUALLY.'
        }), 503


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)