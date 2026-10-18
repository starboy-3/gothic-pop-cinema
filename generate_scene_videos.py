import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
from urllib.parse import urlsplit

from bfl_video_api import API, api_request, download, save_json


ROOT = Path(__file__).resolve().parent
FILM = ROOT / 'films/pop-culture-city'
MAX_WORKERS = 3
POLL_TIMEOUT = 30 * 60
TERMINAL = {'error', 'failed', 'failure', 'cancelled', 'canceled', 'request moderated', 'content moderated'}


def confined_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Expected a relative workspace path')
    candidate = root / path
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes film workspace')
    for component in [candidate, *candidate.parents]:
        if component == root:
            break
        if component.is_symlink():
            raise ValueError('Symlink artifact paths are not supported')
    return candidate


def read_selection():
    selection = json.loads(confined_path(FILM, 'video-selection.json').read_text(encoding='utf-8'))
    if not isinstance(selection, dict) or not isinstance(selection.get('shots'), list) or not selection['shots']:
        raise ValueError('Selection must contain a nonempty shots list')
    identifiers = set()
    for shot in selection['shots']:
        if not isinstance(shot, dict) or not all(name in shot for name in ('id', 'frame', 'motion_prompt', 'duration')):
            raise ValueError('Incomplete shot entry')
        identifier = shot['id']
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', identifier) or identifier in identifiers:
            raise ValueError('Invalid or duplicate shot ID')
        identifiers.add(identifier)
        if not isinstance(shot['frame'], str) or not isinstance(shot['motion_prompt'], str) or not shot['motion_prompt'].strip():
            raise ValueError('Frame and motion prompt must be nonempty strings')
        if type(shot['duration']) is not int or shot['duration'] != 10:
            raise ValueError('Scene drafts must be 10 seconds')
        frame_names = [shot['frame']]
        if 'end_frame' in shot:
            if not isinstance(shot['end_frame'], str) or not shot['end_frame'].strip():
                raise ValueError('End frame must be a nonempty relative PNG path')
            frame_names.append(shot['end_frame'])
        if 'castle' in ' '.join([identifier, shot['motion_prompt'], *frame_names]).casefold():
            raise ValueError('Castle is blocked; no submission or retry is allowed')
        for name in frame_names:
            frame = confined_path(FILM, name)
            if frame.suffix.lower() != '.png':
                raise ValueError('Expected a PNG frame')
        confined_path(FILM, f'video/{identifier}')
    return selection['shots']


def prepare_frame(shot, field, role):
    frame = confined_path(FILM, shot[field])
    image = frame.read_bytes()
    if len(image) < 33 or image[:8] != b'\x89PNG\r\n\x1a\n' or image[12:16] != b'IHDR':
        raise ValueError('Invalid PNG frame header')
    width, height = struct.unpack('>II', image[16:24])
    if not width or not height:
        raise ValueError('Invalid PNG dimensions')
    image_hash = hashlib.sha256(image).hexdigest()
    generation_path = confined_path(FILM, f'generation/{frame.stem}.json')
    scale_verified = False
    if generation_path.exists():
        generation = json.loads(generation_path.read_text(encoding='utf-8'))
        scales = [generation[name] for name in ('scale', 'adapter_scale', 'lora_scale') if name in generation]
        if any(isinstance(scale, bool) or float(scale) != 0.75 for scale in scales):
            raise ValueError('Source generation adapter scale must be 0.75')
        if generation.get('sha256') and generation['sha256'] != image_hash:
            raise ValueError('Source image does not match generation metadata hash')
        scale_verified = bool(scales)
    print(f"{shot['id']}: {role} source scale 0.75 {'verified' if scale_verified else 'unverified (metadata unavailable)'}", flush=True)
    return base64.b64encode(image).decode('ascii'), {
        'role': role,
        'path': shot[field],
        'sha256': image_hash,
        'dimensions': [width, height],
        'checkpoint': 1500,
        'adapter_scale': 0.75,
        'adapter_scale_verified': scale_verified,
        'source_generation': str(generation_path.relative_to(FILM)) if generation_path.exists() else None,
    }


def prepare(shot, output):
    start_data, start_metadata = prepare_frame(shot, 'frame', 'start')
    encoded = [start_data]
    keyframe_metadata = [start_metadata]
    if 'end_frame' in shot:
        end_data, end_metadata = prepare_frame(shot, 'end_frame', 'end')
        encoded.append(end_data)
        keyframe_metadata.append(end_metadata)
    payload = {
        'mode': 'i2v',
        'prompt': shot['motion_prompt'],
        'keyframes': encoded if len(encoded) == 2 else start_data,
        'duration': shot['duration'],
        'aspect_ratio': '16:9',
        'resolution': 'hd',
        'draft': True,
        'generate_audio': False,
        'safety_tolerance': 2,
    }
    metadata = {
        **payload,
        'id': shot['id'],
        'keyframes': [entry['path'] for entry in keyframe_metadata] if len(encoded) == 2 else shot['frame'],
        'keyframe_sha256': start_metadata['sha256'],
        'keyframe_dimensions': start_metadata['dimensions'],
        'keyframe_checkpoint': 1500,
        'keyframe_adapter_scale': 0.75,
        'keyframe_adapter_scale_verified': start_metadata['adapter_scale_verified'],
        'source_generation': start_metadata['source_generation'],
        'keyframe_details': keyframe_metadata,
        'endpoint': API + '/flux-3-video',
    }
    if 'end_frame' in shot:
        metadata.update(end_frame=shot['end_frame'], end_keyframe_sha256=end_metadata['sha256'])
    if 'target_seconds' in shot:
        metadata['target_seconds'] = shot['target_seconds']
    save_json(confined_path(output, 'request.json'), metadata)
    return payload


def terminal(status):
    normalized = str(status).strip().casefold()
    return normalized in TERMINAL or 'moderat' in normalized or re.sub(r'[^a-z]', '', normalized) == 'protectedcontent'


def polling_url(submission):
    url = submission.get('polling_url')
    if not isinstance(url, str):
        raise ValueError('Accepted job has no polling URL; no repeat POST allowed')
    address = urlsplit(url)
    host = address.hostname or ''
    if address.scheme != 'https' or address.username or address.password or not (host == 'api.bfl.ai' or host.endswith('.bfl.ai')):
        raise ValueError('Untrusted polling URL')
    return url


def request(url, key, payload=None):
    result = api_request(url, key, payload)
    return json.loads(json.dumps(result).replace(key, '[REDACTED]'))


def finish(output, state, result):
    artifact = result['result']
    video = confined_path(output, 'draft.mp4')
    download(artifact['sample'], video)
    cache = artifact.get('draft_cache') or result.get('draft_cache')
    if cache is not None:
        save_json(confined_path(output, 'draft-cache.json'), {'draft_cache': cache})
        if isinstance(cache, str) and cache.startswith('https://'):
            download(cache, confined_path(output, 'draft-cache.bin'))
    state.update(status='downloaded', completed_at=time.time())
    save_json(confined_path(output, 'job.json'), state)


def run_shot(shot, submit, key):
    output = confined_path(FILM, f"video/{shot['id']}")
    output.mkdir(parents=True, exist_ok=True)
    with confined_path(output, '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = confined_path(output, 'job.json')
        result_path = confined_path(output, 'result.json')
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        if state is not None and (not isinstance(state, dict) or not state):
            raise ValueError('Invalid existing job state; no repeat POST allowed')
        if (state and terminal(state.get('status'))) or (result and terminal(result.get('status'))):
            raise RuntimeError('Terminal job; no retry allowed')
        if state is None and result is not None:
            raise RuntimeError('Orphan result exists; no repeat POST allowed')
        video = confined_path(output, 'draft.mp4')
        if state and state.get('status') == 'downloaded' and video.is_file() and video.stat().st_size:
            return 'already downloaded'
        if state:
            poll = polling_url(state.get('submission', {}))
            if not submit:
                return 'existing job retained; use --submit to resume'
            print(f"{shot['id']}: resuming job {state['submission'].get('id', 'unknown')} — {state.get('status')}", flush=True)
        else:
            payload = prepare(shot, output)
            if not submit:
                return 'prepared; no network request'
            state = {'status': 'submitting', 'started_at': time.time()}
            save_json(state_path, state)
            print(f"{shot['id']}: submitting", flush=True)
            submission = request(API + '/flux-3-video', key, payload)
            state.update(status='submitted', submission=submission)
            save_json(state_path, state)
            print(f"{shot['id']}: job {submission.get('id', 'unknown')} — {submission.get('status', 'submitted')}", flush=True)
            if terminal(submission.get('status')):
                save_json(result_path, submission)
                state['status'] = submission['status']
                save_json(state_path, state)
                raise RuntimeError('Terminal submission; no retry allowed')
            poll = polling_url(submission)
        if result and result.get('status') == 'Ready':
            finish(output, state, result)
            return 'downloaded existing result'
        deadline = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < deadline:
            result = request(poll, key)
            save_json(result_path, result)
            status = result.get('status')
            if status != state.get('status'):
                print(f"{shot['id']}: job {state['submission'].get('id', 'unknown')} — {status}", flush=True)
            state['status'] = status
            save_json(state_path, state)
            if terminal(status):
                raise RuntimeError('Terminal job; no retry allowed')
            if status == 'Ready':
                finish(output, state, result)
                return 'downloaded'
            time.sleep(min(5, max(0, deadline - time.monotonic())))
        return 'polling timed out; use --submit to resume accepted job'


def process_shot(shot, submit, key):
    try:
        message = run_shot(shot, submit, key)
        success = 'timed out' not in message
    except BlockingIOError:
        message, success = 'locked by another worker; skipped', False
    except Exception as error:
        message, success = f'stopped ({type(error).__name__}); inspect local state; no automatic resubmission', False
    print(f"{shot['id']}: {message}", flush=True)
    return success


def main(argv=None):
    global FILM
    parser = argparse.ArgumentParser(description='Prepare or submit parallel silent HD scene drafts; castle is blocked.')
    parser.add_argument('--film', default='pop-culture-city', help='Film workspace slug (default: pop-culture-city)')
    parser.add_argument('--shots', nargs='+', help='Process only these IDs from video-selection.json')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true', help='Prepare metadata only (default)')
    mode.add_argument('--submit', action='store_true', help='Submit new shots once, or resume accepted jobs')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.film):
        parser.error('--film must be a simple workspace slug')
    os.umask(0o077)
    key = None
    try:
        FILM = confined_path(ROOT, f'films/{args.film}')
        shots = read_selection()
        if args.shots:
            selected = set(args.shots)
            if len(selected) != len(args.shots):
                parser.error('--shots must not contain duplicate IDs')
            unknown = selected - {shot['id'] for shot in shots}
            if unknown:
                parser.error('Unknown shot IDs: ' + ', '.join(sorted(unknown)))
            shots = [shot for shot in shots if shot['id'] in selected]
        if args.submit:
            if not sys.stdin.isatty():
                raise RuntimeError('Run interactively for hidden BFL_API_KEY entry')
            key = getpass.getpass('BFL_API_KEY (hidden; never saved): ').strip()
            if not key:
                raise ValueError('Missing BFL_API_KEY')
            credits = request(API + '/credits', key)
            print('Credit check:', json.dumps(credits), flush=True)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(lambda shot: process_shot(shot, args.submit, key), shots))
        return 0 if all(results) else 1
    except Exception as error:
        print(f'Batch stopped ({type(error).__name__}); check selection, paths, and interactive key entry.', file=sys.stderr)
        return 1
    finally:
        key = None


if __name__ == '__main__':
    raise SystemExit(main())
