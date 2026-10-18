import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


FILM = Path(__file__).resolve().parent / 'films/pop-culture-recreation'
EXPECTED = [
    ('01-shrek', 7),
    ('02-joker', 8),
    ('03-squid-game', 8),
    ('04-ego-adult', 6),
    ('05-ego-childhood', 6),
    ('06-simba-rooftop', 8),
    ('07-joi', 9),
]
FPS = 24


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed ({result.returncode}):\n{result.stderr[-6000:]}')
    return result.stdout


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path):
    data = json.loads(run([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format',
        '-of', 'json', str(path),
    ]))
    streams = data.get('streams', [])
    video = next((stream for stream in streams if stream.get('codec_type') == 'video'
                  and not stream.get('disposition', {}).get('attached_pic')), None)
    if video is None:
        raise ValueError(f'No video stream: {path}')
    duration = None
    for candidate in [video.get('duration'), data.get('format', {}).get('duration')]:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            duration = value
            break
    width, height = int(video.get('width', 0)), int(video.get('height', 0))
    if duration is None or width <= 0 or height <= 0:
        raise ValueError(f'Invalid duration or dimensions: {path}')
    return {
        'duration_seconds': duration,
        'width': width,
        'height': height,
        'codec': video.get('codec_name'),
        'pixel_format': video.get('pix_fmt'),
        'frame_rate': video.get('avg_frame_rate'),
        'sample_aspect_ratio': video.get('sample_aspect_ratio'),
        'video_stream_index': video['index'],
        'audio_streams': sum(stream.get('codec_type') == 'audio' for stream in streams),
    }


def verify_normalized(info, label):
    if (info['width'], info['height'], info['codec'], info['pixel_format']) != (
        1280, 720, 'h264', 'yuv420p'
    ) or info['audio_streams']:
        raise RuntimeError(f'Unexpected output format: {label}: {info}')
    numerator, denominator = info['frame_rate'].split('/')
    if float(denominator) == 0 or abs(float(numerator) / float(denominator) - FPS) > 0.001:
        raise RuntimeError(f'Unexpected frame rate: {label}: {info["frame_rate"]}')


def main():
    parser = argparse.ArgumentParser(description='Assemble seven rendered clips as a silent 720p draft; never substitute stills.')
    parser.add_argument('--allow-partial', action='store_true',
                        help='Explicit preview-only mode: skip missing clips and use a partial output filename.')
    args = parser.parse_args()
    selection_path = FILM / 'video-selection.json'
    selection_hash = sha256(selection_path)
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    shots = selection.get('shots', [])
    if [shot.get('id') for shot in shots] != [shot_id for shot_id, _ in EXPECTED]:
        raise ValueError('Selection must contain exactly the seven expected shot IDs in their original order.')
    for shot, (shot_id, target) in zip(shots, EXPECTED):
        if shot.get('target_seconds') != target:
            raise ValueError(f'{shot_id}: target_seconds must be {target}.')
    missing = []
    available = []
    for shot in shots:
        path = FILM / 'video' / shot['id'] / 'draft.mp4'
        if not path.is_file():
            missing.append(str(path.relative_to(FILM)))
        else:
            available.append((shot, path))
    if missing and not args.allow_partial:
        raise ValueError('All seven rendered videos are required. Missing:\n  ' + '\n  '.join(missing)
                         + '\nWait for rendering, or explicitly use --allow-partial for a preview only.')
    if not available:
        raise ValueError('No draft.mp4 clips available; at least one rendered video is required.')
    for executable in ['ffmpeg', 'ffprobe']:
        if not shutil.which(executable):
            raise ValueError(f'Required executable not found on PATH: {executable}')
    records = []
    warnings = []
    for shot, path in available:
        info = probe(path)
        target = shot['target_seconds']
        frame_count = math.floor(min(target, info['duration_seconds']) * FPS + 1e-7)
        if frame_count < 1:
            raise ValueError(f'Clip is shorter than one output frame: {path}')
        short = info['duration_seconds'] < target
        if short:
            warnings.append(f'{shot["id"]}: only {info["duration_seconds"]:.3f}s available; target {target}s.')
        records.append({
            'id': shot['id'], 'source': str(path.relative_to(FILM)),
            'source_sha256': sha256(path), 'source_probe': info,
            'target_seconds': target, 'trim_start_seconds': 0,
            'planned_seconds': frame_count / FPS, 'planned_frames': frame_count,
            'shorter_than_target': short,
        })
    for warning in warnings:
        print(f'WARNING: {warning}', file=sys.stderr)
    stem = 'pop-culture-draft-partial-preview' if args.allow_partial else 'pop-culture-draft'
    output = FILM / f'{stem}.mp4'
    assembly_path = FILM / f'{stem}.assembly.json'
    with tempfile.TemporaryDirectory(prefix='.pop-assembly-', dir=FILM) as temporary:
        staging = Path(temporary)
        normalized = []
        for index, record in enumerate(records):
            source = FILM / record['source']
            clip = staging / f'clip-{index:02d}.mp4'
            filters = (
                'setpts=PTS-STARTPTS,'
                'scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2:reset_sar=1,'
                'pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,'
                f'fps={FPS},trim=end_frame={record["planned_frames"]},setpts=PTS-STARTPTS'
            )
            print(f'Normalizing {record["id"]}: {record["planned_seconds"]:.3f}s', flush=True)
            run([
                'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                '-i', str(source), '-map', f'0:{record["source_probe"]["video_stream_index"]}',
                '-vf', filters, '-frames:v', str(record['planned_frames']),
                '-an', '-sn', '-dn', '-map_metadata', '-1', '-map_chapters', '-1',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                '-pix_fmt', 'yuv420p', '-r', str(FPS), '-video_track_timescale', '24000',
                '-movflags', '+faststart', str(clip),
            ])
            if sha256(source) != record['source_sha256']:
                raise RuntimeError(f'Source changed during assembly: {source}; retry when downloads finish.')
            actual = probe(clip)
            verify_normalized(actual, record['id'])
            if abs(actual['duration_seconds'] - record['planned_seconds']) > 1 / FPS + 0.001:
                raise RuntimeError(f'Unexpected normalized duration for {record["id"]}: {actual}')
            record['assembled_seconds'] = actual['duration_seconds']
            record['normalized_probe'] = actual
            normalized.append(clip)
        concat = staging / 'concat.txt'
        concat.write_text(''.join(f"file '{clip.name}'\n" for clip in normalized), encoding='utf-8')
        staged_output = staging / output.name
        run([
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
            '-f', 'concat', '-safe', '1', '-i', str(concat), '-map', '0:v:0',
            '-c:v', 'copy', '-an', '-map_metadata', '-1', '-map_chapters', '-1',
            '-movflags', '+faststart', str(staged_output),
        ])
        final_probe = probe(staged_output)
        verify_normalized(final_probe, output.name)
        expected_duration = sum(record['assembled_seconds'] for record in records)
        if abs(final_probe['duration_seconds'] - expected_duration) > 1 / FPS + 0.001:
            raise RuntimeError('Concatenated duration does not match the sum of assembled clips.')
        if sha256(selection_path) != selection_hash:
            raise RuntimeError('Selection changed during assembly; retry with a stable selection.')
        assembly = {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'selection': 'video-selection.json', 'selection_sha256': selection_hash,
            'output': output.name, 'output_sha256': sha256(staged_output),
            'quality_label': '720p HD draft; not Full HD',
            'partial_preview': args.allow_partial, 'all_seven_present': len(records) == 7,
            'expected_shots': 7, 'assembled_shots': len(records),
            'missing_sources': missing, 'warnings': warnings,
            'requested_total_seconds': sum(target for _, target in EXPECTED),
            'output_probe': final_probe, 'silent': True,
            'transitions': 'hard cuts', 'sources': records,
        }
        staged_metadata = staging / assembly_path.name
        staged_metadata.write_text(json.dumps(assembly, indent=2) + '\n', encoding='utf-8')
        staged_output.replace(output)
        staged_metadata.replace(assembly_path)
    print(f'Saved {output} ({final_probe["duration_seconds"]:.3f}s, silent 1280x720 at 24 fps)')
    print(f'Assembly provenance: {assembly_path}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
