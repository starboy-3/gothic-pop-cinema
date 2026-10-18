import argparse
import hashlib
import json
from pathlib import Path
import re


PUP_MODEL = 'black-forest-labs/flux-3-image-klein-pup'
PUP_REVISION = '3f3edb52b2c55de0d49feab28598b68f7f8e4830'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generate reference-linked film storyboards; never calls the video API.')
    parser.add_argument('--film', required=True)
    parser.add_argument('--phase', choices=['references', 'shots'], required=True)
    parser.add_argument('--operation', choices=['expand', 'review', 'render'], required=True)
    parser.add_argument('--frames', nargs='+', help='Process only these frame IDs within the selected phase')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', args.film):
        parser.error('Film must be a lowercase slug.')
    root = Path(__file__).resolve().parent
    workspace = root / 'films' / args.film
    config = json.loads((workspace / 'film.json').read_text())
    for directory in ['prompts', 'frames', 'generation']:
        (workspace / directory).mkdir(exist_ok=True)
    kind = 'reference' if args.phase == 'references' else 'shot'
    frames = [frame for frame in config['frames'] if frame['kind'] == kind]
    if args.frames:
        selected = set(args.frames)
        invalid = selected - {frame['id'] for frame in frames}
        if invalid:
            parser.error('Unknown or wrong-phase frame IDs: ' + ', '.join(sorted(invalid)))
        frames = [frame for frame in frames if frame['id'] in selected]
    if not frames:
        raise RuntimeError('No frames for this phase.')
    for frame in frames:
        for reference in frame['references']:
            if not (workspace / 'frames' / f'{reference}.png').is_file():
                raise RuntimeError(f'Missing reference {reference}; render references first.')
    if args.operation == 'review':
        for frame in frames:
            raw_path = workspace / 'prompts' / f"{frame['id']}.raw.txt"
            reviewed_path = workspace / 'prompts' / f"{frame['id']}.reviewed.txt"
            if reviewed_path.exists():
                raise RuntimeError(f'Preserving existing review: {reviewed_path}')
            raw = raw_path.read_text().strip()
            caption = re.sub(r'^Full-color (?:outdoor|indoor) photograph[, ]+', '', raw)
            caption = re.sub(r'^Full-color cinematic digital painting of ', '', caption)
            caption = re.sub(r'^Create a widescreen cinematic fantasy (?:image|illustration) of ', '', caption)
            caption = caption[0].upper() + caption[1:]
            reviewed_path.write_text(caption + ' ' + config['trigger'] + '\n')
            write_json(workspace / 'prompts' / f"{frame['id']}.review.json", {
                'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                'removed_prefix': raw[:len(raw) - len(caption)] if len(raw) != len(caption) else '',
                'trigger': config['trigger'], 'reviewed_by': 'operator_requested_review',
            })
            print('REVIEWED ' + frame['id'], flush=True)
        return

    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError('Run on the configured Nebius GPU.')
    if args.operation == 'expand':
        from huggingface_hub import hf_hub_download
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        for frame in frames:
            if (workspace / 'prompts' / f"{frame['id']}.raw.txt").exists():
                raise RuntimeError('Raw prompts already exist; preserve outputs instead of rerunning.')
        system_path = Path(hf_hub_download(PUP_MODEL, 'system_prompts.json', revision=PUP_REVISION))
        system_prompts = json.loads(system_path.read_text())
        processor = AutoProcessor.from_pretrained(PUP_MODEL, revision=PUP_REVISION)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            PUP_MODEL, revision=PUP_REVISION, dtype=torch.bfloat16, device_map='auto',
        ).eval()
        for frame in frames:
            references = [workspace / 'frames' / f'{reference}.png' for reference in frame['references']]
            mode = 'i2i' if references else 't2i'
            request = frame['request'] + '\n\nShared direction: ' + config['shared_direction']
            content = [{'type': 'image', 'image': str(path)} for path in references]
            if references:
                request = 'Reference images follow in order (the first image, the second image, ...).\nUser request: ' + request + '\nOutput the caption.'
            content.append({'type': 'text', 'text': request})
            messages = [{'role': 'system', 'content': [{'type': 'text', 'text': system_prompts[mode]}]},
                        {'role': 'user', 'content': content}]
            manifest = {'model': PUP_MODEL, 'revision': PUP_REVISION, 'mode': mode, 'messages': messages,
                        'system_prompts_sha256': hashlib.sha256(system_path.read_bytes()).hexdigest(),
                        'references': [{'id': reference, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                                       for reference, path in zip(frame['references'], references)]}
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                  return_dict=True, return_tensors='pt', enable_thinking=False).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=3584, do_sample=False)
            tokens = generated[0, inputs['input_ids'].shape[1]:]
            caption = processor.decode(tokens, skip_special_tokens=True).strip()
            if not caption or len(tokens) >= 3584:
                raise RuntimeError('Empty or truncated caption; inspect before continuing.')
            (workspace / 'prompts' / f"{frame['id']}.raw.txt").write_text(caption + '\n')
            write_json(workspace / 'prompts' / f"{frame['id']}.request.json", manifest)
            print('EXPANDED ' + frame['id'], flush=True)
        return

    from diffusers import ModularPipeline

    for frame in frames:
        if (workspace / 'frames' / f"{frame['id']}.png").exists():
            raise RuntimeError('Image already exists; preserve results instead of rerunning.')
        if not (workspace / 'prompts' / f"{frame['id']}.reviewed.txt").is_file():
            raise RuntimeError('Review captions before rendering.')
    pipe = ModularPipeline.from_pretrained(config['model'], revision=config['model_revision'])
    blocks = pipe.blocks
    blocks.sub_blocks.pop('prompt_upsampler')
    pipe = type(pipe)(blocks=blocks, pretrained_model_name_or_path=config['model'], revision=config['model_revision'])
    if 'prompt_upsampler' in pipe.blocks.sub_blocks:
        raise RuntimeError('Automatic PUP must be absent for reviewed captions.')
    required = ['text_encoder', 'tokenizer', 'transformer', 'vae']
    pipe.load_components(names=required, dtype=torch.bfloat16, revision=config['model_revision'])
    for name in required:
        if getattr(pipe, name, None) is None:
            raise RuntimeError(f'Component failed to load: {name}')
    pipe.to('cuda')
    pipe.vae.enable_tiling()
    adapter = root / config['checkpoint']
    adapter_hash = hashlib.sha256(adapter.read_bytes()).hexdigest()
    pipe.load_lora_weights(str(adapter), adapter_name='style')
    pipe.set_adapters('style', config['adapter_scale'])
    for frame in frames:
        prompt = (workspace / 'prompts' / f"{frame['id']}.reviewed.txt").read_text().strip()
        references = [workspace / 'frames' / f'{reference}.png' for reference in frame['references']]
        parameters = {'prompt': prompt, 'width': config['width'], 'height': config['height'],
                      'num_inference_steps': config['steps'], 'output': 'images',
                      'generator': torch.Generator(device='cuda').manual_seed(frame['seed'])}
        if references:
            parameters['image'] = [Image.open(path).convert('RGB') for path in references]
        picture = pipe(**parameters)[0]
        destination = workspace / 'frames' / f"{frame['id']}.png"
        picture.save(destination)
        write_json(workspace / 'generation' / f"{frame['id']}.json", {
            'model': config['model'], 'revision': config['model_revision'], 'adapter_sha256': adapter_hash,
            'scale': config['adapter_scale'], 'seed': frame['seed'], 'steps': config['steps'],
            'width': picture.width, 'height': picture.height, 'prompt': prompt,
            'references': [{'id': reference, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                           for reference, path in zip(frame['references'], references)],
            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        })
        print('SAVED ' + frame['id'], flush=True)


if __name__ == '__main__':
    main()
