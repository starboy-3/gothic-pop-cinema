import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


MODEL_ID = 'black-forest-labs/flux-3-image-klein-pup'
REVISION = '3f3edb52b2c55de0d49feab28598b68f7f8e4830'


def main():
    parser = argparse.ArgumentParser(description='Expand an image request with BFL PUP and its unchanged official system prompt.')
    parser.add_argument('--request-file', type=Path, required=True)
    parser.add_argument('--image', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Save the request without downloading weights or running inference.')
    args = parser.parse_args()
    request = args.request_file.read_text().strip()
    if not request:
        parser.error('Request must not be empty.')
    for path in args.image:
        if not path.is_file():
            parser.error(f'Reference image does not exist: {path}')
    manifest_path = args.output.with_suffix('.request.json')
    if args.output.exists() or manifest_path.exists():
        parser.error('Output already exists; choose a new output path to preserve prior results.')
    system_path = Path(hf_hub_download(MODEL_ID, 'system_prompts.json', revision=REVISION))
    system_prompts = json.loads(system_path.read_text())
    mode = 'i2i' if args.image else 't2i'
    content = [{'type': 'text', 'text': request}]
    if args.image:
        content = [{'type': 'image', 'image': str(path.resolve())} for path in args.image]
        content.append({'type': 'text', 'text': 'Reference images follow in order (the first image, the second image, ...).\n'
                        f'User request: {request}\n Output the caption.'})
    messages = [{'role': 'system', 'content': [{'type': 'text', 'text': system_prompts[mode]}]}, {'role': 'user', 'content': content}]
    manifest = {'model': MODEL_ID, 'revision': REVISION, 'mode': mode, 'messages': messages,
                'system_prompts_sha256': hashlib.sha256(system_path.read_bytes()).hexdigest(),
                'status': 'prepared', 'style_postprocessing': 'Review/remove conflicting opening medium phrase, then append the style trigger before image inference.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if args.prepare_only:
        print(f'Prepared official {mode} request: {manifest_path}. No model inference performed.')
        return

    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError('Run inference in the configured Nebius GPU environment, not on the laptop.')
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, device_map='auto',
    ).eval()
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors='pt', enable_thinking=False,
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=3584, do_sample=False)
    tokens = generated[0, inputs['input_ids'].shape[1]:]
    if len(tokens) >= 3584:
        raise RuntimeError('Upsampler reached the token limit; review before using a potentially truncated caption.')
    caption = processor.decode(tokens, skip_special_tokens=True).strip()
    if not caption:
        raise RuntimeError('Upsampler returned an empty caption.')
    args.output.write_text(caption + '\n')
    manifest['status'] = 'generated_needs_style_review'
    manifest['caption'] = caption
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved raw caption: {args.output}. Review the medium phrase and append your trigger before generation.')


if __name__ == '__main__':
    main()
