import argparse
import hashlib
import re
from pathlib import Path

from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(description="Check a Klein style training dataset without changing images.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--trigger", default="zqvr style")
    parser.add_argument("--allow-low-resolution", action="store_true", help="Warn instead of failing on sub-megapixel images.")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error(f"Not a directory: {args.directory}")

    images = sorted(path for path in args.directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    issues = []
    warnings = []
    seen_pixels = {}
    seen_stems = set()
    if not 20 <= len(images) <= 40:
        issues.append(f"Found {len(images)} images; the brief recommends 20–40.")

    for path in images:
        if path.stem in seen_stems:
            issues.append(f"{path.name}: duplicate filename stem shares a caption.")
        seen_stems.add(path.stem)
        try:
            with Image.open(path) as source:
                normalized = ImageOps.exif_transpose(source).convert("RGB")
                width, height = normalized.size
                digest = hashlib.sha256(str(normalized.size).encode() + normalized.tobytes()).hexdigest()
            if width * height < 1_000_000:
                resolution_issues = warnings if args.allow_low_resolution else issues
                resolution_issues.append(f"{path.name}: {width}×{height}, below approximately 1 megapixel.")
            if digest in seen_pixels:
                issues.append(f"{path.name}: identical decoded pixels to {seen_pixels[digest]}.")
            seen_pixels[digest] = path.name
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            issues.append(f"{path.name}: unreadable image ({error}).")

        caption_path = path.with_suffix(".txt")
        if not caption_path.exists():
            issues.append(f"{path.name}: missing {caption_path.name}.")
            continue
        try:
            caption = caption_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as error:
            issues.append(f"{caption_path.name}: unreadable caption ({error}).")
            continue
        if not caption.endswith(args.trigger):
            issues.append(f"{caption_path.name}: must end with '{args.trigger}'.")
        content = caption.removesuffix(args.trigger).strip()
        if not content:
            issues.append(f"{caption_path.name}: missing scene description.")
        style_words = re.findall(r"\b(?:painting|illustration|drawing|watercolou?r|ink|sketch|vintage|style)\b", content, flags=re.IGNORECASE)
        if style_words:
            issues.append(f"{caption_path.name}: review possible style words: {', '.join(sorted(set(style_words)))}.")

    print(f"Checked {len(images)} images in {args.directory.resolve()}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for issue in issues:
        print(f"REVIEW: {issue}")
    if not issues:
        print("Basic file checks passed. Visually review style consistency, near-duplicates, borders, and watermarks separately.")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
