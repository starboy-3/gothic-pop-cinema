import argparse
import re
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Create an independent Klein LoRA training workspace.")
    parser.add_argument("name", help="Lowercase style name, using letters, numbers, and hyphens.")
    parser.add_argument("--trigger", help="Caption trigger; defaults to '<name-without-hyphens> style'.")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", args.name):
        parser.error("Use lowercase letters, numbers, and single separating hyphens.")
    trigger = args.trigger or f"{args.name.replace('-', '')} style"
    if not trigger.strip() or "\n" in trigger or "\r" in trigger:
        parser.error("The trigger must be a nonempty single line.")

    root = Path(__file__).resolve().parent
    style_path = root / "styles" / args.name
    if style_path.exists():
        parser.error(f"Already exists: {style_path}; existing training data will not be overwritten.")
    config = yaml.safe_load((root / "templates" / "train.yaml").read_text())
    config["config"]["name"] = f"klein-{args.name}"
    process = config["config"]["process"][0]
    process["training_folder"] = f"styles/{args.name}/output"
    process["datasets"][0]["folder_path"] = f"styles/{args.name}/data"
    process["sample"]["prompts"] = [prompt.replace("zqvr style", trigger) for prompt in process["sample"]["prompts"]]
    style_path.mkdir(parents=True)
    (style_path / "data").mkdir()
    (style_path / "output").mkdir()
    (style_path / "train.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (style_path / "trigger.txt").write_text(trigger + "\n", encoding="utf-8")
    print(f"Created {style_path}")
    print(f"Caption trigger: {trigger}")
    print(f"Put images and matching captions in {style_path / 'data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
