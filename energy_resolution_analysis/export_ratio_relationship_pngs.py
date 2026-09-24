#!/usr/bin/env python3
"""Render combined and individual ratio relationship PNG files via Edge."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from PIL import Image, ImageChops


KEY_TO_FILE = {
    "dt": "dt",
    "wS2CDF_max": "wS2CDF_max",
    "yS2Tcor_max": "yS2Tcor_max",
    "xS2Bcor_max": "xS2Bcor_max",
}


def make_fragment(template, payload, title, single=False):
    text = re.sub(r"<h2>.*?</h2>", "<h2>{}</h2>".format(title), template, count=1)
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    text = re.sub(
        r'(<script type="application/json" id="energy-ratio-data">).*?(</script>)',
        lambda match: match.group(1) + encoded + match.group(2), text, count=1,
        flags=re.DOTALL,
    )
    if single:
        text += "\n<style>#energy-ratio-relations .ratio-grid{grid-template-columns:1fr;max-width:940px}</style>\n"
    return text


def crop_white(path):
    image = Image.open(str(path)).convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    difference = ImageChops.difference(image, background).convert("L")
    difference = difference.point(lambda value: 255 if value > 8 else 0)
    box = difference.getbbox()
    if box:
        pad = 18
        box = (max(0, box[0]-pad), max(0, box[1]-pad),
               min(image.width, box[2]+pad), min(image.height, box[3]+pad))
        image.crop(box).save(str(path), dpi=(300, 300))


def render(fragment_path, standalone_path, png_path, render_script, edge, size):
    subprocess.check_call([sys.executable, str(render_script), str(fragment_path), str(standalone_path)])
    command = [str(edge), "--headless", "--disable-gpu", "--hide-scrollbars",
               "--virtual-time-budget=3000", "--window-size={},{}".format(*size),
               "--screenshot={}".format(png_path.resolve()), standalone_path.resolve().as_uri()]
    subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not png_path.exists(): raise RuntimeError("Edge did not create {}".format(png_path))
    crop_white(png_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--background-json", required=True)
    parser.add_argument("--render-script", required=True)
    parser.add_argument("--edge", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    scratch = output / "_render_cache"; scratch.mkdir(exist_ok=True)
    template = Path(args.template).read_text(encoding="utf-8")
    datasets = [
        ("calibration", json.loads(Path(args.calibration_json).read_text(encoding="utf-8")), "Calibration OOF events"),
        ("background", json.loads(Path(args.background_json).read_text(encoding="utf-8")), "Background events"),
    ]
    render_script = Path(args.render_script); edge = Path(args.edge)

    for dataset_index, (prefix, payload, label) in enumerate(datasets, start=1):
        fragment = scratch / (prefix + "_combined_fragment.html")
        standalone = scratch / (prefix + "_combined.html")
        fragment.write_text(make_fragment(template, payload,
                           "v15 S10: energy ratio versus model inputs - " + label), encoding="utf-8")
        target = output / ("{:02d}_{}_ratio_vs_inputs_2x2.png".format(dataset_index, prefix))
        render(fragment, standalone, target, render_script, edge, (1600, 1200))
        print(target)

    number = 3
    for prefix, payload, label in datasets:
        for panel in payload["panels"]:
            single_payload = dict(payload); single_payload["panels"] = [panel]
            fragment = scratch / (prefix + "_" + panel["key"] + "_fragment.html")
            standalone = scratch / (prefix + "_" + panel["key"] + ".html")
            fragment.write_text(make_fragment(template, single_payload,
                               "{}: ratio versus {}".format(label, panel["label"]), single=True),
                               encoding="utf-8")
            target = output / ("{:02d}_{}_ratio_vs_{}.png".format(
                number, prefix, KEY_TO_FILE[panel["key"]]))
            render(fragment, standalone, target, render_script, edge, (1100, 760))
            print(target); number += 1


if __name__ == "__main__":
    main()
