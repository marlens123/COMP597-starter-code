from pathlib import Path
import argparse
from src.data.fakeimagenet.datagen import generate_fakeimagenet

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", default=512, type=int)
parser.add_argument("--batch-count", default=60, type=int)
parser.add_argument("--device-count", default=1, type=int)
parser.add_argument("--device", default=None, type=str)
parser.add_argument("--image-size", default=[3, 384, 384], type=int, nargs="+")
parser.add_argument("--val", default=0.1, type=float, nargs="+")
parser.add_argument("--test", default=0.1, type=float, nargs="+")
parser.add_argument("--output", default="data/fakeimagenet", type=str)
args, _ = parser.parse_known_args()

if not Path(args.output).exists():
    generate_fakeimagenet(args=args)
