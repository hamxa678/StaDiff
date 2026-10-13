"""Create a small toy image folder so you can try the pipeline:  python make_demo_data.py --out demo_images --n 80"""
import argparse

from stadiff.data import write_demo_images

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="demo_images")
    p.add_argument("--n", type=int, default=80)
    a = p.parse_args()
    write_demo_images(a.out, a.n)
