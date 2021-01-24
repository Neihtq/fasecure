import os
import argparse
import tqdm
import multiprocessing as mp

from data.crop_and_align_LFW import detect_and_align

parser  = argparse.ArgumentParser(description='Detect, crop and align faces.')
parser.add_argument('--dir', type=str, help='Directory of images')
args = parser.parse_args()

def main():
    src = args.dir

    paths = []
    for folder in os.listdir(src):
        tmp = os.path.join(src, folder)
        for img in os.listdir(tmp):
            fpath = os.path.join(tmp, img)
            paths.append(fpath)

    with mp.Pool(processes=os.cpu_count()) as pool:
        res = list(tqdm.tqdm(pool.imap(detect_and_align, paths), total=len(paths)))

if __name__ == '__main__':
    main()