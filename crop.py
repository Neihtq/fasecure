import os
import pathlib
import argparse
import tqdm
import multiprocessing as mp

from data.crop_and_align import detect_and_align

parser  = argparse.ArgumentParser(description='Detect, crop and align faces.')
parser.add_argument('--dir', type=str, help='Directory of images')
parser.add_argument('--o', type=str, help='Directory where aligned images will be saved.')
args = parser.parse_args()

def main():
    src = args.dir
    
    output = pathlib.Path(output(args.dir))
    output.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for folder in os.listdir(src):
        tmp = os.path.join(src, folder)
        dest = os.path.join(output, folder)
        pathlib.Path(dest).mkdir(parents=True, exist_ok=True)

        for img in os.listdir(tmp):
            fpath = os.path.join(tmp, img)
            paths.append((fpath, str(output)))

    with mp.Pool(processes=os.cpu_count()) as pool:
        res = list(tqdm.tqdm(pool.imap(detect_and_align, paths), total=len(paths)))
        
    for folder in os.listdir(output):
        tmp = os.path.join(output, folder)
        if len(os.listdir(tmp)) == 0:
            os.rmdir(tmp)

if __name__ == '__main__':
    main()