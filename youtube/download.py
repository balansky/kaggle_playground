import argparse
from kaggle import api
import os

COMPETITION = "imaterialist-challenge-furniture-2018"

def main(dataset_dir, force=False):
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    if not os.listdir(dataset_dir) or force:
        api.dataset_download_files("datasnaek/youtube-new", path=dataset_dir, force=True, quiet=False, unzip=True)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default="data"
    )
    parser.add_argument(
        '--force',
        default=False,
        action='store_true'
    )
    args, unparsed = parser.parse_known_args()
    main(args.dataset_dir, args.force)