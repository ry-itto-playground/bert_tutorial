import urllib.request
import os
import sys

DATA_DIR = f'{os.path.dirname(__file__)}/data'
DATASET_FILE = 'yelp_review_full_csv.tgz'
URL = 'https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz'


def progress(block_count, block_size, total_size):
    ''' コールバック関数 '''
    percentage = 100.0 * block_count * block_size / total_size
    # 改行したくないので print 文は使わない
    sys.stdout.write("%.2f %% ( %d KB )\r"
                     % (percentage, total_size / 1024))


def download():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    zip_file: str = '/'.join([DATA_DIR, DATASET_FILE])

    if not os.path.exists(zip_file):
        print(f'Download to {zip_file}')
        urllib.request.urlretrieve(
            url=URL,
            filename=zip_file,
            reporthook=progress)
        print('Done.')
    else:
        print('Skip download dataset.')


if __name__ == "__main__":
    download()
