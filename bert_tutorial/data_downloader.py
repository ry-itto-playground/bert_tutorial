import urllib.request
import os
import sys
import shutil
import tarfile

DATA_DIR = f'{os.path.dirname(__file__)}/data'
DATASET_FILE = 'yelp_review_full_csv.tgz'
TRAIN_CSV_FILE = 'train.csv'
TEST_CSV_FILE = 'test.csv'
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
    tgz_file: str = '/'.join([DATA_DIR, DATASET_FILE])

    if not os.path.exists(tgz_file):
        print(f'Download to {tgz_file}')
        urllib.request.urlretrieve(
            url=URL,
            filename=tgz_file,
            reporthook=progress)
        print('Done.')
    else:
        print('Skip download dataset.')

    extract_tgz_file(tgz_file)


def extract_tgz_file(filepath: str):
    print('extract download tgz file')
    with tarfile.open(filepath) as tar:
        tar.extractall(
            path=DATA_DIR,
        )

    extracted_dir = f'{DATA_DIR}/yelp_review_full_csv'
    for file in [TRAIN_CSV_FILE, TEST_CSV_FILE]:
        shutil.copy(f'{extracted_dir}/{file}', DATA_DIR)
    shutil.rmtree(extracted_dir)

    print('Done.')


if __name__ == "__main__":
    download()
