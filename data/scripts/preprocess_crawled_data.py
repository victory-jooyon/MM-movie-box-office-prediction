import glob
import json
import pathlib


if __name__ == '__main__':

    RAW_DIR = pathlib.Path().absolute().parent
    crawled_dir = f'{RAW_DIR}/json/crawled_data'
    crawled_files = glob.glob(f'{crawled_dir}/crawled_data_*.json')
    all = []
    for f in crawled_files:
        with open(f, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all.extend(data)

    print('all data:', len(all))
    with open(f'{crawled_dir}/crawled_data_all.json', 'w', encoding='utf-8') as f:
        json.dump(all, f, indent=4)
