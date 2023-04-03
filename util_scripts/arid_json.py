import argparse
import json
from pathlib import Path

import pandas as pd

from utils import get_n_frames
import pdb
import tqdm
import json

def convert_json_to_dict(csv_path, subset):
    lines = json.load(open(csv_path,'r'))
    database = {}

    for line in lines:
        video_id = line['id']
        database[video_id] = {}
        database[video_id]['subset'] = subset
        if subset != 'testing':
            label = line['template'].replace('[','').replace(']','')
            database[video_id]['annotations'] = {'label': label}
        else:
            database[video_id]['annotations'] = {}

    return database


def convert_txt_to_dict(txt_path, subset):
    lines = open(txt_path, 'r').readlines()
    keys = []
    key_labels = []
    database = {}

    for line in lines:
        if subset != 'testing':
            label, video_id = lines[0].split()[1], lines[0].split()[2].split('/')[1][:-4]
        else:
            video_id = line.split()[2].split('/')[1][:-4]

        database[video_id] = {}
        database[video_id]['subset'] = subset
        if subset != 'testing':
            database[video_id]['annotations'] = {'label': label}
        else:
            database[video_id]['annotations'] = {}

    return database


def load_labels(class_file_path):
    data = open(class_file_path, 'r').readlines()
    data = [e.strip('\n') for e in data]
    return data
#    data = pd.read_csv(train_csv_path, header=None)
#    return data.iloc[:, 0].tolist()


def convert_arid_txt_to_json(class_file_path, train_txt_path, val_txt_path,
                            test_txt_path, video_dir_path, dst_json_path):
    if class_file_path.exists():
        labels = load_labels(class_file_path)
    else:
        labels = [str(i) for i in range(11)]
    train_database = convert_txt_to_dict(train_txt_path, 'training')
    val_database = convert_txt_to_dict(val_txt_path, 'validation')
    if test_txt_path.exists():
        test_database = convert_txt_to_dict(test_txt_path, 'testing')

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    if test_txt_path.exists():
        dst_data['database'].update(test_database)

    count = 0
    for k, v in tqdm.tqdm(dst_data['database'].items()):
        if 'label' in v['annotations']:
            label = v['annotations']['label']
        else:
            label = 'test'

        video_path = video_dir_path / k
        n_frames = get_n_frames(video_path)
        v['annotations']['segment'] = (1, n_frames + 1)
        v['video_path'] = str(video_path)
#        count += 1
#        if count == 1000:
#            break

    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path',
        default='/mnt/ssd1/yuecong/data/ARID/list_cvt_v1',
        type=Path,
        help=('Directory path including moments_categories.txt, '
              'trainingSet.csv, validationSet.csv, '
              '(testingSet.csv (optional))'))
    parser.add_argument('video_path',
        default='/mnt/ssd1/yuecong/data/ARID/video_jpg',
                        type=Path,
                        help=('Path of video directory (jpg).'
                              'Using to get n_frames of each video.'))
    parser.add_argument('dst_path',
                        default='/mnt/ssd1/yuecong/data/ARID/video_jpg',
                        type=Path,
                        help='Path of dst json file.')

    args = parser.parse_args()

    class_file_path = args.dir_path / 'category.txt'
    train_txt_path = args.dir_path / 'ARID_split1_train.txt'
    val_txt_path = args.dir_path / 'ARID_split1_other.txt'
    test_txt_path = args.dir_path / 'ARID_split1_test.txt'
#    train_csv_path = args.dir_path / 'train_videofolder.txt'
#    val_csv_path = args.dir_path / 'val_videofolder.txt'
#    test_csv_path = args.dir_path / 'test_videofolder.txt'

    convert_arid_txt_to_json(class_file_path, train_txt_path, val_txt_path,
                            test_txt_path, args.video_path, args.dst_path)
