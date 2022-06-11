import os
import json

import numpy as np
from tqdm import tqdm
import pandas as pd

DATAPATH = '/home/q/data/20220518_train/'
OUTPUTNAME = '价量'
FEATURES = ['tradeDate', 'openPrice', 'closePrice' ,'highestPrice', 'lowestPrice', 'turnoverRate', 'returns', 'returns_ind', 'stock', '1d_alpha_rank']

def format_transform(PATH, output_name, FEATURES):
    '''
    transform csv data into Numpy array
    输入参数：csv路径，npy保存路径，读取需要的csv中的列
    '''
    t = os.listdir(PATH)
    t.sort()
    train_data_frame = []
    slice_idx = {}
    for file in tqdm(t):
        try:
            slice_idx[file[:-4]] = len(train_data_frame)
            data_frame = pd.read_csv(PATH + file, usecols = FEATURES)

            tmp = np.array(data_frame).reshape(-1, 60, len(FEATURES))
            train_data_frame.extend(tmp)
        except:
            print(PATH + file + ' is empty')
            continue

    data = np.array(train_data_frame, dtype = object)
   
    np.save(os.path.join(os.path.dirname(__file__), f'dataset/{output_name}.npy'), data)
    with open(os.path.join(os.path.dirname(__file__), f'dataset/{output_name}.txt'), 'w') as convert_file:
        convert_file.write(json.dumps(slice_idx))


if __name__ == '__main__':
    format_transform(DATAPATH, OUTPUTNAME, FEATURES)

