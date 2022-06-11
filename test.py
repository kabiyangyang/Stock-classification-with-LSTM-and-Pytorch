import pandas as pd
import os
import torch
from huice import BackTesting

holdday = 1
hold_num = 5

date_parameter = {'train_start_date' :'2015-01-05', 'train_end_date' : '2022-01-04', 
                        'valid_end_date' : '2022-04-27'
                    }

back_test = BackTesting(holdday, hold_num, date_parameter)
back_test.get_stock_data()


factor = pd.read_csv("/home/q/下载/xuyang_test/checkpoint/{'features_num': 7, 'seq_length': 60, 'Bidirectional_layers': [], 'lstm_layers': [32], 'FC_layers': [3]}/-3.61309276336264{'features_num': 7, 'seq_length': 60, 'Bidirectional_layers': [], 'lstm_layers': [32], 'FC_layers': [3]}_{'batch_size': 1024, 'epoch': 40, 'LR': 0.0001}.csv", index_col=0)

back_test.draw_profit(factor, '/home/q/下载/xuyang_test/', '1', '1')




