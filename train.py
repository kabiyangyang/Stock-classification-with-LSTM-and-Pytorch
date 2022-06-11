from multiprocessing.sharedctypes import Value
import os
import json
import random

# #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"]=""
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torchinfo import summary
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import OneHotEncoder

#from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from huice import BackTesting





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device( 'cpu')
def seed(num : int):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    

class PrepareData():
    def __init__(self, datapath: str = '/home/q/下载/xuyang_test/dataset/', features: str = '价量'):
        def __load_data__():
            '''
            读取文件 
            返回data数据，contents是data的索引，dict格式，key是日期，value是idx
            '''
            data = np.load(datapath + features + '.npy', allow_pickle = True)
            with open(datapath + features + '.txt', 'r') as j:
                contents = json.loads(j.read())
            
            return data, contents
        self.data, self.conents = __load_data__()


        


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, data, contents, 
                train_start_date :str = '2015-01-05', train_end_date : str = '2022-01-04', valid_end_date : str = '2022-04-27', mode: str = 'train'):
                      
        def __split_train_test__(data, slices):
            '''
            start_date normaly 
            '2012-01-04', '2013-01-04', '2014-01-02', '2015-01-05'
            train_end_date normaly
            '2018-01-02', '2019-01-02', '2020-01-02', '2021-01-04'
            valid_end_date normaly
            '2019-01-02', '2020-01-02', '2021-01-04', -1
            
            返回训练和测试数据
            '''
            idx_train_start = slices[train_start_date]
            idx_train_end = slices[train_end_date]
            idx_test_end = slices[valid_end_date]
            
            return data[idx_train_start:idx_train_end, :, :], data[idx_train_end:idx_test_end, :, :]           


        def __data_process__(data, feature_col, label_col, stock_col, date_col, mode : str = 'classification'):
            '''
            将原始数据分成训练特征和标签
            返回特征数据和标签数据
            '''
            features = data[:, :, feature_col].astype('float32')
            label = data[:, -1, label_col].reshape(-1, 1)
            stk = data[:, -1, stock_col].reshape(-1, ).tolist()
            date = data[:, -1, date_col].reshape(-1, ).tolist()
            if mode == 'classification':
                #前100名是‘1’，其余为‘0’
                ohe = OneHotEncoder(sparse=False, dtype="int32")
                label = np.array(list(map(lambda x : 1 if x<100 else (-1 if x>700 else 0),label))).reshape(-1, 1).astype('int32')
                label = ohe.fit_transform(label)
            return features, label, stk, date

        
        train_data, valid_data = __split_train_test__(data, contents)
        if(mode == 'train'):
            self.x, self.y, self.stock, self.date = __data_process__(train_data, [1, 2, 3, 4, 5, 6, 7], [9], [8], [0])
        elif(mode == 'valid'):
            self.x, self.y, self.stock, self.date = __data_process__(valid_data, [1, 2, 3, 4, 5, 6, 7], [9], [8], [0])
        else:
            raise ValueError('not a valid mode')

        
    def __getitem__(self, index):
     
        return self.x[index], self.y[index], self.stock[index], self.date[index]
        
    def __len__(self):
       
        return self.x.shape[0]
        

class GetinterLSTMOutput(nn.Module):
    def forward(self, x):
        out, _ = x
        return out


class GetlastLSTMOutput(nn.Module):
    def forward(self, x):
        
        return x.reshape([-1, x.shape[1] * x.shape[2]])
    

class LSTM(nn.Module):
    def __init__(self, features_num : int, seq_length : int,
                Bidirectional_layers : list = [], lstm_layers : list = [], FC_layers : list = []):
        super().__init__()

        self.seq_length = seq_length
        self.feature_size = features_num

        
        layers = []
        for layer_num, unit in enumerate(Bidirectional_layers):
            input_size = features_num if  layer_num == 0 else Bidirectional_layers[layer_num - 1]*2
            
            layers.append(nn.LSTM(input_size = input_size,
                                 hidden_size = unit, 
                                 num_layers = 1, 
                                 batch_first = True, 
                                 bidirectional = True))
            layers.append(GetinterLSTMOutput())
            
        
        for layer_num, unit in enumerate(lstm_layers):
            
            if(len(Bidirectional_layers) == 0):
                if(layer_num == 0):
                    input_size = features_num
                else:
                    input_size = lstm_layers[layer_num - 1]
            else:
                if(layer_num == 0):
                    input_size = Bidirectional_layers[-1]*2
                else:
                    input_size = lstm_layers[layer_num - 1]
            
            layers.append(nn.LSTM(input_size = input_size,
                                 hidden_size = unit, 
                                 num_layers = 1, 
                                 batch_first = True, 
                                 bidirectional = False))
            layers.append(GetinterLSTMOutput())
        layers.append(GetlastLSTMOutput())

        
        for layer_num, unit in enumerate(FC_layers):
            if(len(lstm_layers) == 0):
                if(layer_num == 0):
                    input_size = Bidirectional_layers[-1] * 2 * self.seq_length
                else:
                    input_size = FC_layers[layer_num - 1]
            else:
                if(layer_num == 0):
                    input_size = lstm_layers[-1] * self.seq_length
                else:
                    input_size = FC_layers[layer_num - 1]
            layers.append(nn.Linear(input_size, unit))
        #layers.append(nn.Softmax(dim = 1))


        self.model = nn.Sequential(*layers)
      
           
        
       
        
     
    def forward(self, x):
        x = x.view(-1, self.seq_length, self.feature_size)

        return self.model(x)



def train_and_predict(train_data: object, valid_data : object, 
                      Model_parameter, Hyper_parameter : dict,
                      model_path:str):

    seed(158)
    batch_size = Hyper_parameter['batch_size']
    LR = Hyper_parameter['LR']
    epoch_num = Hyper_parameter['epoch']
    model_name = 'LSTM'

    train_loader = DataLoader(train_data, batch_size, drop_last = True)
   
    model = LSTM(**Model_parameter).to(device=device)
    summary(model, input_size = (batch_size, 60, 7))
    optim = torch.optim.Adam(model.parameters(), LR)
    scheduler = StepLR(optim, step_size=10, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epoch_num):
        for batch_idx, (x, y_true, _, _) in enumerate(train_loader):
            #forward
            model.train()
            x = x.to(device, dtype = torch.float)
            y_true = y_true.to(device, dtype = torch.float)
            y_pred = model(x).to(device, dtype = torch.float)
            #backward
            loss = loss_fn(y_pred, y_true)
            optim.zero_grad()
            loss.backward()
            optim.step()
        #scheduler.step()
        
        print(f'train : epoch={epoch},loss={loss.item()}, lr = {scheduler.get_last_lr()}')
        
    
    state = {'model': model.state_dict(), 'optimizer' : optim.state_dict()}
    torch.save(state, os.path.join(model_path, 
                f'{model_name}_{Model_parameter}_{Hyper_parameter}.pth'))

   
    valid_loader = DataLoader(valid_data, batch_size, drop_last = True)
    model = LSTM(**Model_parameter).to(device=device)
    checkpoint = torch.load(os.path.join(model_path, 
                f'{model_name}_{Model_parameter}_{Hyper_parameter}.pth'))
    model.load_state_dict(checkpoint['model'])


    y_pred = []
    y = []
    stock = []
    trade_date = []
    with torch.no_grad():
        for batch_idx, (valid_X, y_true, stk, date) in enumerate(valid_loader):
            model.eval()
            x_pred = valid_X.to(device, dtype = torch.float)
            y_pred_batch = model(x_pred).cpu().numpy()
            y_pred.extend((y_pred_batch[:, 2] - y_pred_batch[:, 0]) .reshape(-1, ).tolist())
            y.extend(y_true.cpu().numpy().reshape(-1, ).tolist())
            stock.extend(stk)
            trade_date.extend(date)
    result = np.array([y_pred, stock, trade_date]).T
    factor = pd.DataFrame(index = np.unique(trade_date), columns = np.unique(stock))
    for i in tqdm(range(result.shape[0])):
        factor[result[i][1]][result[i][2]] = float(result[i][0])
    return factor



def run(holdday: int, hold_num : int, date_parameter: dict, param_path: str):
    print('#############Getting stock data#################')
    back_test = BackTesting(holdday, hold_num, date_parameter)
    back_test.get_stock_data()
    D = PrepareData()
    print('############Loading dataset####################')
    
    train_data = LoadDataset(D.data, D.conents, mode = 'train', train_start_date= date_parameter['train_start_date'],
                            train_end_date = date_parameter['train_end_date'], valid_end_date=date_parameter['valid_end_date'])

    valid_data = LoadDataset(D.data, D.conents, mode = 'valid', train_start_date= date_parameter['train_start_date'],
                            train_end_date = date_parameter['train_end_date'], valid_end_date=date_parameter['valid_end_date'])
    print('############Get training parameter#############')
    with open(param_path, 'r') as j:
        params = json.loads(j.read())
    
    hyper_param = params['hyper_param']
    model_param = params['model_param']

    print('############Training start####################')

    for m in model_param:
        modelpath = os.path.join(os.path.dirname(__file__), f'checkpoint/{m}')
        if not os.path.exists(modelpath):
            os.makedirs(modelpath)

        for h in hyper_param:
            factor = train_and_predict(train_data, valid_data, m, h, modelpath)
            back_test.draw_profit(factor, modelpath, m, h)
    
    


if __name__ == '__main__':

    #Date Parameter
    date_parameter = {'train_start_date' :'2015-01-05', 'train_end_date' : '2022-01-04', 
                        'valid_end_date' : '2022-04-27'
                    }
  
    run(1, 20, date_parameter, '/home/q/下载/xuyang_test/parameters.json')

  

