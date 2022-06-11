import os
import json


import numpy as np

def generate_hyper_parameter(batch : list, epoch : np.arange,  LR: list):
    hyper_param = []
    mesh = np.array(np.meshgrid(batch, epoch, LR)).T.reshape(-1,3).tolist()
    for para in mesh:
        hyper_param.append({'batch_size': int(para[0]), 'epoch': int(para[1]), 'LR': para[2]})
    return hyper_param


def generate_model_parameter(layer_sum : int, features_num : int, seq_length: int, Output_len: int):
    #Model_parameter = {'features_num': 7, 'seq_length' : 60, 'Bidirectional_layers' : [128], 'lstm_layers' : [64, 32], 'FC_layers' : [3]}
    model_param = []

    for BiL in np.arange(0, layer_sum+1):
        for L in np.arange(0, layer_sum+1):
            summe = BiL + L
            if(BiL + L != 0 and summe <= layer_sum):
                #unit = list(map(lambda x: 32 * (x+1), range(summe)))

                unit = [32]
                while(summe > 1):
                    unit.append(unit[-1] * 2)
                    summe -= 1
                unit.reverse()
                model_param.append({'features_num': features_num, 'seq_length' : seq_length,'Bidirectional_layers' : unit[:BiL], 
                                    'lstm_layers' : unit[BiL:], 'FC_layers' : [Output_len]})


    return model_param



if __name__ == '__main__':
    hyper_param = generate_hyper_parameter([512, 1024, 4096, 8192], np.arange(20, 60, 20), [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    model_param = generate_model_parameter(3, 7, 60, 3)
    param = {'hyper_param': hyper_param, 'model_param': model_param}
    with open(os.path.join(os.path.dirname(__file__), f'parameters.json'), 'w') as convert_file:
        convert_file.write(json.dumps(param))
    



