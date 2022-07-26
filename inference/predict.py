# -*-Encoding: utf-8 -*-
import os
import time
import pickle
import numpy as np
import pandas as pd
import paddle
from paddle import nn
from Common import *
from testf import pred_test, pred_cnn, pred_gru
from dataset import TsDataset, TsDataset_cnn, TsDataset_gru
from paddle.io import DataLoader
from TSModel import TSModel, CNN, GRU


def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if paddle.device.is_compiled_with_cuda():
        paddle.device.set_device('gpu')
    else:
        paddle.device.set_device('cpu')

    test_x = pd.read_csv(settings["path_to_test_x"])
    test_x = test_x.fillna(0)
    test_x = test_x.sort_values(['TurbID','Day','Tmstamp'], ascending=True).reset_index(drop=True)
    
    
    model0 = TSModel()
    model1 = CNN()
    model2 = CNN()
    model3 = CNN()
    
    model4 = GRU()
    model5 = GRU()
    model6 = GRU()
    
    ## load CNN model with only time series
    model_state_dict = paddle.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'model/cnn_copy1.pdparams')  )
    model0.set_state_dict(model_state_dict)

    ## load CNN models with time series and space information
    param_dict_cnn1 = paddle.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'model/CNNSP_nor.pdparams'))
    model1.set_state_dict(param_dict_cnn1)
    param_dict_cnn2 = paddle.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'model/CNNSP_42.pdparams'))
    model2.set_state_dict(param_dict_cnn2)
    param_dict_cnn3 = paddle.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'model/CNNSP_2022.pdparams'))
    model3.set_state_dict(param_dict_cnn3)
    
    ## load GRU models with time series and space information
    model_state_dict4 = paddle.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'model/paddle_gru.pdparams')  )
    model4.set_state_dict(model_state_dict4)
    model_state_dict5 = paddle.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'model/paddle_gru42.pdparams')  )
    model5.set_state_dict(model_state_dict5)
    model_state_dict6 = paddle.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'model/paddle_gru2022.pdparams')  )
    model6.set_state_dict(model_state_dict6)

    ## load space information
    path_to_distdict = os.path.join(os.path.dirname(os.path.realpath(__file__)),'dist_dict.pickle')
    with open(path_to_distdict, 'rb') as f:
        dist_dict = pickle.load(f)
        
    df_group = test_x[['Day','Tmstamp','Patv']].groupby(['Day','Tmstamp'], as_index=False).agg(list)
    df_group.columns = ['Day','Tmstamp','Patv_list']

    for turbid in range(settings["capacity"]):        
        
        turb_data = test_x[test_x['TurbID']==turbid+1].reset_index(drop=True)
        turb_data = pd.merge(turb_data, df_group, how='left', on=['Day','Tmstamp'])
        turb_data['Patv_space'] = turb_data['Patv_list'].apply(lambda x: [x[i-1] for i in dist_dict[turbid+1][:121]])
    
        df_test = pd.DataFrame({'Wspd_seq':[list(turb_data['Wspd'])[-144:]],
                               'Patv_seq':[list(turb_data['Patv'])[-144:]],
                               'Etmp_seq':[list(turb_data['Etmp'])[-144:]],
                               'Itmp_seq':[list(turb_data['Itmp'])[-144:]],
                               'Pab1_seq':[list(turb_data['Pab1'])[-144:]],
                                'Patv_space':[list(turb_data['Patv_space'])[-1]],
                               'target': [[0]*288]})
        df_test['Temp_seq'] = df_test.apply(get_diff, axis=1)
        if turbid==0:
            test_new = df_test.copy()
        else:
            test_new = pd.concat([test_new, df_test])   
    
    ## CNN model with only time series + dataset1
    test_dataset = TsDataset(test_new)
    test_loader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False,
                             num_workers=4)
    pred_list0 = pred_test(model0, test_loader)
    
    ## CNN model with time series and space info+ dataset1
    test_dataset = TsDataset_cnn(test_new)
    test_loader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False,
                             num_workers=4)
    pred_list1 = pred_cnn( model1, test_loader)
    
    ## CNN model with time series and space info+ dataset2
    test_dataset = TsDataset_cnn(test_new)
    test_loader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False,
                             num_workers=4)
    pred_list2 = pred_cnn( model2, test_loader)
    
    ## CNN model with time series and space info+ dataset3
    test_dataset = TsDataset_cnn(test_new)
    test_loader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False,
                             num_workers=4)
    pred_list3 = pred_cnn( model3, test_loader)
    
    ## GRU model with time series and space info+ dataset1
    test_dataset = TsDataset_gru(test_new)
    test_loader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False,
                             num_workers=4)
    pred_list4 = pred_gru( model4, test_loader)
    
    ## GRU model with time series and space info+ dataset2
    test_dataset = TsDataset_gru(test_new)
    test_loader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False,
                             num_workers=4)
    pred_list5 = pred_gru( model5, test_loader)
    
    ## GRU model with time series and space info+ dataset3
    test_dataset = TsDataset_gru(test_new)
    test_loader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False,
                             num_workers=4)
    pred_list6 = pred_gru( model6, test_loader)

    pred_final = (np.array(pred_list1) + np.array(pred_list2) + np.array(pred_list3) + np.array(pred_list0) + 
                 np.array(pred_list4) + np.array(pred_list5) + np.array(pred_list6))/7


    return pred_final.reshape(settings["capacity"], settings['output_len'],1)
