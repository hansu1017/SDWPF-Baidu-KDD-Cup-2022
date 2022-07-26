import pandas as pd
import numpy as np
import paddle
from paddle.io import Dataset

## CNN model using only time series
class TsDataset(Dataset):
    def __init__(self, df):
        self.seq_list1 = list(df['Wspd_seq'])
        self.seq_list2 = list(df['Patv_seq'])

        self.seq_list3 = list(df['Pab1_seq'])
        self.seq_list4 = list(df['Temp_seq'])

        self.label_list = df.target.values

    def __getitem__(self, index):
        seq = self.seq_list1[index] + self.seq_list2[index] + self.seq_list3[index] + self.seq_list4[index]
        seq = np.array(seq).astype('float')
        seq.resize(4, 12, 12)

        label = np.array(self.label_list[index]).astype('float')

        seq = paddle.to_tensor(seq)

        return seq, label

    def __len__(self):
        return len(self.seq_list1)

## CNN model using time series and space information
class TsDataset_cnn(Dataset):
    def __init__(self, df):

        self.seq_list1 = list(df['Wspd_seq'])
        self.seq_list2 = list(df['Patv_seq'])
        
        self.seq_list3 = list(df['Pab1_seq'])
        self.seq_list4 = list(df['Temp_seq'])
        
        self.seq_list5 = list(df['Patv_space'])

        
        self.label_list = df.target.values

    def __getitem__(self, index):
        
        seq = self.seq_list1[index][-121:]+self.seq_list2[index][-121:]+\
                  self.seq_list3[index][-121:]+self.seq_list4[index][-121:]
        seq = np.array(seq).astype('float') 
        seq.resize(11, 11, 4)
        
        image = np.array(self.seq_list5[index]).astype('float') 
        image.resize(11, 11, 1)

        label = np.array( self.label_list[index] ).astype( 'float' )
        
        seq = paddle.to_tensor(seq)
        space_data = paddle.to_tensor(image)
        

        return seq, space_data, label


    def __len__(self):
        return len(self.seq_list1)

## GRU model using time series and space information
class TsDataset_gru(Dataset):
    def __init__(self, df):

        self.seq_list1 = list(df['Wspd_seq'])
        self.seq_list2 = list(df['Patv_seq'])
        
        self.seq_list3 = list(df['Etmp_seq'])
        self.seq_list4 = list(df['Itmp_seq'])
        
        self.seq_list5 = list(df['Patv_space'])

        
        self.label_list = df.target.values

    def __getitem__(self, index):
        
        seq = np.vstack((self.seq_list1[index], self.seq_list2[index], self.seq_list3[index], self.seq_list4[index]))
        seq = np.array(seq).astype('float') 
        
        image = np.array(self.seq_list5[index]).astype('float') 
        image.resize(11, 11, 1)

        label = np.array( self.label_list[index] ).astype( 'float' )
        
        seq = paddle.to_tensor( seq )
        space_data = paddle.to_tensor(image)

        return seq, space_data, label


    def __len__(self):
        return len(self.seq_list1)