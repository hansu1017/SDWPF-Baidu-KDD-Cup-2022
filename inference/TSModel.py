import os
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.nn.utils import weight_norm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if paddle.device.is_compiled_with_cuda():
    paddle.device.set_device('gpu')
else:
    paddle.device.set_device('cpu')

## CNN model with multichannel time series
class TSModel(nn.Layer):
    def __init__(self):
        super(TSModel, self).__init__()

        self.cnnLayer1 = nn.Sequential(
            nn.Conv2D(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2D((12, 12))
        )
        self.res1 = nn.Conv2D(4, 64, kernel_size=1, stride=1)
        self.cnnLayer2 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(128),
            nn.GELU(),
        )
        self.res2 = nn.Conv2D(64, 128, kernel_size=1, stride=1)
        self.cnnLayer3 = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2D((3, 3))
        )

        self.Linear1 = nn.Linear(8, 8, bias_attr=True)
        self.Linear2 = nn.Linear(8, 1, bias_attr=True)

    def forward(self, X):

        cnn_out1 = self.cnnLayer1(X)
        res_out1 = self.res1(X)
        output1 = cnn_out1 + res_out1


        cnn_out2 = self.cnnLayer2(output1)
        res_out2 = self.res2(output1)
        output2 = cnn_out2 + res_out2

        cnn_out3 = self.cnnLayer3(output2)
        output3 = paddle.reshape(cnn_out3, ( cnn_out3.shape[0], 288, -1))

        out1 = self.Linear1(output3)
        final_output = self.Linear2(out1)


        return final_output

## CNN model with spatial-temporal information
class CNN(nn.Layer):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.cnnLayer1 = nn.Sequential(
        nn.Conv2D(4, 64, kernel_size=3, stride=1, padding=2),
        nn.BatchNorm2D(64),
        nn.GELU(),
        nn.AdaptiveAvgPool2D((11, 11))
        )
        self.res1 = nn.Conv2D(4, 64, kernel_size=1, stride=1)
          
        self.cnnLayer2 = nn.Sequential(
            nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=2), 
            nn.BatchNorm2D(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2D((3, 3))
        )
         
        self.cnnLayer3 = nn.Sequential(
        nn.Conv2D(1, 64, kernel_size=3, stride=1, padding=1), 
        nn.BatchNorm2D(64),
        nn.GELU(),
        nn.AdaptiveMaxPool2D((3,3)))
       
        self.Linear1 = nn.Linear(4, 8, bias_attr=True)
        
        self.Linear2 = nn.Linear(2, 4, bias_attr=True)
        self.Linear3 = nn.Linear(12, 1, bias_attr=True)
        
    def forward(self, X, space_data):
        ## time series cnn
        cnn_out1 = self.cnnLayer1(X)
        res_out1 = self.res1(X)
        output1 = cnn_out1 + res_out1
        
        output2 = self.cnnLayer2(output1)
        #output2 = output2.reshape(output2.shape[0], 288, -1)
        output2 = paddle.reshape(output2, ( output2.shape[0], 288, -1))
        out1 = self.Linear1(output2)
        
        ## space cnn
        outputsp = self.cnnLayer3(space_data)
        #output3 = output3.reshape(output3.shape[0], 288, -1)
        outputsp = paddle.reshape(outputsp, ( outputsp.shape[0], 288, -1))
        out2 = self.Linear2(outputsp)

        ## combine
        final_input = out1
        final_input = paddle.concat(x=[final_input, out2], axis=2)
        final_output = self.Linear3(final_input)
        #final_output = self.Linear4(final_output)
       
        return final_output
    
## GRU model with spatial-temporal information
class GRU(nn.Layer):
    def __init__(self):
        super(GRU, self).__init__()
        
        self.gru = nn.GRU(input_size=4, hidden_size=48, num_layers=2)
        self.dropout = nn.Dropout(0.1)

        self.Linear = nn.Linear(48+2, 1, bias_attr=True)

        self.cnnLayer = nn.Sequential(
        nn.Conv2D(1, 64, kernel_size=3, stride=1, padding=1), 
        nn.BatchNorm2D(64),
        nn.GELU(),
        nn.MaxPool2D((3,3)))

        
    def forward(self, X, space_data):
        z = paddle.zeros([X.shape[0], 144, X.shape[1]], dtype="float32")
        x = paddle.concat((X.transpose([0,2,1]),z), axis=1)
        out1, _ = self.gru(x)
        ou1 = self.dropout(out1)

        cnn_out = self.cnnLayer(space_data)
        cnn_out = paddle.reshape(cnn_out, (cnn_out.shape[0], 288, -1))

        out2 = self.Linear(paddle.concat((out1, cnn_out), 2))

       
        return out2