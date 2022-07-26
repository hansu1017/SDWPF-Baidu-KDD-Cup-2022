import paddle
import numpy as np
import pandas as pd

def pred_test(model, test_loader):
    model.eval()

    pred_list = []
    label_list = []

    for seq, label in test_loader:
        seq = paddle.reshape(seq, (-1, 4, 12, 12))
        seq = paddle.cast(seq, dtype='float32')
        label = paddle.cast(label, dtype='int64')

        pred = model(seq)

        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    return pred_list
    
def pred_cnn(model, test_loader):
    model.eval()
    size = 11
    seq_len = size * size
    pred_list = []
    label_list = []

    for seq, space_data, label in test_loader:
        seq = paddle.reshape(seq, (-1, 4, size, size))
        seq = paddle.cast(seq, dtype='float32')
        space_data = paddle.reshape(space_data, ( -1,1,11,11 ))
        space_data = paddle.cast(space_data, dtype='float32')

        pred = model(seq, space_data)

        pred_list.extend(pred.squeeze().cpu().detach().numpy())

    return pred_list


def pred_gru(model, test_loader):
    model.eval()

    pred_list = []

    for seq, space_data, label in test_loader:
        seq = paddle.cast(seq, dtype='float32')
        space_data = paddle.reshape(space_data, (-1,1,11,11))
        space_data = paddle.cast(space_data, dtype='float32')
        label = paddle.cast(label, dtype='float32')

        pred = model(seq, space_data)

        pred_list.extend(pred.squeeze().cpu().detach().numpy())

    return pred_list