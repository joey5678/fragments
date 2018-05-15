#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
from numpy.lib.format import open_memmap
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

"""
key: work_dir , value: ./work_dir/temp  . 
key: config , value: config/st_gcn/kinetics-skeleton/test.yaml  . 
key: phase , value: test  . 
key: save_score , value: False  . 
key: seed , value: 1  . 
key: log_interval , value: 100  . 
key: save_interval , value: 10  . 
key: eval_interval , value: 5  . 
key: print_log , value: True  . 
key: show_topk , value: [1, 5]  . 
key: feeder , value: st_gcn.feeder.Feeder_kinetics  . 
key: num_worker , value: 128  . 
key: train_feeder_args , value: {}  . 
key: test_feeder_args , value: {'mode': 'test', 'data_path': './data/kinetics-skeleton/kinetics_val', 'label_path': './data/kinetics-skeleton/kinetics_val_label.json', 'window_size': 150}  . 
key: model , value: st_gcn.net.ST_GCN  . 
key: model_args , value: {'num_class': 400, 'channel': 3, 'window_size': 150, 'num_person': 2, 'num_point': 18, 'dropout': 0, 'graph': 'st_gcn.graph.Kinetics', 'graph_args': {'labeling_mode': 'spatial'}, 'mask_learning': True, 'use_data_bn': True}  . 
key: weights , value: ./model/kinetics-st_gcn.pt  . 
key: ignore_weights , value: []  . 
key: base_lr , value: 0.01  . 
key: step , value: [20, 40, 60]  . 
key: device , value: 0  . 
key: optimizer , value: SGD  . 
key: nesterov , value: False  . 
key: batch_size , value: 256  . 
key: test_batch_size , value: 64  . 
key: start_epoch , value: 0  . 
key: num_epoch , value: 80  . 
key: weight_decay , value: 0.0005  .


"""

base_lr = 0.01
weight_decay = 0.0005
model_args = {'num_class': 60, 
                'channel': 3, 
                'window_size': 300, 
                'num_person': 2, 
                'num_point': 25, 
                'dropout': 0, 
                'graph': 'st_gcn.graph.NTU_RGB_D', 
                'graph_args': {'labeling_mode': 'spatial'}, 
                'mask_learning': True, 
                'use_data_bn': True
                }

weight_file = './model/ntuxview-st_gcn.pt'

def load_model():
    output_device = 0
    Model = import_class('st_gcn.net.ST_GCN')
    model = Model(**model_args).cuda(output_device)
    weights = torch.load(weight_file)
    weights = OrderedDict(
            [[k.split('module.')[-1],
                v.cuda(output_device)] for k, v in weights.items()])
    try:
        model.load_state_dict(weights)
    except:
        print('Can not find these weights:')
        raise ValueError()
    # for idx, param in enumerate(model.modules()):
    #     print(idx, '----->', param)
    return model


def load_optimizer(optimizer_type):
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(
            model_args,
            lr=base_lr,
            momentum=0.9,
            nesterov=False,
            weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(
            model_args,
            lr=base_lr,
            weight_decay=weight_decay)
    else:
        raise ValueError()

    return optimizer



"""
##############Load Data#####################
"""
feeder = 'st_gcn.feeder.Feeder'
test_feeder_args = { 
                    'data_path': './data/actions-10/xview/train_data.npy', 
                    'label_path': './data/actions-10/xview/train_label.pkl', 
                    'window_size': 300}
test_batch_size = 1
num_worker = 1

def load_data():
    Feeder = import_class(feeder)
    data_loader = dict()
    data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**test_feeder_args),
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_worker)

    return data_loader

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def eval(model, data_loader, output_device=0):
    model.eval()

    part = 'train'
    out_path = './data/actions-10/features'

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(data_loader['test']), 1, 256, 1))
    label_fp = open('{}/{}_label.txt'.format(out_path, part),'+w')
    for i, (data, label) in enumerate(data_loader['test']):

        data = Variable(
            data.float().cuda(output_device),
            requires_grad=False,
            volatile=True)
        label = Variable(
                    label.long().cuda(output_device),
                    requires_grad=False,
                    volatile=True)
        label_int = int(label.data.cpu().numpy())
        print("label is {}".format(label_int))
        output = model(data)
        label_fp.write(str(label_int)+'\n')
        np_output = output.data.cpu().numpy()
        print(np_output.shape)
        fp[i,:,:,:] = np_output
        # print(np_output[0][:,0])

        # _label_str = str(label.data.cpu().numpy())
        # _pred_label_str = str(torch.max(output, 1)[1].data.cpu().numpy())
        # _pred_score = str(torch.max(output, 1)[0].data.cpu().numpy())

        # print("--True Label -- Pred Label -- Pred Score--")
        # print(_label_str + " -- " + _pred_label_str + " -- " + _pred_score)
    label_fp.close()



if __name__ == '__main__':
    _model = load_model()
    _dloader = load_data()
    eval(_model, _dloader, 0)    
