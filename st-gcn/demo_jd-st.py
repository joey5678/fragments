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
from tools.ntu_read_skeleton import read_xyz
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

model_args = {'num_class': 60, 
                'channel': 3, 
                'window_size': 300, 
                'num_person': 2, 
                'num_point': 25, 
                'dropout': 0, 
                'graph': 'st_gcn.graph.NTU_RGB_D', 
                'graph_args': {'labeling_mode': 'spatial'}, 
                'mask_learning': True, 
                'use_data_bn': False
}

weight_file = './model/ntuxview-st_gcn.pt'
feeder = 'st_gcn.feeder.Feeder'
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


"""
##############Load Data#####################
"""
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


def eval(model, data_loader, output_device=0, dstype='train'):

    part = 'train' if dstype == "train" else 'test'
    out_path = './data/{}/features'.format(dataset)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(data_loader['test']), 1, 256, 1))
    label_fp = open('{}/{}_label.txt'.format(out_path, part),'+w')
    for i, (data, label, sample_name) in enumerate(data_loader['test']):

        data = Variable(
            data.float().cuda(output_device),
            requires_grad=False,
            volatile=True)
        label = Variable(
                    label.long().cuda(output_device),
                    requires_grad=False,
                    volatile=True)
        label_int = int(label.data.cpu().numpy())

        label_fp.write(sample_name[0] + ", " + str(label_int)+'\n')
        ddata = data.data.cpu().numpy()
        label_fp.write(" ".join(list(map(lambda x: str(x), ddata.flatten()[:10]))) + '\n')
        output = model(data)
        np_output = output.data.cpu().numpy()
        label_fp.write(" ".join(list(map(lambda x: str(x), np_output.flatten()[:10]))) + '\n')

        fp[i,:,:,:] = np_output

    label_fp.close()


def eval_single(model, sk_file, output_device=0):
    data = read_xyz(sk_file, max_body=2, num_joint=25)
    data = data.reshape((1,) + data.shape)
    print(data.shape)

    # data = Variable(
    #     torch.Tensor(data).float().cuda(output_device),
    #     requires_grad=False,
    #     volatile=True)
        
    # ddata = data.data.cpu().numpy()
    print(">> data : >>>")
    print(data.flatten()[-10:])
    print(data.shape[2])
    myddata = np.zeros(shape=(1, 3, 300, 25, 2))
    myddata[:,:,0:data.shape[2],:,:] = data
    print(myddata.shape)
    mydata = Variable(
        torch.Tensor(myddata).float().cuda(output_device),
        requires_grad=False,
        volatile=True)
    output = model(mydata)
    np_output = output.data.cpu().numpy()
    print(">> result : >>")
    print(np_output.flatten()[:10])
       



if __name__ == '__main__':
    run_batch = False
    _model = load_model()
    _model.eval()

    if run_batch:
        datasets = ["actions-10", "actions-20"]
        dstypes = ['train', 'val']
        # dstype = "train"
        for dataset in datasets:
            for dstype in dstypes:
                test_feeder_args = { 
                    'data_path': './data/{}/xview/{}_data.npy'.format(dataset, dstype), 
                    'label_path': './data/{}/xview/{}_label.pkl'.format(dataset, dstype), 
                    'window_size': 300}
                _dloader = load_data()
                eval(_model, _dloader, 0, dstype)    
    else:
        datasets = ['actions-10', 'actions-20']
        dstype = 'val' 
        sk_names = ['S011C002P019R001A006.skeleton']
        for sk_name in sk_names:
            print("----------------------------")
            print(sk_name)
            for dataset in datasets:
                print("================")
                print(dataset)
                sk_file = os.path.join('/home/joey/ai/dl/data-set/actions/{}'.format(dataset), sk_name)
                eval_single(_model, sk_file)

