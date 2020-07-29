import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init

from models.modules.loss import Loss
from models.modules.util import get_network_description, load_network, save_network

class ESPCNModel(nn.Module):
    def __init__(self, opt):
        upscale_factor = opt['upscale_factor']
        super(ESPCNModel, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

        self.criterion = Loss(opt["train"].get("criterion"))()
        self.lr = opt['train'].get('lr')
        self.weight_decay = opt["train"].get("weight_decay") if opt["train"].get("weight_decay") else 0.0
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.save_dir = opt["path"]["trained_models"]
        self.opt = opt

        self.device = opt.get('device')
        if self.device == 'cuda':
            self.to(torch.device('cuda'))

    def name(self):
        return 'ESPCNModel'

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

    def feed_data(self, data):
        input_H = data['H']
        input_L = data['L']
        self.real_H = input_H.requires_grad_().to(torch.device('cuda'))
        self.real_L = input_L.requires_grad_().to(torch.device('cuda'))

    def forward(self):
        self.fake_H = self.relu(self.conv1(self.real_L))
        self.fake_H = self.relu(self.conv2(self.fake_H))
        self.fake_H = self.relu(self.conv3(self.fake_H))
        self.fake_H = self.pixel_shuffle(self.conv4(self.fake_H))
        return self.fake_H

    def backward(self):
        self.loss = self.criterion(self.fake_H, self.real_H)
        self.loss.backward()

    def optimize_parameters(self, step):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_current_losses(self):
        out_dict = OrderedDict()
        out_dict['loss'] = self.loss.item()
        return out_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['low-resolution'] = self.real_L.data
        out_dict['super-resolution'] = self.fake_H.data
        out_dict['ground-truth'] = self.real_H.data
        return out_dict

    def write_description(self):
        total_n = 0
        message = ''
        s, n = get_network_description(self.netG.module)
        print('Number of parameters in ESPCN: %d' % n)
        # message += '-------------- Generator --------------\n' + s + '\n'
        # total_n += n
        #
        # network_path = os.path.join(self.save_dir, 'network.txt')
        # with open(network_path, 'w') as f:
        #     f.write(message)
        # os.chmod(network_path, S_IREAD|S_IRGRP|S_IROTH)

    # def load(self):
    #     if self.load_path_G is not None:
    #         print('loading model for G [%s] ...' % self.load_path_G)
    #         load_network(self.load_path_G, self.netG)

    def save(self, iter_label, network_label='ESPCN'):
        save_network(self.save_dir, self, network_label, iter_label, self.opt["gpu_ids"], self.optimizer)