import os
from stat import S_IREAD, S_IRGRP, S_IROTH
from collections import OrderedDict

import torch

import models.networks as networks
import models.modules.block as B
from models.modules.loss import Loss
from models.modules.util import get_network_description, load_network, save_network
from models.base_model import BaseModel

class SRResNetModel(BaseModel):


    def __init__(self, opt):
        super(SRResNetModel, self).initialize(opt)
        assert opt["is_train"]

        self.input_L = self.Tensor()
        self.input_H = self.Tensor()

        self.device = opt['device']
        if self.device == 'cuda':
            self.criterion = Loss(opt["train"].get("criterion"))().cuda(opt["gpu_ids"][0])
            self.netG = networks.define_G(opt).to(torch.device('cuda'))

        # Load pretrained_models
        self.load_path_G = opt["path"].get("pretrain_model_G")
        self.load()

        # if opt["train"].get("lr_scheme") == 'multi_steps':
        #     self.lr_steps = self.opt["train"].get("lr_steps")
        #     self.lr_gamma = self.opt["train"].get("lr_gamma")

        self.optimizers = []

        self.lr_G = opt["train"].get('lr_G')
        self.weight_decay_G = opt["train"].get("weight_decay_G") if opt["train"].get("weight_decay_G") else 0.0
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr_G, weight_decay=self.weight_decay_G)
        self.optimizers.append(self.optimizer_G)

        print('---------- Model initialized -------------')
        self.write_description()
        print('-----------------------------------------------')

    def name(self):
        return 'SRResNetModel'

    def feed_data(self, data):
        input_H = data['H']
        input_L = data['L']
        self.real_H = input_H.requires_grad_().to(torch.device('cuda'))
        self.real_L = input_L.requires_grad_().to(torch.device('cuda'))

    def forward_G(self):
        self.fake_H = self.netG(self.real_L)

    def backward_G(self):
        self.loss = self.criterion(self.fake_H, self.real_H)
        self.loss.backward()

    def optimize_parameters(self, step):
        # G
        self.forward_G()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def val(self):
        self.fake_H = self.netG(self.real_L)

    def get_current_losses(self):
        out_dict = OrderedDict()
        out_dict['loss'] = self.loss.item()
        return out_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['low-resolution'] = self.real_L.data[0]
        out_dict['super-resolution'] = self.fake_H.data[0]
        out_dict['ground-truth'] = self.real_H.data[0]
        return out_dict

    def write_description(self):
        total_n = 0
        message = ''
        s, n = get_network_description(self.netG)
        print('Number of parameters in G: %d' % n)
        message += '-------------- Generator --------------\n' + s + '\n'
        total_n += n

        network_path = os.path.join(self.save_dir, 'network.txt')
        with open(network_path, 'w') as f:
            f.write(message)
        os.chmod(network_path, S_IREAD|S_IRGRP|S_IROTH)

    def load(self):
        if self.load_path_G is not None:
            print('loading model for G [%s] ...' % self.load_path_G)
            load_network(self.load_path_G, self.netG)

    def save(self, iter_label):
        save_network(self.save_dir, self.netG, 'G', iter_label, self.opt["gpu_ids"])

    def update_learning_rate(self, step=None, scheme=None):
        if scheme == 'multi_steps':
            if step in self.lr_steps:
                for optimizer in self.optimizers:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * self.lr_gamma
                print('learning rate switches to next step.')

    def train(self):
        self.netG.train()

    def eval(self):
        self.netG.eval()
