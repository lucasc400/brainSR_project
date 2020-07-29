import os
import torch
from models.base_model import BaseModel
from models.modules.loss import Loss
from models.modules.sft_net import SFTLayer, ResBlock_SFT, SFT_Net
from models.modules.util import get_network_description, load_network, save_network
from stat import S_IREAD, S_IRGRP, S_IROTH


class SFTResNetModel(BaseModel):
    def __init__(self, opt):
        super(SFTResNetModel, self).initialize(opt)

        self.input_L = self.Tensor()
        self.input_H = self.Tensor()

        self.device = torch.device("cuda") if opt["device"] == "cuda" else torch.device("cpu")
        if self.device.type == "cuda":
            self.criterion = Loss(opt["train"].get("criterion"))().cuda(opt["gpu_ids"][0])
        self.sft_net = SFT_Net().to(self.device)

        # Load pretrained_models
        self.load_path_sftnet = opt["path"].get("pretrained_model_sft")

        self.optimizers = []
        self.lr = opt["train"].get("lr")
        self.weight_decay = opt["train"].get("weight_decay") if opt["train"].get("weight_decay_G") else 0.0

        self.optimizers = []
        self.optimizer = torch.optim.Adam(self.sft_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizers.append(self.optimizer)

        print('---------- Model initialized -------------')
        self.write_description()
        print('-----------------------------------------------')


    def write_description(self):
        total_n = 0
        message = ''
        s, n = get_network_description(self.sft_net)
        print('Number of parameters in G: %d' % n)
        message += '-------------- Generator --------------\n' + s + '\n'
        total_n += n

        network_path = os.path.join(self.save_dir, 'network.txt')
        with open(network_path, 'w') as f:
            f.write(message)
        os.chmod(network_path, S_IREAD|S_IRGRP|S_IROTH)

    def name(self):
        return 'SRSFTResNet'

    def feed_data(self, data):
        input_H = data['H']
        input_L = data['L'] # [0]:image (LR space), [1]:label (HR space)
        self.real_H = input_H.requires_grad_().to(self.device)
        self.real_L = list(map(lambda t: t.requires_grad_().to(self.device), input_L))

    def forward(self):
        self.fake_H = self.sft_net.forward(self.real_L)

    def backward(self):
        self.loss = self.criterion(self.fake_H, self.real_H)
        self.loss.backward()

    def optimize_parameters(self, step):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def val(self):
        self.fake_H = self.sft_net.forward(self.real_L)

    def get_current_losses(self):
        out_dict = OrderedDict()
        out_dict['loss'] = self.loss.item()
        return out_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['low-resolution'] = self.real_L[0].data[0]
        out_dict['label'] = self.real_L[1].data[0]
        out_dict['super-resolution'] = self.fake_H.data[0]
        out_dict['ground-truth'] = self.real_H.data[0]
        return out_dict

    def train(self):
        self.sft_net.train()

    def eval(self):
        self.sft_net.eval()

    def load(self):
        if self.load_path_sftnet is not None:
            print('loading model for SFTNET [%s] ...' % self.load_path_sftnet)
            load_network(self.load_path_sftnet, self.sft_net)

    def save(self, iter_label, network_label='SFT'):
        save_network(self.save_dir, self.sft_net, network_label, iter_label, self.opt["gpu_ids"])

