import os
import torch
import torch.nn as nn

# helper printing function
def get_network_description(network):
    s = str(network)
    n = sum(map(lambda x: x.numel(), network.parameters()))
    return s, n

# helper saving function
def save_network(save_dir, network, network_label, iter_label, gpu_ids, optimizer, scheduler):
    save_filename = '%s_%s.pth' % (iter_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    output_dict = dict(iteration=iter_label,
                       model=network.cpu().state_dict(),
                       optimizer=optimizer.state_dict(),
                       )
    if scheduler is not None:
        output_dict['scheduler'] = scheduler.state_dict()
    torch.save(output_dict, save_path)
    network.cuda(gpu_ids[0])

# helper loading function
def load_network(load_path, network):
    network.load_state_dict(torch.load(load_path),False)
