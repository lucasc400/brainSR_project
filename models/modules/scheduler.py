import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# set up scheduler
def create_scheduler(opt, model):
    opt_scheduler = opt['train'].get('scheduler')
    if opt_scheduler is None:
        return
    
    if opt_scheduler["name"] == "ReduceLROnPlateau":
        return ReduceLROnPlateau(model.optimizer, 
                                 mode = "max", 
                                 factor = opt_scheduler.get("factor"), 
                                 threshold = opt_scheduler.get("threshold"), 
                                 patience = opt_scheduler.get("patience"),
                                 cooldown = opt_scheduler.get("cooldown"))