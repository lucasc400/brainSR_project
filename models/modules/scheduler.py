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
        factor = opt_scheduler.get("factor") if opt_scheduler.get("factor") is not None else 0.2
        threshold = opt_scheduler.get("threshold") if opt_scheduler.get("threshold") is not None else 0.05
        patience = opt_scheduler.get("patience") if opt_scheduler.get("patience") is not None else 5
        cooldown = opt_scheduler.get("cooldown") if opt_scheduler.get("cooldown") is not None else 0
        return ReduceLROnPlateau(model.optimizer, 
                                 mode = "max", 
                                 factor = factor, 
                                 threshold = threshold, 
                                 patience = patience,
                                 cooldown = cooldown)