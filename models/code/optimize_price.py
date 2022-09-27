import os
import time
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

# custom import
from utils import SfparkDataset, SeattleDataset
import parse
import models

# MALI integrator
from TorchDiffEqPack import odesolve_adjoint_sym12

args = parse.parse_args()
batch_size = args.batch_size
manual_seed = args.seed
experiment_id = str(time.time()).replace(".", "")
seq_len = args.seq_len
initial_coeff_value = args.initial_coeff_value
model = args.model

np.random.seed(manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.random.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)

# Load data
if args.dataset == 'sfpark':
    data = SfparkDataset(seq_len=seq_len)
elif args.dataset == 'seattle':
    data = SeattleDataset(seq_len=seq_len)
else:
    print('Not available dataset!')
    
train_len, test_len = len(data.X_train), len(data.X_test)
dim_out = data.y_train.shape[1]  # output dimension (number of blocks)
dim_in = (
    data.X_train.shape[2] - dim_out
)  # input dimension (includes short term occupancy and long term occupancy)
train = data_utils.TensorDataset(
    data.X_train, data.y_train
)
train_loader = data_utils.DataLoader(
    train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
)


class InputBlock(nn.Module):
    """
    The first ODE block for the single sequence input (1:1)
    We use a ODE for short-term occupancy and FC-layer for long-term occupancy
    """

    def __init__(self, odefunc, short_feature_func, long_feature_func, apply_price=False):
        super().__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.coeff = torch.nn.Parameter(torch.ones(dim_out) * initial_coeff_value)
        self.bias = torch.nn.Parameter(torch.zeros(dim_out))
        self.short_feature_func = short_feature_func
        self.long_feature_func = long_feature_func
        self.apply_price = apply_price

    def forward(self, x):
        # configure training options
        options = {}
        options.update({"method": "sym12async"})
        options.update({"h": None})
        options.update({"t0": 0.0})
        options.update({"t1": 1.0})
        options.update({"rtol": args.tol})
        options.update({"atol": args.tol})
        options.update({"print_neval": False})
        options.update({"neval_max": 1000000})
        options.update({"t_eval": None})
        options.update({"interpolation_method": "cubic"})
        options.update({"regenerate_graph": False})
        if len(x.shape) != 2:
            if x.shape[1] > 1: # multi-seq
                price = x[:, :, dim_in:][:, 0, :].squeeze(1)
                short_out = x[:, :, :dim_out]  # short term features
                long_out = x[:, :, dim_out:dim_in][:, 0, :].squeeze(1)  # long term features
            else: # single-seq
                price = x[:, :, dim_in:].squeeze(1)
                short_out = x[:, :, :dim_out]  # short term features
                long_out = x[:, :, dim_out:dim_in].squeeze(1)  # long term features
        else:
            price = x[:, dim_in:]
            short_out = x[:, :dim_out].unsqueeze(1)  # short term features
            long_out = x[:, dim_out:dim_in]  # long term features            
        short_out = self.short_feature_func(short_out)
        if len(short_out.shape) != 2: 
            short_out = torch.squeeze(short_out)
        for layer in self.long_feature_func:
            long_out = layer(long_out)
        concat_out = torch.cat([short_out, long_out], dim=1)
        out = odesolve_adjoint_sym12(self.odefunc, concat_out, options=options)
        final_out = out[:, :dim_out]
        if self.apply_price:
            final_out = final_out - (torch.mul(self.coeff, price) + self.bias)
        return final_out


class InputBlock2(nn.Module):
    """
    The first ODE block for the single sequence input (1:1)
    We use a ODE for short-term occupancy and FC-layer for long-term occupancy
    """

    def __init__(self, odefunc, short_feature_ode, long_feature_func, apply_price=False):
        super().__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.coeff = torch.nn.Parameter(torch.ones(dim_out) * initial_coeff_value)
        self.bias = torch.nn.Parameter(torch.zeros(dim_out))
        self.short_feature_ode = short_feature_ode
        self.long_feature_func = long_feature_func
        self.apply_price = apply_price

    def forward(self, x):
        # configure training options
        options = {}
        options.update({"method": "sym12async"})
        options.update({"h": None})
        options.update({"t0": 0.0})
        options.update({"t1": 1.0})
        options.update({"rtol": args.tol})
        options.update({"atol": args.tol})
        options.update({"print_neval": False})
        options.update({"neval_max": 1000000})
        options.update({"t_eval": None})
        options.update({"interpolation_method": "cubic"})
        options.update({"regenerate_graph": False})
        if len(x.shape) != 2:
            if x.shape[1] > 1: # multi-seq
                price = x[:, :, dim_in:][:, 0, :].squeeze(1)
                short_out = x[:, :, :dim_out]  # short term features
                long_out = x[:, :, dim_out:dim_in][:, 0, :].squeeze(1)  # long term features
            else: # single-seq
                price = x[:, :, dim_in:].squeeze(1)
                short_out = x[:, :, :dim_out]  # short term features
                long_out = x[:, :, dim_out:dim_in].squeeze(1)  # long term features
        else:
            price = x[:, dim_in:]
            short_out = x[:, :dim_out].unsqueeze(1)  # short term features
            long_out = x[:, dim_out:dim_in]  # long term features            
        short_out = self.short_feature_ode(short_out)
        if len(short_out.shape) != 2:
            short_out = torch.squeeze(short_out)
        for layer in self.long_feature_func:
            long_out = layer(long_out)
        concat_out = torch.cat([short_out, long_out], dim=1)
        out = odesolve_adjoint_sym12(self.odefunc, concat_out, options=options)
        final_out = out[:, :dim_out]
        if self.apply_price:
            final_out = final_out - (torch.mul(self.coeff, price) + self.bias)
        return final_out


class ReverseODEBlock(nn.Module):
    def __init__(self, odefunc):
        super().__init__()
        self.odefunc = odefunc

    def forward(self, x):
        # configure training options
        options = {}
        options.update({"method": "sym12async"})
        options.update({"h": None})
        options.update({"t0": 1.0})
        options.update({"t1": 0.0})
        options.update({"rtol": args.tol})
        options.update({"atol": args.tol})
        options.update({"print_neval": False})
        options.update({"neval_max": 1000000})
        options.update({"t_eval": None})
        options.update({"interpolation_method": "cubic"})
        options.update({"regenerate_graph": False})
        out = odesolve_adjoint_sym12(self.odefunc, x, options=options)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def save(array, file_name):
    array = array.cpu().detach().numpy()
    df = pd.DataFrame(array)
    df.to_csv(file_name)
    return df


if __name__ == "__main__":
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )

    if model == 'node':
        if args.dataset == 'sfpark':
            short_feature_ode = models.ODEBlock(models.ODEfunc(dim_out, dim_out, dim_out), args.tol)
        else:
            short_feature_func = models.ODEBlock(models.ODEfunc(dim_out, dim_out, dim_out), args.tol)
    else:
        print('Not available model!')

    long_feature_func = nn.ModuleList(
        [nn.Linear((dim_in - dim_out), (dim_in - dim_out))] * 3
    )
    apply_price = 1
    if args.dataset == 'sfpark':
        input_ode_block = [
            InputBlock2(
                models.InputODEfunc(dim_in, dim_in//2, dim_out, device), short_feature_ode, long_feature_func, apply_price
            )
        ]
    else:
        input_ode_block = [
            InputBlock(
                models.InputODEfunc(dim_in, dim_in//2, dim_out, device), short_feature_func, long_feature_func, apply_price
            )
        ]
    ode_blocks = [models.ODEBlock(models.ODEfunc(dim_out, dim_out, dim_out), args.tol) for _ in range(args.nblocks)]
    ode_blocks = input_ode_block + ode_blocks
    model = nn.Sequential(*ode_blocks).to(device)

    print("\n----- Optimization has started! -----\n")

    # Load pre-trained prediction model
    SAVE_PATH = "./experiment/"

    if args.dataset == 'sfpark':
        PATH = SAVE_PATH + "model_sfpark.pth"    
    elif args.dataset == 'seattle':
        PATH = SAVE_PATH + "model_seattle.pth"
    else:
        print('Not available dataset!')

    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()   

    input_block = model[0]
    input_ode_func = input_block.odefunc
    if args.dataset == 'sfpark':
        short_feature_ode = input_block.short_feature_ode
    elif args.dataset == 'seattle':
        short_feature_func = input_block.short_feature_func
    long_feature_func = input_block.long_feature_func
    coeff = input_block.coeff
    bias = input_block.bias
    apply_price = 0

    test_x = data.X_test.squeeze(1).to(device)
    test_y = data.y_test.to(device)

    if args.dataset == 'sfpark':
        input_ode_block = InputBlock(
            input_ode_func, short_feature_ode, long_feature_func, apply_price
            )
    else:
        input_ode_block = InputBlock(
            input_ode_func, short_feature_func, long_feature_func, apply_price
            )      

    price = test_x[:, dim_in:]
    initial_occ = input_ode_block(test_x)

    # Give an ideal occupancy
    ideal_occ = 0.7
    ideal_y = torch.tensor([[ideal_occ] * dim_out] * len(test_y)).to(device)
    z1 = ideal_y    

    start_time = time.time()
    print("Start time: {}".format(start_time))

    # Descending order bc we run model in reverse mode
    for block_idx in range(args.nblocks, 0, -1):
        ode_func = model[block_idx].odefunc
        z0 = ReverseODEBlock(ode_func)(z1)
        z1 = z0

    optimized_price = torch.div((initial_occ - z0), coeff) - bias
   
    end_time = time.time()
    test_size = test_y.shape[0] * test_y.shape[1]
    print("\n----- Optimization has finished! -----\n")
    print("End time: {}".format(end_time))
    print("Total runtime: {}".format(end_time - start_time))
    print("Average runtime: {}".format((end_time - start_time) / test_size))

    if args.dataset == 'sfpark':
        optimized_price = torch.clamp(optimized_price, min=0.25, max=34.5)
    elif args.dataset == 'seattle':
         optimized_price = torch.clamp(optimized_price, min=0.5, max=3.0)
    else:
        print('Not available dataset!')

    optimized_x = torch.cat([test_x[:, :dim_in], optimized_price], dim=1)
    optimized_y = model(optimized_x)

    print("\n-- Optimization Performance --")
    print("Total test data size: {}".format(test_size))
    print("-- ours --")
    print("occupancy rate exceed threshold {}: {} ({})".format(round(ideal_occ+0.1,2), round(torch.sum(optimized_y > ideal_occ+0.1).item()/test_size,4), torch.sum(optimized_y > ideal_occ+0.1)))
    print("occupancy rate exceed threshold {}: {} ({})".format(round(ideal_occ+0.05,2), round(torch.sum(optimized_y > ideal_occ+0.05).item()/test_size,4), torch.sum(optimized_y > ideal_occ+0.05)))
    print("occupancy rate exceed threshold {}: {} ({})".format(ideal_occ, round(torch.sum(optimized_y > ideal_occ).item()/test_size,4), torch.sum(optimized_y > ideal_occ)))
    print("-- ground truth --")
    print("occupancy rate exceed threshold {}: {} ({})".format(round(ideal_occ+0.1,2), round(torch.sum(test_y > ideal_occ+0.1).item()/test_size,4), torch.sum(test_y > ideal_occ+0.1)))
    print("occupancy rate exceed threshold {}: {} ({})".format(round(ideal_occ+0.05,2), round(torch.sum(test_y > ideal_occ+0.05).item()/test_size,4), torch.sum(test_y > ideal_occ+0.05)))
    print("occupancy rate exceed threshold {}: {} ({})".format(ideal_occ, round(torch.sum(test_y > ideal_occ).item()/test_size,4), torch.sum(test_y > ideal_occ)))

    print("\n-- initial occupancy --\n", initial_occ)
    print("\n-- z0 --\n", z0)
    print("\n-- coefficient --\n", coeff)
    print("\n-- bias --\n", bias)
    print("\n-- optimized price --\n", optimized_price)