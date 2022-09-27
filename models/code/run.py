import os
import time
import random
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

# custom import
from utils import SfparkDataset, SeattleDataset, load_adj, get_normalized_adj
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

    def __init__(self, odefunc, short_feature_func, long_feature_func, apply_price=True):
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


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(
    batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates
):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def get_r2_score(output, target):
    target_mean = torch.mean(target)
    sst = torch.sum((target - target_mean) ** 2)
    sse = torch.sum((target - output) ** 2)
    r2 = 1 - sse / sst
    return r2


def get_sse(model, test_x, test_y):
    preds = model(test_x)
    sum_squared_error = torch.sum((preds - test_y) ** 2)
    return sum_squared_error


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def write_to_tensorboard(writer, itr, batches_per_epoch, metrics, row):
    for key, metric in metrics.items():
        writer.add_scalar(str(row) + "/" + key, metric, itr // batches_per_epoch)


if __name__ == "__main__":
    makedirs(args.save)
    print(args)

    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )

    if model == 'node':
        short_feature_func = models.ODEBlock(models.ODEfunc(dim_out, dim_out, dim_out), args.tol)
    elif model == 'rnn':
        short_feature_func =  models.RNN(input_dim=dim_out, hidden_dim=500, num_layers=2, output_dim=dim_out)
    elif model == 'lstm':
        short_feature_func =  models.LSTM(input_dim=dim_out, hidden_dim=500, num_layers=2, output_dim=dim_out)
    elif model == 'gru':
        short_feature_func =  models.GRU(input_dim=dim_out, hidden_dim=500, num_layers=2, output_dim=dim_out)
    elif model == 'agcrn':
        num_nodes = dim_out
        short_feature_func = models.AGCRN(args=args, num_nodes=num_nodes, input_dim=args.input_dim, 
                            rnn_units=args.rnn_units, output_dim=args.output_dim, horizon=args.agcrn_horizon, num_layers=args.num_layers)
    elif model == 'dcrnn':
        num_nodes = dim_out
        adj_matrix = load_adj(args.dataset)
        adj = get_normalized_adj(adj_matrix).to(device)
        short_feature_func = models.DCRNNModel(args=args, supports=[adj], num_node=num_nodes, input_dim=args.input_dim, 
                            hidden_dim=args.rnn_units, out_dim=args.output_dim, order=1, num_layers=args.num_layers )
    elif model == 'stgcn':
        num_nodes = dim_out
        adj_matrix = load_adj(args.dataset)
        adj = get_normalized_adj(adj_matrix).to(device)
        adj = adj.to_sparse()
        short_feature_func = models.STGCN(args=args, num_node=num_nodes, input_dim=args.input_dim, 
                            hidden_dim=args.rnn_units, out_dim=args.output_dim, horizon=1, adj=adj)
    else:
        print('Not available model!')

    long_feature_func = nn.ModuleList(
        [nn.Linear((dim_in - dim_out), (dim_in - dim_out))] * 3
    )
    apply_price = args.price
    input_ode_block = [
        InputBlock(
            models.InputODEfunc(dim_in, dim_in//2, dim_out, device), short_feature_func, long_feature_func, apply_price
        )
    ]

    ode_blocks = [models.ODEBlock(models.ODEfunc(dim_out, dim_out, dim_out), args.tol) for _ in range(args.nblocks)]
    ode_blocks = input_ode_block + ode_blocks
    model = nn.Sequential(*ode_blocks).to(device)

    print("Experiment ID: {}".format(experiment_id))
    print("-" * 50)
    print(model)
    print("-" * 50)
    print("The sequence length: {}".format(args.seq_len))
    print("The dimension of input X: {}".format(dim_in))
    print("Train dataset size: {} & Test dataset size: {}".format(train_len, test_len))
    print("Number of parameters: {}".format(count_parameters(model)))
    print("-" * 50)

    criterion = nn.MSELoss().to(device)

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size,
        batch_denom=128,
        batches_per_epoch=batches_per_epoch,
        boundary_epochs=[1000],  # turned off learning rate decay
        decay_rates=[1, 0.1],
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.decay
    )
    min_mse = float("inf")
    batch_time_meter = RunningAverageMeter()
    end = time.time()

    writer = SummaryWriter(
        "runs/{}/{}".format(os.path.basename(__file__), experiment_id)
    )

    val_y_mean = torch.mean(data.y_test)  # for r2 calculation
    val_sst = torch.sum((data.y_test - val_y_mean) ** 2)

    test_x = data.X_test.squeeze(1).to(device)
    test_y = data.y_test.to(device)

    epoch_loss = 0
    for itr in range(args.nepochs * batches_per_epoch):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = criterion(output, y)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
        batch_time_meter.update(time.time() - end)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                # train metrics
                train_mse = (epoch_loss / batches_per_epoch)
                epoch_loss = 0
                train_rmse = math.sqrt(train_mse)
                train_r2 = get_r2_score(output, y)
                # test metrics
                val_sse = get_sse(model, test_x, test_y)
                val_mse = val_sse / (test_len * dim_out)
                val_rmse = math.sqrt(val_mse)
                val_r2 = 1 - val_sse / val_sst

                # tensorboard logging
                train_metrics = {
                    "train mse": train_mse,
                    "train rmse": train_rmse,
                    "train r2": train_r2,
                }
                write_to_tensorboard(writer, itr, batches_per_epoch, train_metrics, 1)
                val_metrics = {
                    "val mse": val_mse,
                    "val rmse": val_rmse,
                    "val r2": val_r2,
                }
                write_to_tensorboard(writer, itr, batches_per_epoch, val_metrics, 2)

                if val_mse < min_mse:
                    best_epoch = itr // batches_per_epoch
                    if best_epoch >= 300:
                        torch.save(
                            {"state_dict": model.state_dict(), "args": args},
                            os.path.join(
                                args.save,
                                "model_{}.pth".format(experiment_id),
                            ),
                        )
                    min_mse = val_mse

                print(
                    "Epoch {:04d} | Time {:.3f} ({:.3f})".format(
                        itr // batches_per_epoch,
                        batch_time_meter.val,
                        batch_time_meter.avg
                    )
                )
                print(
                    "\tTrain mse {:.6f} | Train rmse {:.6f} | Train r2 {:.6f}".format(
                        train_mse, train_rmse, train_r2
                    )
                )
                print(
                    "\tTest mse {:.6f} | Test rmse {:.6f} | Test r2 {:.6f}".format(
                        val_mse, val_rmse, val_r2
                    )
                )
                
    print("Best performance {} at {}".format(min_mse, best_epoch))