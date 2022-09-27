import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# MALI integrator of NODE
from TorchDiffEqPack import odesolve_adjoint_sym12

from AGCRNCell import AGCRNCell
from DCRNNCell import DCGRUCell
from torch_geometric_temporal.nn.attention import STConv


# ODE
class ODEfunc(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, t, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class InputODEfunc(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out, device):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.device = device

    def forward(self, t, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        zero_tensor = torch.zeros(out.shape[0], self.dim_in - self.dim_out).to(self.device)
        return torch.cat([out, zero_tensor], dim=1)


class ODEBlock(nn.Module):
    def __init__(self, odefunc, tol):
        super().__init__()
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x):
        # configure training options
        options = {}
        options.update({"method": "sym12async"})
        options.update({"h": None})
        options.update({"t0": 0.0})
        options.update({"t1": 1.0})
        options.update({"rtol": self.tol})
        options.update({"atol": self.tol})
        options.update({"print_neval": False})
        options.update({"neval_max": 1000000})
        options.update({"t_eval": None})
        options.update({"interpolation_method": "cubic"})
        options.update({"regenerate_graph": False})

        if len(x.shape) == 2:
            pass
        elif x.shape[1] > 1: # multi
            x = x[:, -1, :].squeeze(1)
        else: # single
            x = x.squeeze(1)

        out = odesolve_adjoint_sym12(self.odefunc, x, options=options)
        return out
        

# RNN
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.0):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# LSTM
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.0):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# GRU
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.0):
        super(GRU, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        x = x.unsqueeze(-1) #TODO: unsqueeze x: (B,T,N) to (B,T,N,D)
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    def __init__(self, args, num_nodes, input_dim, rnn_units, output_dim, horizon, num_layers):
        super(AGCRN, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.embed_dim = args.emb_dim
        self.cheb_k=args.cheb_k

        # self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(self.num_node, self.input_dim, self.hidden_dim, self.cheb_k,
                                self.embed_dim, self.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    # def forward(self, source, targets, teacher_forcing_ratio=0.5):
    def forward(self, source):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output

class STGCN(torch.nn.Module):
    def __init__(self, args,num_node, input_dim, hidden_dim, out_dim, horizon, adj):
        super(STGCN, self).__init__()
        K=1
        self.recurrent = STConv(num_node, input_dim, hidden_dim, hidden_dim, kernel_size=1, K=2)
        self.linear = torch.nn.Linear(hidden_dim, horizon)
        self.edge_index = adj.indices()
        self.edge_weight = adj.values()
    def forward(self, x):
        x = x.unsqueeze(-1)
        h = self.recurrent(x, self.edge_index, self.edge_weight)
        output = self.linear(h[:,-1:,:,:])
        return output

class DCRNN(nn.Module):
    def __init__(self, supports, num_node, input_dim, hidden_dim, order, num_layers=1):
        super(DCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(DCGRUCell(supports, num_node, input_dim, hidden_dim, order))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(DCGRUCell(supports, num_node, hidden_dim, hidden_dim, order))

    def forward(self, x, init_state):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)
class DCRNNModel(nn.Module):
    def __init__(self, args, supports, num_node, input_dim, hidden_dim, out_dim, order, num_layers=1):
        super(DCRNNModel, self).__init__()
        self.num_node = num_node
        self.input_dim = input_dim
        self.output_dim = out_dim
        self.encoder = DCRNN(supports, num_node, input_dim, hidden_dim, order, num_layers)
        self.projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, source):
        source = source.unsqueeze(-1)
        init_state = self.encoder.init_hidden(source.shape[0])
        out, _ = self.encoder(source, init_state)
        outputs = self.projection(out)
        return outputs[:,-1:,:,:]      #B, T, N, D