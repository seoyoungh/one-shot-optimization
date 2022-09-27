import torch
import torch.nn as nn
from DiffusionGCN import DiffusionGCN

class DCGRUCell(nn.Module):
    def __init__(self, supports, num_node, input_dim, hidden_dim, order):
        super(DCGRUCell, self).__init__()
        self.kernel = 'mlp'  # kernel of GCN
        self.num_node = num_node
        self.hidden_dim = hidden_dim
        self.gate = DiffusionGCN(supports, num_node, input_dim+hidden_dim, 2*hidden_dim, order, self.kernel)
        self.update = DiffusionGCN(supports, num_node, input_dim+hidden_dim, hidden_dim, order, self.kernel)

    def forward(self, x, state):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.num_node, self.hidden_dim)