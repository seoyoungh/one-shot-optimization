import torch
import torch.nn.functional as F
import torch.nn as nn

class DiffusionGCN(nn.Module):
    def __init__(self, supports, node_num, dim_in, dim_out, order, kernel='conv'):
        #order must be integer
        super(DiffusionGCN, self).__init__()
        self.node_num = node_num
        self.supports = supports
        self.supports_len = len(supports)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.order = order
        self.kernel = kernel
        self.weight = nn.Parameter(torch.FloatTensor(size=(dim_in*(order*self.supports_len+1), dim_out)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(dim_out,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=0.)

    def forward(self, x):
        #shape of x is [B, N, D]
        batch_size = x.shape[0]
        #print(x.shape[1] , self.node_num , self.dim_in , x.shape[2])
        assert x.shape[1] == self.node_num and self.dim_in == x.shape[2]
        out = [x]
        x0 = x
        for support in self.supports:
        # support = self.supports
        # import pdb; pdb.set_trace()
            x1 = torch.einsum('ij, bjk -> bik', support, x0)
            out.append(x1)
            for k in range(2, self.order+1):
                x2 = 2 * torch.einsum('ij, bjk -> bik', support, x1) - x0
                out.append(x2)
                x1, x0 = x2, x1
        out = torch.cat(out,dim=-1)     #B, N, D, order
        out = out.reshape(batch_size*self.node_num, -1)     #B*N, D
        out = torch.matmul(out, self.weight)  # (batch_size * self._num_nodes, output_size)
        out = torch.add(out, self.biases)
        out = out.reshape(batch_size, self.node_num, self.dim_out)
        return out