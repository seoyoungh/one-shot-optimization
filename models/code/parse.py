import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nblocks",
        type=int,
        default=1,
        help="the ODE block numbers except the InputODEblock",
    )
    parser.add_argument("--seq_len", type=int, default=1, help="the length of sequence")
    parser.add_argument(
        "--initial_coeff_value",
        type=float,
        default=0.03,
        help="the initial value for the InputODEBlock's coefficient.",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-5, help="the param in odeint function"
    )
    parser.add_argument("--adjoint", type=eval, default=True, choices=[True, False])
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--decay", type=float, default=1e-5, help="the weight decay for l2 normalizaton"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--save", type=str, default="./experiment")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--model", type=str, default="node", choices=["node", "rnn", "lstm", "gru", "agcrn", "dcrnn", "stgcn"]
    )
    parser.add_argument(
        "--dataset", type=str, default="sfpark", choices=["sfpark", "seattle"]
    )
    parser.add_argument(
        "--price", type=int, default=1
    )

    # COMMON
    parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')

    # AGCRN
    parser.add_argument('--input_dim',type=int,default=1,help='input dimension')
    parser.add_argument('--output_dim',type=int,default=1,help='output dimension')
    parser.add_argument('--emb_dim',type=int,default=2,help='dim of node embedding')
    parser.add_argument('--num_layers',type=int,default=2,help='number of layers')
    parser.add_argument('--agcrn_horizon',type=int,default=1,help='horizon')
    parser.add_argument('--rnn_units',type=int,default=64,help='rnn_units')
    parser.add_argument('--cheb_k',type=int,default=2,help='cheb_k')

    return parser.parse_args()