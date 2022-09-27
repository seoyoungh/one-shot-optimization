import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def split_sf(data):
    train = data[data["DATETIME"] <= "2012-12-31"]  # training set: before 2013
    test = data[data["DATETIME"] >= "2013-01-01"]  # test set: after 2013
    train_tensor = torch.tensor(
        train.drop(["DATETIME"], axis=1).values.astype(np.float32)
    )
    test_tensor = torch.tensor(
        test.drop(["DATETIME"], axis=1).values.astype(np.float32)
    )
    return train_tensor, test_tensor

def split_sea(data):
    train = data[data["0"] <= "2019-07-18"]  # training set: before 07/18
    test = data[data["0"] >= "2019-07-19"]  # test set: after 07/19
    train_tensor = torch.tensor(
        train.drop(["0", "1"], axis=1).values.astype(np.float32)
    )
    test_tensor = torch.tensor(
        test.drop(["0", "1"], axis=1).values.astype(np.float32)
    )
    return train_tensor, test_tensor

def make_seq(x_data, x_add_data, rate_data, y_data, seq_len):
    data_total_seq = []
    data_label = []
    data_len = len(x_data)
    time_len = 8  # 10pm to 5pm, a unit of a day

    for loop in range(int(data_len / time_len)):
        start_idx = loop * time_len
        end_idx = (loop + 1) * time_len
        # import a daily data from given data (data on a same day only)
        x_day = x_data[start_idx:end_idx]
        x_add_day = x_add_data[start_idx:end_idx]
        x_rate_day = rate_data[start_idx:end_idx]
        y_day = y_data[start_idx:end_idx]
        for time in range(time_len - seq_len + 1):
            # get short term data of n hours (n = sequence length)
            x_seq = x_day[time : time + seq_len]
            # we repeat data seq_len times
            x_add_seq = x_add_day[time + seq_len - 1].repeat(seq_len, 1)
            x_rate_seq = x_rate_day[time + seq_len - 1].repeat(seq_len, 1)
            # concat x data
            X_seq = torch.cat([x_seq, x_add_seq, x_rate_seq], dim=1)
            # get target occupancy that we need to predict
            y_seq = y_day[time + seq_len - 1]
            data_total_seq.append(X_seq)
            data_label.append(y_seq)
    data_total_seq = torch.stack(data_total_seq)  # list to tensor
    data_label = torch.stack(data_label)  # list to tensor
    return data_total_seq, data_label


class SeattleDataset(Dataset):
    def __init__(self, path="../data/seattle/", seq_len=1):
        x_data = pd.read_csv(path + "x_data.csv")  # short-term occupancy
        x_add_data = pd.read_csv(path + "x_add_data.csv")  # long-term occupancy
        rate_data = pd.read_csv(path + "rate.csv")  # rate data
        y_data = pd.read_csv(path + "y_data.csv")  # target occupancy (after 1 hour)

        self.seq_len = seq_len
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_dataset(
            x_data, x_add_data, rate_data, y_data, seq_len
        )

    def __len__(self):
        return len(self.X_train), len(self.X_test)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

    def split_dataset(self, x_data, x_add_data, rate_data, y_data, seq_len):
        # split dataset to training set and test set
        x_train, x_test = split_sea(x_data)
        x_add_train, x_add_test = split_sea(x_add_data)
        rate_train, rate_test = split_sea(rate_data)
        y_train, y_test = split_sea(y_data)
        # make sequences from given data
        X_train, y_train = make_seq(x_train, x_add_train, rate_train, y_train, seq_len)
        X_test, y_test = make_seq(x_test, x_add_test, rate_test, y_test, seq_len)
        return X_train, X_test, y_train, y_test


class SfparkDataset(Dataset):
    def __init__(self, path="../data/sfpark/", seq_len=1):
        x_data = pd.read_csv(path + "x_data.csv")  # short-term occupancy
        x_add_data = pd.read_csv(path + "x_add_data.csv")  # long-term occupancy
        rate_data = pd.read_csv(path + "rate.csv")  # rate data
        y_data = pd.read_csv(path + "y_data.csv")  # target occupancy (after 1 hour)

        self.seq_len = seq_len
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_dataset(
            x_data, x_add_data, rate_data, y_data, seq_len
        )

    def __len__(self):
        return len(self.X_train), len(self.X_test)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

    def split_dataset(self, x_data, x_add_data, rate_data, y_data, seq_len):
        # split dataset to training set and test set
        x_train, x_test = split_sf(x_data)
        x_add_train, x_add_test = split_sf(x_add_data)
        rate_train, rate_test = split_sf(rate_data)
        y_train, y_test = split_sf(y_data)
        # make sequences from given data
        X_train, y_train = make_seq(x_train, x_add_train, rate_train, y_train, seq_len)
        X_test, y_test = make_seq(x_test, x_add_test, rate_test, y_test, seq_len)
        return X_train, X_test, y_train, y_test


def adj_process(adj,train_num,topk,disteps):
    """
    return: sparse CxtConv and sparse PropConv adj
    """
    # sparse context graph adj (2,E)
    edge_1 = []
    edge_2 = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if(i==j or (adj[i,j]<=disteps)):
                edge_1.append(i)
                edge_2.append(j)
    edge_adj = np.asarray([edge_1,edge_2],dtype=int)
    
    # sparse propagating adj (2,E)
    edge_1 = []
    edge_2 = [] 
    for i in range(adj.shape[0]):
        cnt = 0
        adj_row = adj[i,:train_num]
        adj_row = sorted(enumerate(adj_row), key=lambda x:x[1])  # [(idx,dis),...]
        for j,dis in adj_row:
            if(i!=j):  
                edge_1.append(i)
                edge_2.append(j)
                cnt += 1
            if(cnt >= topk and dis>disteps):
                break
    adj_label = np.asarray([edge_1,edge_2],dtype=int)
    return edge_adj, adj_label


def load_adj(dataset):
    """
    load weighted adjacency matrix
    """ 
    if dataset == 'seattle':
        adj = np.load('../data/seattle/seattle_adj.npy')
    elif dataset == 'sfpark':
        adj = np.load('../data/sfpark/sfpark_adj.npy')
    else:
        ValueError("Check adjacency matrix path")
    print('adj shape:',adj.shape) # (N, N)
    return adj

def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))