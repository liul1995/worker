import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,TensorDataset
import h5py
import torch.nn.functional as F
import os
import random
import numpy as np
import pandas as pd
import time
import pandas as pd


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2))  # 1.Matmul
        u = u / self.scale  # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)  # 3.Mask

        attn = self.softmax(u)  # 4.Softmax
        output = torch.bmm(attn, v)  # 5.Output

        return attn, output


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):
        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q)  # 1.单头变多头
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask)  # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1)  # 3.Concat
        output = self.fc_o(output)  # 4.仿射变换得到最终输出

        return attn, output


class SelfAttention(nn.Module):
    """ Self-Attention """

    def __init__(self, n_head, d_k, d_v, d_x, d_o, l):
        super(SelfAttention, self).__init__()
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)
        self.rnn = nn.LSTM(d_o, l)
        self.out = nn.Linear(l, 1)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)

        attn, output = self.mha(q, k, v, mask=mask)
        output, (hn, cn) = self.rnn(output)
        output = self.out(output[:, :, -1])

        return attn, output


def get_file_list(folder):
    filetype = 'hdf5'
    filelist = []
    for dirpath, dirnames, filenames in os.walk(folder):
        for file in filenames:
            filename = file.split('.')[0][:4]
            file_type = file.split('.')[-1]
            if file_type == filetype and filename in ['2017']:
                file_fullname = os.path.join(dirpath, file)  # 文件全名
                filelist.append(file_fullname)
    return filelist


def make_val_loader(file_path):
    # start = time.clock()
    temp = h5py.File(file_path, "r")
    x_data = temp['vol'][()]
    y_data = temp['pct_change'][()]

    x_data = torch.from_numpy(x_data).float().sum(axis=3)
    y_data = torch.from_numpy(y_data).float()
    dataset = TensorDataset(x_data, y_data)

    loader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=16)

    return loader


if __name__ == '__main__':
    ########
    n_x = 31
    d_x = 601
    batch = 256
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    mask = None

    model = SelfAttention(n_head=8, d_k=128, d_v=64, d_x=601, d_o=80, l=31)
    model.to(device)

    model.load_state_dict(torch.load('./0.0777,0.0775.pkl'), strict=False)

    #     filelist = get_file_list(r'/data1/lanwei/chouma_h5')
    prefilelist = get_file_list(r'./')[:3]
    print(prefilelist)
    #     val_dates_2=['20170103.hdf5','20170104.hdf5','20170105.hdf5','20170106.hdf5','20170109.hdf5','20170110.hdf5','20170111.hdf5','20170112.hdf5','20170113.hdf5','20170116.hdf5','20170117.hdf5','20170118.hdf5','20170119.hdf5','20170120.hdf5','20170123.hdf5','20170124.hdf5','20170125.hdf5','20170126.hdf5','20170203.hdf5','20170206.hdf5','20170207.hdf5','20170208.hdf5','20170209.hdf5','20170210.hdf5','20170213.hdf5','20170214.hdf5','20170215.hdf5','20170216.hdf5','20170217.hdf5','20170220.hdf5','20170221.hdf5','20170222.hdf5','20170223.hdf5','20170224.hdf5','20170227.hdf5','20170228.hdf5','20170301.hdf5','20170302.hdf5','20170303.hdf5','20170306.hdf5','20170307.hdf5','20170308.hdf5','20170309.hdf5','20170310.hdf5','20170313.hdf5','20170314.hdf5','20170315.hdf5','20170316.hdf5','20170317.hdf5','20170320.hdf5','20170321.hdf5','20170322.hdf5','20170323.hdf5','20170324.hdf5','20170327.hdf5','20170328.hdf5','20170329.hdf5','20170330.hdf5','20170331.hdf5']
    predict = {'pre': [], 'pct': [], 'date': []}
    for path in prefilelist:
        print(path)
        date = path.split('/')[-1][:8]
        print(date)

        val_loader = make_val_loader(path)
        # print('loader2')
        for k, (x, y) in enumerate(val_loader):
            x = x.to(device)
            attn, output = model(x)

            output = output.squeeze().cpu().detach().tolist()
            y = y.squeeze().tolist()

            predict['pre'].extend(output)
            predict['pct'].extend(y)
            predict['date'].extend([date] * batch)

    print(len(predict['pre']))
    print(len(predict['pct']))
    print(set(predict['date']))

    val_pre = pd.DataFrame(predict)
    val_pre["pre_rank"] = val_pre.groupby("date")["pre"].rank(pct=True)
    val_pre["pre_rank"] = pd.cut(val_pre["pre_rank"], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                 labels=False)
    print(val_pre.groupby("pre_rank")["pct"].mean())