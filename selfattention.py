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

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask

        attn = self.softmax(u) # 4.Softmax
        output = torch.bmm(attn, v) # 5.Output

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

        q = self.fc_q(q) # 1.单头变多头
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask) # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat
        output = self.fc_o(output) # 4.仿射变换得到最终输出

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
    for dirpath,dirnames,filenames in os.walk(folder):
        for file in filenames:
            filename = file.split('.')[0][:4]
            file_type = file.split('.')[-1]
            if file_type == filetype and filename in ['2015','2016']:
                file_fullname = os.path.join(dirpath, file) #文件全名
                filelist.append(file_fullname)
    return filelist

def splitdatalist(full_list,shuffle=False,ratio=0.1):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2

def make_loader(file_path):
    # start = time.clock()
    temp=h5py.File(file_path,"r")
    x_data = temp['vol'][()]
    # x_data = temp["vol"][:,:,:,0] + temp['vol'][:,:,:,1]
    y_data = temp['labels'][()]
    x_data = torch.from_numpy(x_data).float().sum(axis=3)
    y_data = torch.from_numpy(y_data).float()
    dataset = TensorDataset(x_data,y_data)
    loader = DataLoader(dataset=dataset,batch_size=256,shuffle=True,drop_last=True,pin_memory=True,num_workers=16)
    # print('1',time.clock()-start)
    return loader


if __name__ == '__main__':
    learning_rate = 0.001
    epoch_num = 20
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    train_loss_list = []
    val_loss_list = []

    n_x = 31
    d_x = 601
    batch = 256

    mask = None

    model = SelfAttention(n_head=8, d_k=128, d_v=64, d_x=601, d_o=80, l=31)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #     optimizer = torch.optim.RMSprop(model.parameters(),lr=learning_rate,alpha=0.99,eps=1e-08, weight_decay=0, momentum=0, centered=False)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    filelist = get_file_list(r'/data1/lanwei/chouma_h5')
    trainlist, testlist = splitdatalist(filelist, shuffle=True, ratio=0.9)
    print(len(trainlist),len(testlist))

    for i in range(epoch_num):
        print('bingo start')
        print('epoch:%i'%i)
        model.train()
        train_loss = 0
        for path in trainlist:
            train_loader = make_loader(path)
            # print('loader1')
            # start2 = time.clock()
            for j, (x, y) in enumerate(train_loader):

                x = x.to(device)
                y = y.to(device)
                attn, output = model(x, mask=mask)
                loss = loss_fn(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            # print(time.clock()-start2)
            # print(loss.item())
        train_loss_list.append(train_loss / (j + 1) / len(trainlist))
        #         train_loss_list.append(loss.item())

        model.eval()
        val_loss = 0
        for path in testlist:
            test_loader = make_loader(path)
            # print('loader2')
            for k, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                attn, ouput = model(x)
                loss = loss_fn(output, y)
                val_loss += loss.item()
            # print(loss.item())
        val_loss_list.append(val_loss / (k + 1) / (len(testlist)))
        #         val_loss_list.append(loss.item())

        if train_loss_list[i] == min(train_loss_list) or val_loss_list[i] == min(val_loss_list):
            value = '%.4f' % train_loss_list[i] + ',' + '%.4f' % val_loss_list[i]
            torch.save(model.state_dict(), '%s.pkl' % value)

        print(train_loss_list[i], val_loss_list[i])

    df = pd.DataFrame({'trainloss': train_loss_list, 'testloss': val_loss_list})
    df.to_csv('./selfattention.csv', encoding='UTF-8', index=False)
    print('ok')
