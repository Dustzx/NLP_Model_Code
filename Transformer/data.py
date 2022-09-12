#定义字典
dic_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m'
dic_x = {word: i for i ,word in enumerate(dic_x.split(','))}
dic_xr = [k for k, v in dic_x.items()]
dic_y = {k.upper(): v for k, v in dic_x.items()}
dic_yr = [k for k, v in dic_y.items()]
# print(dic_x)
# print(dic_xr)
# print(dic_y)
# print(dic_yr)
import random
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
'''
每次调用得到一对x,y
长度为30-48随机并补足到x:50 y:51
y首字母重复故len多1
'''
def get_data():
    #定义词集合
    words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'q', 'w', 'e', 'r',
        't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k',
        'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm']
    #定义每个词被选中的概率
    p = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
    ])
    p = p/p.sum()
    #随机选取n个词
    n = random.randinit(30,48)
    x = np.random.choice(words,size=n,replace=True,p=p)

    #采样结果即为x
    x = x.tolist()

    #y通过对x的变换得到
    #规律为字母变大写，数字取10以内的互补数
    def word_trans(i):
        i  = i.upper()
        if not i.isdigit():
            return i
        i = 9 - int(i)
        return str(i)
    y = [word_trans(i) for i in x]
    #将y的尾重复一遍，再进行倒置
    y = y + [y[-1]]
    y = y[::-1]

    #x,y添加首尾符号
    x = ['<SOS>'] + x + ['EOS']
    y = ['<SOS>'] + y + ['EOS']
    #补PAD
    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    x = x[:50]
    y = y[:51]

    #根据字典进行编码
    x = [dic_x(i) for i in x]
    y = [dic_x(i) for i in y]
    # 转tensor
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return x,y

#定义数据集
class Dataset(Dataset):
    def __init__(self):
        super(Dataset,self).__init__()

    def __len__(self):
        return 100000

    def __getitem__(self, i):

        return get_data()
'''
每次调用会得到8个x，y
'''
#数据加载
dataloader = DataLoader(dataset=Dataset(),
                        batch_size=8,
                        shuffle=True,
                        collate_fn=None)