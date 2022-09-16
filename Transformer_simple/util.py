import torch
import math


def attention(Q, K, V, mask):
    '''
    :param Q:查询矩阵
    :param K:目标矩阵
    :param V:目标矩阵
    :param mask:掩码
    :return:通过Q、K相似度得分更新后的V’
    Q、K、V = [b,4,50,8]
    为b句话，每句话50个词，(每个词编码成32维，4个头，每个头分到8维向量)
    '''
    # permute将K后两个维度进行交换
    # [b, 4, 50, 8] * [b, 4, 8, 50] -> [b, 4, 50, 50]
    # Q,K矩阵相乘,求每个词相对其他所有词的注意力
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))

    # 进行scale对注意力进行缩放,防止softmax后差距过大
    score /= 8 ** 0.5

    # 将mask是True的地方都替换成-inf，softmax计算时，-inf会压缩为0
    # mask = [b,1,50,50]
    score = score.masked_fill(mask, -float('inf'))
    score = torch.softmax(score, dim=-1)

    # score*V得到包含注意力分数的V
    # [b,4,50,50] * [b,4,50,8] -> [b,4,50,8]
    scored_V = torch.matmul(score, V)

    # 每个头计算结果进行concate
    # [b,4,50,8]--》[b,50,32]
    scored_V = scored_V.permute(0, 2, 1, 3).reshape(-1, 50, 32)
    return scored_V


# 多头注意力计算层
class MultiHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q = torch.nn.Linear(32, 32)
        self.fc_K = torch.nn.Linear(32, 32)
        self.fc_V = torch.nn.Linear(32, 32)

        self.out_fc = torch.nn.Linear(32, 32)

        # 规范化后，均值为0，标准差为1(方差的平均数取根号)
        # BatchNormalization是取不同样本做归一化
        # LayerNormalization是取不同通道做归一化
        # affine=True,elementwise_affine=True,指定规范化后,再计算一个线性映射
        # norm = torch.nn.BatchNorm1d(num_features=4, affine=True)
        # print(norm(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)))
        """
        [[[-1.1761, -1.0523, -0.9285, -0.8047],
         [-1.1761, -1.0523, -0.9285, -0.8047],
         [-1.1761, -1.0523, -0.9285, -0.8047],
         [-1.1761, -1.0523, -0.9285, -0.8047]],
        [[ 0.8047,  0.9285,  1.0523,  1.1761],
         [ 0.8047,  0.9285,  1.0523,  1.1761],
         [ 0.8047,  0.9285,  1.0523,  1.1761],
         [ 0.8047,  0.9285,  1.0523,  1.1761]]]"""

        # norm = torch.nn.LayerNorm(normalized_shape=4, elementwise_affine=True)
        # print(norm(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)))
        """
        [[[-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416]],
        [[-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416]]]"""

        self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask):
        # b句话,每句话50个词,每个词编码成32维向量
        # Q,K,V = [b, 50, 32]
        b = Q.shape[0]

        # 保留原始Q做残差连接
        clone_Q = Q.clone()

        # 规范化（由模型的实际应用知先规范化后线性变换效果更好）
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        # 线性运算，维度不变
        # [b, 50, 32] -> [b, 50, 32]
        Q = self.fc_Q(Q)
        K = self.fc_K(K)
        V = self.fc_V(V)

        # 拆分成多个头
        # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
        # [b, 50, 32] -> [b, 4, 50, 8]
        Q = Q.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)

        # 计算注意力
        # [b, 50, 32] -> [b, 50, 32]
        score = attention(Q, K, V, mask)

        # 计算输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        score = self.dropout(self.out_fc(score))

        # 短接
        score = clone_Q + score
        return score


# 位置编码层
class PositionalEmbedding(torch.nn.Module):
    def __init__(self):
        super(PositionalEmbedding, self).__init__()

        # pos是第几个词，i是第几个维度，d_model是维度总数
        def get_pe(pos, i, d_model):
            # a e n=a*10的n次方
            fenmu = 1e4 ** (i / d_model)
            pe = pos / fenmu
            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        pe = torch.empty(50, 32)
        for i in range(50):
            for j in range(32):
                pe[i, j] = get_pe(i, j, 32)
        # 在0维添加1个维度[50,32]->[1,50,32]
        pe = pe.unsqueeze(0)

        # 定义为不更新的常量
        self.register_buffer('pe', pe)

        # 词编码层
        self.embed = torch.nn.Embedding(39, 32)
        # 初始化参数
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # [8, 50] -> [8, 50, 32]将每一个词编码成32维向量
        embed = self.embed(x)

        # 词编码和位置编码相加
        # [8, 50, 32] + [1, 50, 32] -> [8, 50, 32]
        embed = embed + self.pe
        return embed


# 全连接输出层
class FullyConnectedOutput(torch.nn.Module):
    def __init__(self):
        super(FullyConnectedOutput, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.Dropout(p=0.1),
        )
        self.norm = torch.nn.LayerNorm(normalized_shape=32,
                                       elementwise_affine=True)

    def forward(self, x):
        clone_x = x.clone()

        x = self.norm(x)
        out = self.fc(x)
        out = clone_x + out

        return out


if __name__ == "__main__":
    a = torch.ones(2, 2)
    print(a)
    b = torch.zeros(1, 2)
    a = a.unsqueeze(0)
    print(a)
    '''
    [[1., 1.]]              
    0维上插入一个维度       1维上插入一个维度       2维上插入一个维度
    [[[1., 1.]]]              [[[1., 1.]]]         [[[1.],
                                                     [1.]]] 
                                                     
    [[1., 1.],
    [1., 1.]]
     0维插入                   1维插入              2维插入    
     [[[1., 1.],                [[[1., 1.]],       [[[1.],
      [1., 1.]]]                 [[1., 1.]]]        [1.]],
                                                    [[1.],
                                                    [1.]]]    
        
        
                       
    '''
