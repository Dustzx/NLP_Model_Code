import torch

def attention(Q,K,V,mask):
    '''
    :param Q:查询矩阵
    :param K:目标矩阵
    :param V:目标矩阵
    :param mask:掩码
    :return:通过Q、K相似度得分更新后的V’
    Q、K、V = [b,4,50,8]
    为b句话，每句话50个词，(每个词编码成32维，4个头，每个头分到8维向量)
    '''
    #permute将K后两个维度进行交换
    # [b, 4, 50, 8] * [b, 4, 8, 50] -> [b, 4, 50, 50]
    # Q,K矩阵相乘,求每个词相对其他所有词的注意力
    score = torch.matmul(Q,K.permute(0,1,3,2))

    #进行scale对注意力进行缩放,防止softmax后差距过大
    score /= 8**0.5

    #将mask是True的地方都替换成-inf，softmax计算时，-inf会压缩为0
    #mask = [b,1,50,50]
    score = score.masked_fill(mask,-float('inf'))
    score = torch.softmax(score,dim=-1)

    # score*V得到包含注意力分数的V
    # [b,4,50,50] * [b,4,50,8] -> [b,4,50,8]
    scored_V = torch.matul(score,V)

    # 每个头计算结果进行concate
    # [b,4,50,8]--》[b,50,32]
    scored_V = scored_V.permute(0,2,1,3).reshape(-1,50,32)
    return scored_V