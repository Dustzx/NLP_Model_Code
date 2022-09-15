import torch

from mask import mask_pad,mask_tril
from util import MultiHead,PositionalEmbedding,FullyConnectedOutput
#编码器层
class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.mh = MultiHead()
        self.fc = FullyConnectedOutput()

    def forward(self,x,mask):
        # 计算自注意力,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        score = self.mh(x, x, x, mask)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(score)

        return out
class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer_1 = EncoderLayer()
        self.layer_2 = EncoderLayer()
        self.layer_3 = EncoderLayer()
    def forward(self,x,mask):
        x = self.layer_1(x,mask)
        x = self.layer_2(x,mask)
        x = self.layer_3(x,mask)
        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.mh1 = MultiHead()
        self.mh2 = MultiHead()

        self.fc = FullyConnectedOutput()

        def forward(self,x,y,mask_pad_x,mask_tril_y):
            # 先计算y的自注意力,维度不变
            # [b, 50, 32] -> [b, 50, 32]
            y = self.mh1(y, y, y, mask_tril_y)

            # 结合x和y的注意力计算,维度不变
            # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
            y = self.mh2(y, x, x, mask_pad_x)

            # 全连接输出,维度不变
            # [b, 50, 32] -> [b, 50, 32]
            y = self.fc(y)

            return y
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer_1 = DecoderLayer()
        self.layer_2 = DecoderLayer()
        self.layer_3 = DecoderLayer()

    def forward(self,x,y,mask_pad_x,mask_tril_y):
        y = self.layer_1(x,y,mask_pad_x,mask_tril_y)
        y = self.layer_2(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_3(x, y, mask_pad_x, mask_tril_y)
        return y
