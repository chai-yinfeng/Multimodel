import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence

import config

class Net(nn.Module):
    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.5,
        )
        # Replace original attention with BiAttention
        self.attention = BAN(
            vision_features=vision_features,
            question_features=question_features,
            glimpses=glimpses,
        )
        self.classifier = Classifier(
            in_features=glimpses * vision_features + question_features,
            mid_features=1024,
            out_features=config.max_answers,
            drop=0.5,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))

         # 规范化视觉特征
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        # Replace original attention mechanism with BAN
        a = self.attention(v, q)
        v = apply_attention(v, a)

        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)
        return answer


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))


class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0, num_layers=1):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()

        # 多层LSTM
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=num_layers,  # 可增加LSTM层数
                            batch_first=True)
        self.features = lstm_features

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        # 初始化所有层的权重
        # self._init_lstm()

        init.xavier_uniform_(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    # def _init_lstm(self):
    #     # 初始化 LSTM 的权重
    #     for name, param in self.lstm.named_parameters():
    #         if 'weight_ih' in name:
    #             init.xavier_uniform_(param)
    #         elif 'weight_hh' in name:
    #             init.orthogonal_(param)  # 可以考虑不同初始化方式，如正交初始化
    #         elif 'bias' in name:
    #             param.data.zero_()

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        _, (_, c) = self.lstm(packed)
        return c.squeeze(0)  # 取最后一层的隐藏状态c[-1]

class BAN(nn.Module):
    def __init__(self, vision_features, question_features, glimpses):
        super(BAN, self).__init__()
        self.glimpses = glimpses
        self.v_lin = nn.Linear(vision_features, vision_features)
        self.q_lin = nn.Linear(question_features, question_features)
        self.bilinear_pooling = nn.Bilinear(vision_features, question_features, glimpses)

    def forward(self, v, q):
        # 这里我们使用全局平均池化而不是直接mean
        v = F.adaptive_avg_pool2d(v, (1, 1))  # 全局平均池化
        v = v.view(v.size(0), -1)  # 将 (batch_size, 2048, 1, 1) 转换为 (batch_size, 2048)

        # 双线性池化
        v_proj = self.v_lin(v)  # 将视觉特征映射到指定维度
        q_proj = self.q_lin(q)  # 将问题特征映射到指定维度

        # Bilinear interaction
        bilinear_output = self.bilinear_pooling(v_proj, q_proj)

        # 不在这里做 softmax，softmax 将在 apply_attention 里进行
        return bilinear_output


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1) # [n, 1, c, s]
    attention = attention.view(n, glimpses, -1)

    # 统一在这里应用 softmax
    attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]

    weighted = attention * input # [n, g, v, s]
    weighted_mean = weighted.sum(dim=-1) # [n, g, v]

    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled
