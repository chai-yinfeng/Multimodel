import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config
import math

class Net(nn.Module):
    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2

        # 使用多头注意力
        # mid_features = 512
        # num_heads = 8  # 根据需要调整头的数量 16

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            rnn_features=question_features,
            drop=0.5,
            rnn_type='lstm',  # 或者 'gru'
        )
        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=2,
            drop=0.5,
        )

        # 多头注意力
        # self.attention = MultiHeadAttention(
        #     v_features=vision_features,
        #     q_features=question_features,
        #     mid_features=mid_features,
        #     num_heads=num_heads,
        #     drop=0.5,
        # )

        self.classifier = Classifier(
            in_features=glimpses * vision_features + question_features,
            # in_features=1536,   # mid_features + rnn_features 对于多头考虑v_emb和q_emb拼接后的形状
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
        q_output, q_mask = self.text(q, q_len)

        # 应用注意力机制
        q_emb = self.apply_attention(q_output, q_mask)

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        a = self.attention(v, q_emb)
        v = apply_attention(v, a)

        combined = torch.cat([v, q_emb], dim=1)
        answer = self.classifier(combined)
        return answer

    # 多头注意力的forward函数
    # def forward(self, v, q, q_len):
    #     q_output, q_mask = self.text(q, q_len)
    #     q_emb = self.apply_attention(q_output, q_mask)  # 这是文本特征

    #     v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
    #     v_emb = self.attention(v, q_emb)  # 这是融合后的视觉特征

    #     combined = torch.cat([v_emb, q_emb], dim=1)
    #     answer = self.classifier(combined)
    #     return answer

    def apply_attention(self, q_output, q_mask):
        # 计算注意力权重
        attn_weights = self.text.attention_linear(q_output).squeeze(-1)  # [batch_size, seq_len]
        attn_weights = attn_weights * q_mask  # 应用掩码
        attn_weights = attn_weights.masked_fill(q_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)

        # 加权求和
        attn_output = torch.bmm(attn_weights.unsqueeze(1), q_output)  # [batch_size, 1, rnn_features]
        attn_output = attn_output.squeeze(1)  # [batch_size, rnn_features]

        return attn_output

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))


class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, rnn_features, drop=0.0, rnn_type='gru'):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.rnn_type = rnn_type.lower()
        self.features = rnn_features

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_features,
                               hidden_size=rnn_features,
                               num_layers=1,
                               batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=embedding_features,
                              hidden_size=rnn_features,
                              num_layers=1,
                              batch_first=True)
        else:
            raise ValueError("Invalid rnn_type. Choose 'lstm' or 'gru'.")

        # 初始化权重
        self._init_rnn()
        # 注意力机制的线性层
        self.attention_linear = nn.Linear(rnn_features, 1)
        init.xavier_uniform_(self.embedding.weight)

    def _init_rnn(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, q, q_len):
        embedded = self.embedding(q)  # [batch_size, seq_len, embedding_features]
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len.cpu(), batch_first=True, enforce_sorted=False)
        if self.rnn_type == 'lstm':     # LSTM
            packed_output, (h_n, c_n) = self.rnn(packed)
        else:   # GRU
            packed_output, h_n = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)  # [batch_size, seq_len, rnn_features]

        # Masking for variable length sequences
        max_len = output.size(1)
        idx = torch.arange(max_len).unsqueeze(0).to(q_len.device)
        mask = (idx < q_len.unsqueeze(1)).float()  # [batch_size, seq_len]

        return output, mask

class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x

# 多头注意力
# class MultiHeadAttention(nn.Module):
#     def __init__(self, v_features, q_features, mid_features, num_heads, drop=0.0):
#         super(MultiHeadAttention, self).__init__()
#         assert mid_features % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = mid_features // num_heads

#         # 定义线性变换层
#         self.v_lin = nn.Linear(v_features, mid_features)
#         self.q_lin = nn.Linear(q_features, mid_features)
#         self.fc_out = nn.Linear(mid_features, mid_features)

#         self.dropout = nn.Dropout(drop)
#         self.relu = nn.ReLU()

#     def forward(self, v, q):
#         batch_size = v.size(0)
#         _, _, height, width = v.size()
#         num_regions = height * width

#         # 将 height 和 width 展平
#         v = v.view(batch_size, -1, num_regions)  # [batch_size, v_features, num_regions]
#         v = v.permute(0, 2, 1)  # [batch_size, num_regions, v_features]

#         # 线性变换并调整维度
#         v = self.v_lin(v)  # [batch_size, num_regions, mid_features]
#         q = self.q_lin(q).unsqueeze(1)  # [batch_size, 1, mid_features]

#         # 拆分为多头
#         v = v.view(batch_size, num_regions, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, num_regions, head_dim]
#         q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, 1, head_dim]

#         # 计算注意力得分
#         attn_scores = torch.matmul(q, v.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, 1, num_regions]
#         attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, 1, num_regions]
#         attn_weights = self.dropout(attn_weights)

#         # 加权求和得到上下文向量
#         attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, 1, head_dim]
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1)  # [batch_size, mid_features]

#         # 最后的线性层
#         output = self.fc_out(attn_output)  # [batch_size, mid_features]
#         output = self.relu(output)

#         return output

def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1) # [n, 1, c, s]
    attention = attention.view(n, glimpses, -1)
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
