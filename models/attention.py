"""

Copyright 2017- IBM Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        '''
        :param output: decoder의 이전 layer의 output (query)
        :param context: encoder의 output (key)
        :return:
        '''
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))

        # softmax(Q*K), (B, O, I)
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # attn*V
        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        #########################################################
        combined = torch.cat((mix, output), dim=2)

        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn

########### Multi-head attention ############
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.d_k = dim // heads
        self.head_num = heads

        self.q_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask


    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def attention(self, q, k, v, mask=None, dropout = None):
        dk = q.size()[-1]
        scores = torch.bmm(q, k.transpose(-2, -1))/math.sqrt(dk)
        if mask is not None:
            scores.data.masked_fill_(mask, -float('inf'))
        attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        return attn

    def forward(self, output, context):
        '''
        :param output: decoder의 이전 layer의 output (query)
        :param context: encoder의 output (key)
        :return:
        '''
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        # perform linear operation and split into h heads
        query, key, value = self.q_linear(output), self.k_linear(context), self.v_linear(context)

        # transpose to get dimensions batch_size*h*sl*dim
        query = self._reshape_to_batches(query)
        key = self._reshape_to_batches(key)
        value = self._reshape_to_batches(value)

        # calculate attention score: [bs*num_head, l, hidden/num_head]
        attn = self.attention(query, key, value, self.mask, self.dropout)
        scores = torch.bmm(attn, value)
        #attn = self._reshape_from_batches(attn)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)

        combined = torch.cat((concat, output), dim=2)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn

# testing
#(32, 39, 6, 85)
# mh_attention = MultiHeadAttention(256, 8)
# context = torch.randn(32, 39, 256)
# output = torch.randn(32, 24, 256)
# output, attn = mh_attention(output, context)

