import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append("/home/0/19B30511/av-nav/myss/xgenerator")

from common.utils import try_cuda


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
    def forward(self, h, context, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class ContextOnlySoftDotAttention(nn.Module):
    '''Like SoftDot, but don't concatenat h or perform the non-linearity transform
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim, context_dim=None):
        '''Initialize layer.'''
        super(ContextOnlySoftDotAttention, self).__init__()
        if context_dim is None:
            context_dim = dim
        self.linear_in = nn.Linear(dim, context_dim, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.
        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        return weighted_context, attn
        

class VisualSoftDotAttention(nn.Module):
    ''' Visual Dot Attention Layer. '''

    def __init__(self, h_dim, v_dim, dot_dim=256):
        '''Initialize layer.'''
        super(VisualSoftDotAttention, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, visual_context, mask=None):
        '''Propagate h through the network.

        h: batch x h_dim
        visual_context: batch x v_num x v_dim
        '''
        target = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1
        context = self.linear_in_v(visual_context)  # batch x v_num x dot_dim

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x v_num
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num

        weighted_context = torch.bmm(
            attn3, visual_context).squeeze(1)  # batch x v_dim
        return weighted_context, attn


class VisualFeatureEncoder(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(VisualFeatureEncoder, self).__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Linear(np.prod(input_shape), output_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.25)

    def forward(self, images):
        x = self.flatten(images)
        x = self.drop(x)
        x = self.mlp(x)
        x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class VisualImageEncoder(nn.Module):
    def __init__(self, image_shape, output_size):
        super(VisualImageEncoder, self).__init__()
        channel_size = image_shape[2]
        if image_shape[0] == 256 and image_shape[1] == 256:
            last_layer_size = 12544
        elif image_shape[0] == 128 and image_shape[1] == 128:
            last_layer_size = 2304
        else:
            raise Exception(
                f"image shape mast be (256, 256) or (128, 128), not ({image_shape[0]}, {image_shape[1]})"
            )
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_size,
                out_channels=32,
                kernel_size=(8, 8),
                stride=(4, 4),
            ),
            nn.ReLU(True), # [batch, 32, 63, 63]
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(4, 4),
                stride=(2, 2),
            ),
            nn.ReLU(True), # [batch, 64, 30, 30]
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(2, 2),
            ), # [batch, 64, 14, 14]
            Flatten(), # [batch, 12544]
            nn.Linear(last_layer_size, output_size), # [batch, output_size]
            nn.ReLU(True),
        )
        self.layer_init()
    
    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
    
    def forward(self, images):
        images = images.permute(0, 3, 1, 2) # [batch, 4, 256, 256]
        output = self.cnn(images)
        return output

