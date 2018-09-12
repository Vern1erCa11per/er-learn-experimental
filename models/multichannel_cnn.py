import torch
import torch.nn as nn
import torch.nn.functional as F

'''
multichannel cnn model

'''
class SkipConnectionBlock(nn.Module):

    def __init__(self, in_channel, wide_ratio, kernel_size=3, under_sampling_size=3,
                 under_sampling_stride=2):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = self.in_channel * wide_ratio

        self.kernel_size = kernel_size

        self.padding = self.kernel_size // 2 + int(self.kernel_size % 2)

        self.under_sampling_kernel_size = under_sampling_size
        self.under_sampling_padding = self.under_sampling_kernel_size // 2
        self.under_sampling_stride = under_sampling_stride

        self.identity_map = nn.Conv2d(self.in_channel, self.out_channel,
                                      kernel_size=(self.under_sampling_kernel_size, 1),
                                      stride=(self.under_sampling_stride, 1),
                                      padding=(self.under_sampling_padding, 0),
                                      bias=False, groups=self.in_channel)

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel,
                               kernel_size=(self.kernel_size, 1),
                               padding=(self.padding, 0), bias=False,
                               groups=self.in_channel)

        self.batch_norm1 = nn.BatchNorm2d(self.in_channel)
        self.batch_norm2 = nn.BatchNorm2d(self.out_channel)
        self.drop_out = nn.Dropout()
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel,
                               kernel_size=(self.kernel_size, 1),
                               padding=(self.padding, 0), bias=False,
                               groups=self.out_channel)

        self.max_pool = nn.MaxPool2d(kernel_size=(self.under_sampling_kernel_size, 1),
                                     stride=(self.under_sampling_stride, 1),
                                     padding=(self.under_sampling_padding, 0))

    def forward(self, x):
        # x axis = (N, C, H, W)
        res = self.identity_map(x)
        conv_x = self.conv1(F.relu(self.batch_norm1(x)))
        conv_x = F.relu(self.batch_norm2(conv_x))
        conv_x = self.conv2(self.drop_out(conv_x))

        h_size_diff = x.size()[2] - conv_x.size()[2]
        w_size_diff = x.size()[3] - conv_x.size()[3]

        h_pad_size = h_size_diff // 2
        w_pad_size = w_size_diff // 2

        pad_size = ( w_pad_size, w_pad_size + w_size_diff % 2, h_pad_size, h_pad_size + h_size_diff % 2, 0, 0, 0, 0)
        conv_x = F.pad(conv_x, pad=pad_size)
        conv_x = self.max_pool(conv_x)

        res += conv_x
        return res


# In[ ]:


class BasicBlock(nn.Module):
    def __init__(self, in_channel, wide_ratio, kernel_size=3, under_sampling_size=3,
                 under_sampling_stride=2):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = in_channel * wide_ratio

        self.kernel_size = kernel_size

        self.padding = self.kernel_size // 2 + int(self.kernel_size % 2)

        self.under_sampling_kernel_size = under_sampling_size
        self.under_sampling_padding = self.under_sampling_kernel_size // 2
        self.under_sampling_stride = under_sampling_stride

        self.batch_norm1 = nn.BatchNorm2d(self.out_channel)
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel,
                               kernel_size=(self.kernel_size, 1),
                               padding=(self.padding, 0), bias=False,
                               groups=self.in_channel)
        self.batch_norm2 = nn.BatchNorm2d(self.out_channel)
        self.drop_out = nn.Dropout()
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel,
                               kernel_size=(self.kernel_size, 1),
                               padding=(self.padding ,0), bias=False,
                               groups=self.out_channel)

        self.max_pool = nn.MaxPool2d(kernel_size=(self.under_sampling_kernel_size, 1),
                                     stride=(self.under_sampling_stride, 1),
                                     padding=(self.under_sampling_padding, 0))

    def forward(self, x):
        # x axis = (N, C, H, W)
        conv_x = F.relu(self.batch_norm1(self.conv1(x)))
        conv_x = self.conv2(self.drop_out(conv_x))
        conv_x = F.relu(self.batch_norm2(conv_x))
        conv_x = self.max_pool(conv_x)

        return conv_x


class ErCnn(nn.Module):

    def __init__(self, kernel_size, n_blocks, base_channel, width_ratios, strides, dropouts, undersample_srides,
                 n_additional_features, linear_out_featurs, n_class):
        super().__init__()
        assert len(width_ratios) == n_blocks + 2
        assert len(strides) ==  n_blocks + 1

        blocks = [BasicBlock(in_channel=base_channel, wide_ratio=width_ratios[0], kernel_size=kernel_size,
                             under_sampling_stride=undersample_srides[0],
                             under_sampling_size=2)]

        current_in_channel = base_channel
        for i in range(n_blocks):
            current_in_channel *= width_ratios[0]
            blocks.append(
                SkipConnectionBlock(
                    kernel_size=kernel_size,
                    in_channel=current_in_channel, wide_ratio=width_ratios[i + 1],
                    under_sampling_stride=undersample_srides[1 + i], under_sampling_size=kernel_size)
            )
        self.layers = nn.Sequential(*blocks)
        current_in_channel *= width_ratios[-1]

        self.linear1 = torch.nn.Linear(current_in_channel * 300 * 2 + n_additional_features,
                                       linear_out_featurs[0])
        self.dropout = nn.Dropout(dropouts[-1])
        self.linear2 = torch.nn.Linear(linear_out_featurs[0], n_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, additional_x):
        # first block
        x = self.layers(x)
        avg_pooled = F.adaptive_avg_pool2d(x, (1, x.size()[-1]))
        max_pooled = F.adaptive_max_pool2d(x, (1, x.size()[-1]))
        pooled = [avg_pooled.view(avg_pooled.size()[0], -1),
                  max_pooled.view(max_pooled.size()[0], -1)]
        x = torch.cat(pooled + [additional_x], dim=1)

        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x
