# Code from
# https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/models.py
# https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.3.1/lib/non_local_embedded_gaussian.py

import torch
import torchvision


class TSM(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 dropout=0.8,
                 num_segments=8,
                 pretrained_resnet_weights=None):
        super(TSM, self).__init__()
        self.num_segments = num_segments
        base_model, classifier = self.load_base_model(num_classes, dropout, pretrained_resnet_weights)
        base_model = self.make_temporal_shift(base_model, num_segments)
        # base_model = self.make_non_local(base_model, num_segments)
        base_model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.base_model = base_model
        self.classifier = classifier

    def load_base_model(self, num_classes, dropout, pretrained_resnet_weights):
        # Load base model - resnet50
        base_model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )

        feat_dim = base_model.fc.in_features

        # Using custom pretrained weights - need to modify fully connected layer to
        # if pretrained_resnet_weights != None:
        #     num_classes_pretrained = torch.load(pretrained_resnet_weights)['fc.weight'].size(0)
        #     base_model.fc = torch.nn.Linear(base_model.fc.in_features, num_classes_pretrained)
        #     base_model.load_state_dict(torch.load(pretrained_resnet_weights))

        # Add dropout
        base_model.fc = torch.nn.Dropout(p=dropout)

        # Create classifier
        classifier = torch.nn.Linear(feat_dim, num_classes)

        return base_model, classifier

    def make_temporal_shift(self, base_model, num_segments):
        n_round = 1
        if len(list(base_model.layer3.children())) >= 23:
            n_round = 2

        def make_block_temporal(stage):
            blocks = list(stage.children())
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv1 = TemporalShift(b.conv1, num_segments, n_div=8)
            return torch.nn.Sequential(*blocks)

        base_model.layer1 = make_block_temporal(base_model.layer1)
        base_model.layer2 = make_block_temporal(base_model.layer2)
        base_model.layer3 = make_block_temporal(base_model.layer3)
        base_model.layer4 = make_block_temporal(base_model.layer4)
        return base_model

    def make_non_local(self, base_model, num_segments):
        base_model.layer2 = torch.nn.Sequential(
            NL3DWrapper(base_model.layer2[0], num_segments),
            base_model.layer2[1],
            NL3DWrapper(base_model.layer2[2], num_segments),
            base_model.layer2[3],
        )
        base_model.layer3 = torch.nn.Sequential(
            NL3DWrapper(base_model.layer3[0], num_segments),
            base_model.layer3[1],
            NL3DWrapper(base_model.layer3[2], num_segments),
            base_model.layer3[3],
            NL3DWrapper(base_model.layer3[4], num_segments),
            base_model.layer3[5],
        )
        return base_model

    def forward(self, input, no_reshape=False, features_only=False):
        if not no_reshape:
            sample_len = 3
            base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        else:
            base_out = self.base_model(input)

        if not features_only:
            base_out = self.classifier(base_out)

        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        # consensus - averaging over segments
        output = base_out.mean(dim=1, keepdim=True)

        return output.squeeze(1)

class TemporalShift(torch.nn.Module):
    def __init__(self, net, n_segment=8, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=8):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)
        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)


class NONLocalBlock3D(torch.nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__()
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = torch.nn.Conv3d
        max_pool_layer = torch.nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn = torch.nn.BatchNorm3d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = torch.nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            torch.nn.init.constant_(self.W[1].weight, 0)
            torch.nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            torch.nn.init.constant_(self.W.weight, 0)
            torch.nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = torch.nn.Sequential(self.g, max_pool_layer)
            self.phi = torch.nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = torch.nn.functional.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NL3DWrapper(torch.nn.Module):
    def __init__(self, block, n_segment):
        super(NL3DWrapper, self).__init__()
        self.block = block
        self.nl = NONLocalBlock3D(block.bn3.num_features)
        self.n_segment = n_segment

    def forward(self, x):
        x = self.block(x)
        nt, c, h, w = x.size()
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = self.nl(x)
        x = x.transpose(1, 2).contiguous().view(nt, c, h, w)
        return x
