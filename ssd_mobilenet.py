import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc #, coco, cub
from nets import mobilenet
import os


class SSDMobileNetV2(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base MobileNetV2 followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, extras, head, top_down, final_features, cfg):
        super(SSDMobileNetV2, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.backbone = nn.ModuleList(mobilenet.MobileNetV2(num_classes=self.num_classes, width_mult=0.75).features)
        self.norm = L2Norm(int(96 * 0.75), 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.top_down = nn.ModuleList(top_down)
        self.final_features = nn.ModuleList(final_features)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)
    
    # @staticmethod
    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        for k in range(len(self.backbone)):
            x = self.backbone[k](x)
            if k in [13, 17]:
                if len(sources) == 0:
                    s = self.norm(x)
                    sources.append(s)
                else:
                    sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.celu(v(x), inplace=True)
            #x = v(x)
            #if k % 2 == 1:
            sources.append(x)

        top_down_x = []
        for k, v in enumerate(sources):
            x = self.top_down[k](v)
            top_down_x.append(x)

        top_down_x = top_down_x[::-1]

        pyramids = [self.final_features[0](top_down_x[0])]
        for k, v in enumerate(top_down_x[:-1]):
            size = top_down_x[k+1].shape[2:]
            x = F.upsample(v, size=size) + top_down_x[k+1]
            x = self.final_features[k+1](x)
            pyramids.append(x)

        pyramids = pyramids[::-1]

        # apply multibox head to source layers
        for (x, l, c) in zip(pyramids, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


'''def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers'''

def add_extras(cfg):
    layers = []
    block = mobilenet.InvertedResidual

    layers.append(block(int(320*0.75), 512, 2, 512.0/(320.0*0.75)))
    layers.append(block(512, 256, 2, 0.5))
    layers.append(block(256, 256, 2, 1))
    layers.append(block(256, 128, 2, 0.5))

    return layers

def multibox(extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    top_down_layers = []
    final_features = []
    mobilenet_channels = [int(96*0.75), int(320*0.75)]
    for k, channel in enumerate(mobilenet_channels):
        loc_layers += [nn.Conv2d(256,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    out_channels = mobilenet_channels + [512, 256, 256, 128]
    for k, v in enumerate(extra_layers, 2):
        loc_layers += [nn.Conv2d(256, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(out_channels):
        top_down_layers += [nn.Conv2d(v, 256, kernel_size=1)]
        final_features += [nn.Conv2d(256, 256, kernel_size=1, padding=0)]
    return extra_layers, (loc_layers, conf_layers), top_down_layers, final_features


extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd_mobilenet(phase, cfg):
    size = cfg['min_dim']
    num_classes = cfg['num_classes']
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    extras_, head_, top_down_, final_features_ = multibox(add_extras(extras[str(size)]),
                              mbox[str(size)], num_classes)
    return SSDMobileNetV2(phase, size, extras_, head_, top_down_, final_features_, cfg)
