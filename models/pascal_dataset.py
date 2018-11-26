#!/usr/bin/env python

"""Never used. But most probably can be used for pretraining or testing some
features"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch
import os
from PIL import Image
import re
from os.path import join
from collections import OrderedDict
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.utils import join_data


model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self._num_classes = num_classes
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        modulelist = list(self.classifier.modules())
        for m in modulelist[1:5]:
            x = m(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    layers2 = OrderedDict()
    in_channels = 3
    layer_numer = 0
    for v in cfg:
        if v == 'M':
            # layers2['%d' % layer_numer] = nn.MaxPool2d(kernel_size=2, stride=2)
            # layer_numer += 1
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # layers2['%d' % layer_numer] = conv2d
                # layer_numer += 1
                # layers2['%d' % layer_numer] = nn.BatchNorm2d(v)
                # layer_numer += 1
                # layers2['%d' % layer_numer] = nn.ReLU(inplace=True)
                # layer_numer += 1
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                # layers2['%d' % layer_numer] = conv2d
                # layer_numer += 1
                # layers2['%d' % layer_numer] = nn.ReLU(inplace=True)
                # layer_numer += 1
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
    # return nn.Sequential(layers2)

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # if pretrained:
        # kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        # excluded_states = [u'classifier.6.bias', u'classifier.6.weight']
        # model_state_dic = model.state_dict()
        # for key, val in pretrained_dict.items():
        #     if key in excluded_states:
        #         continue
        #     model_state_dic[key] = val
        model.load_state_dict(pretrained_dict)
    return model


class PascalDataset(Dataset):
    def __init__(self, root, transform=None):
        self._root = root
        self._annot_dir = 'ImageSets/Main'
        self._image_dir = 'JPEGImages'
        self._transform = transform

        self._labels = []
        self._paths = []
        self._label2index = {}
        self._current_label = 0

        for filename_annot in os.listdir(join(self._root, self._annot_dir)):
            match = re.match(r'(\w*)_trainval.txt', filename_annot)
            if match is None:
                continue
            self._label2index[match.group(1)] = self._current_label
            with open(join(self._root, self._annot_dir, filename_annot), 'r') as f:
                for line in f:
                    line = re.match(r'(\w*)\s*(\-*\d)', line)
                    filename = line.group(1)
                    if int(line.group(2)) == 0:
                        self._labels.append(self._current_label)
                        self._paths.append(join(self._root,
                                                self._image_dir,
                                                filename + '.jpg'))
            self._current_label += 1
        print()

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        image = Image.open(self._paths[idx])
        if self._transform is not None:
            image = self._transform(image)
        return image, self._labels[idx]


def extract_features(save_path, root):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    model = vgg16_bn(pretrained=True)
    dataset = PascalDataset(transform=transform,
                            root=root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False,
                                             num_workers=4)
    features = None
    for images, labels in dataloader:
        features_cur = model.get_features(images).detach().numpy()
        label_check = labels[0]
        for idx, label in enumerate(labels):
            if label == label_check:
                features = join_data(features, features_cur, np.vstack)
            else:
                print('Label %d %d' % (label_check, features.shape[0]))
                np.save(join(save_path, '%d' % label_check), features)
                features = features_cur
                label_check = label

if __name__ == '__main__':
    save_path = '/media/data/kukleva/lab/PASCAL/VOCtrainval/VOC2012/features'
    root = '/media/data/kukleva/lab/PASCAL/VOCtrainval/VOC2012'
    extract_features(save_path, root)





