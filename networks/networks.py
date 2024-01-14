import torch
from torch import nn
import torchvision.models as models
import segmentation_models_pytorch as smp



class VGG16_fusion(nn.Module):

    def __init__(self, class_num=3, input_channel=4, pretrained=True):
        super(VGG16_fusion, self).__init__()
        base_model = models.vgg16(pretrained=pretrained)

        self.adaptror = nn.Conv2d(input_channel, 3, kernel_size=1, stride=1, padding=0)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
        self.classifier[6] = nn.Linear(4096, class_num)

    def forward(self, x):
        x = self.adaptror(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    


    
class Localizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
    def forward(self, x):
        return self.net(x)
    



class MaxViT(nn.Module):
    def __init__(self, class_num=3, pretrained=True):
        super().__init__()
        base_model = models.maxvit_t(pretrained=pretrained)
        base_model.classifier[5] = nn.Linear(512, class_num)
        self.net = base_model
    def forward(self, x):
        return self.net(x)


class SwinTransformer(nn.Module):
    def __init__(self, class_num=3, pretrained=True):
        super().__init__()
        base_model = models.swin_v2_b(pretrained=pretrained)
        base_model.head = nn.Linear(1024, class_num)
        self.net = base_model
    def forward(self, x):
        return self.net(x)


class ViT(nn.Module):
    def __init__(self, class_num=3, pretrained=True):
        super().__init__()
        base_model = models.vit_b_32(pretrained=pretrained)
        base_model.heads = nn.Linear(768, class_num)
        self.net = base_model
    def forward(self, x):
        return self.net(x)


class EfficientB3(nn.Module):
    def __init__(self, class_num=3, pretrained=True):
        super().__init__()
        base_model = models.efficientnet_b3(pretrained=pretrained)
        base_model.classifier[1] = nn.Linear(1536, class_num)
        self.net = base_model

    def forward(self, x):
        return self.net(x)


class VGG16(nn.Module):

    def __init__(self, class_num=3, pretrained=True):
        super(VGG16, self).__init__()

        base_model = models.vgg16(pretrained=pretrained)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
        self.classifier[6] = nn.Linear(4096, class_num)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, class_num=3, pretrained=True):
        super().__init__()
        base_model = models.resnet50(pretrained=pretrained)
        base_model.fc =  nn.Linear(2048, class_num)
        self.net = base_model

    def forward(self, x):
        return self.net(x)
