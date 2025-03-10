from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .regnet import RegNet
from .res2net import Res2Net
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .our_resnet import Our_ResNet
from .ViTAE_Window_NoShift.base_model import ViTAE_Window_NoShift_basic
from .swin_transformer import swin
from .SatMAE_vit_large import SatMAEVisionTransformer
from .SatMAEpp_vit_large import SatMAEVisionTransformerpp
from .scale_MAE_vit_large import scale_SatMAEVisionTransformer
from .CLIP import CLIP
from .vitae_nc_win_rvsa_v3_wsz7 import ViTAE_NC_Win_RVSA_V3_WSZ7

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
    'HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt','ViTAE_Window_NoShift_basic',
    'Our_ResNet', 'swin',"SatMAEVisionTransformer","SatMAEVisionTransformerpp", "scale_SatMAEVisionTransformer","CLIP",
    'ViTAE_NC_Win_RVSA_V3_WSZ7'
]
