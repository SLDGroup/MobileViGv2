import torch
import copy
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

import numpy

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
from timm.models.registry import register_model

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False


# IMAGENET 
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'mobilevigv2': _cfg(crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
}

    
class Stem(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )
        
    def forward(self, x):
        return self.stem(x)
    

class DepthWiseSeparable(nn.Module):
    def __init__(self, in_dim, kernel, e=4):
        super().__init__()

        self.pw1 = nn.Conv2d(in_dim, in_dim * e, 1) # kernel size = 1
        self.norm1 = nn.BatchNorm2d(in_dim * e)
        self.act1 = nn.GELU()
        
        self.dw = nn.Conv2d(in_dim * e, in_dim * e, kernel_size=kernel, stride=1, padding=1, groups=in_dim * e) # kernel size = 3
        self.norm2 = nn.BatchNorm2d(in_dim * e)
        self.act2 = nn.GELU()
        
        self.pw2 = nn.Conv2d(in_dim * e, in_dim, 1)
        self.norm3 = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        x = self.pw1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.dw(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        x = self.pw2(x)
        x = self.norm3(x)
        return x

    
class InvertedResidual(nn.Module):
    def __init__(self, dim, kernel, expansion_ratio=4., drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.dws = DepthWiseSeparable(in_dim=dim, kernel=kernel, e=expansion_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.dws(x))
        else:
            x = x + self.drop_path(self.dws(x))
        return x
   
    
class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    
    K is the number of superpatches, therefore hops equals res // K.
    """
    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
            )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape
        
        '''
        This is the original SVGA graph construction
        '''
        # x_j = x - x
        # x_m = x
        # cnt = 1
        # for i in range(self.K, H, self.K):
        #     x_c = torch.cat([x[:, :, -i:, :], x[:, :, :-i, :]], dim=2)
        #     # x_m += x_c
        #     # cnt += 1
        #     x_j = torch.max(x_j, x_c - x)
        # for i in range(self.K, W, self.K):
        #     x_r = torch.cat([x[:, :, :, -i:], x[:, :, :, :-i]], dim=3)
        #     # x_m += x_r
        #     # cnt += 1
        #     x_j = torch.max(x_j, x_r - x)
        
        
        '''
        This is the 5 connection graph construction
        '''
        x_j = x - x
        x_c = torch.cat([x[:, :, -self.K:, :], x[:, :, :-self.K, :]], dim=2)
        x_j = torch.max(x_j, x_c - x)
        x_c = torch.cat([x[:, :, self.K:, :], x[:, :, :self.K, :]], dim=2)
        x_j = torch.max(x_j, x_c - x)
        x_r = torch.cat([x[:, :, :, -self.K:], x[:, :, :, :-self.K]], dim=3)
        x_j = torch.max(x_j, x_r - x)
        x_r = torch.cat([x[:, :, :, self.K:], x[:, :, :, :self.K]], dim=3)
        x_j = torch.max(x_j, x_r - x)
        
        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)
    
    
class RepCPE(nn.Module):
    """
    This implementation of reparameterized conditional positional encoding was originally implemented
    in the following repository: https://github.com/apple/ml-fastvit
    
    Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        spatial_shape = (7, 7),
        inference_mode=False,
    ) -> None:
        """Build reparameterizable conditional positional encoding

        Args:
            in_channels: Number of input channels.
            embed_dim: Number of embedding dimensions. Default: 768
            spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """
        super(RepCPE, self).__init__()
        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias=True,
            )
        else:
            self.pe = nn.Conv2d(
                in_channels,
                embed_dim,
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                bias=True,
                groups=embed_dim,
            )

    def forward(self, x: torch.Tensor):
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            x = self.pe(x) + x
            return x

    def reparameterize(self):
        # Build equivalent Id tensor
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros(
            (
                self.in_channels,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ),
            dtype=self.pe.weight.dtype,
            device=self.pe.weight.device,
        )
        for i in range(self.in_channels):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        # Reparameterize Id tensor and conv
        w_final = id_tensor + self.pe.weight
        b_final = self.pe.bias

        # Introduce reparam conv
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.embed_dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        for para in self.parameters():
            para.detach_()
        self.__delattr__("pe")


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, K):
        super(Grapher, self).__init__()
        self.cpe = RepCPE(in_channels=in_channels, embed_dim=in_channels, spatial_shape=(7, 7))
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = MRConv4d(in_channels * 2, in_channels, K=K)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
       
    def forward(self, x):
        x = self.cpe(x)
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)

        return x

    
class MGC(nn.Module):
    def __init__(self, in_dim, drop_path=0., K=2, use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        
        self.mixer = Grapher(in_dim, K)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim * 4),
            nn.GELU(),
            nn.Conv2d(in_dim * 4, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((in_dim, 1, 1)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((in_dim, 1, 1)), requires_grad=True)
        
    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.ffn(x))
        else:
            x = x + self.drop_path(self.mixer(x))
            x = x + self.drop_path(self.ffn(x))
        return x


class Downsample(nn.Module):
    """ 
    Convolution-based downsample
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MobileViG(torch.nn.Module):
    def __init__(self, blocks, channels, dropout=0., drop_path=0., emb_dims=512, rep=False,
                 K=[0, 8, 4, 2], conv_branches=1, distillation=True, num_classes=1000,
                 pretrained=None, out_indices=None):
        super(MobileViG, self).__init__()

        self.distillation = distillation
        self.out_indices = out_indices
        self.pretrained = pretrained
        
        self.stage_names = ['stem', 'local_1', 'local_2', 'local_3', 'global']
        
        n_blocks = sum([sum(x) for x in blocks])
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]  # stochastic depth decay rule 
        dpr_idx = 0

        self.stem = Stem(input_dim=3, output_dim=channels[0])
        
        self.backbone = []
        for i in range(len(blocks)):
            stage = []
            local_stages = blocks[i][0]
            global_stages = blocks[i][1]
            if i > 0:
                stage.append(Downsample(channels[i-1], channels[i]))
            for _ in range(local_stages):
                stage.append(InvertedResidual(dim=channels[i], kernel=3, expansion_ratio=4, drop_path=dpr[dpr_idx]))
            for _ in range(global_stages):
                stage.append(MGC(channels[i], drop_path=dpr[dpr_idx], K=K[i]))
                dpr_idx += 1
            self.backbone.append(nn.Sequential(*stage))
            
        self.backbone = nn.Sequential(*self.backbone)

        self.init_weights()
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)



    def init_weights(self):
        logger = get_root_logger()
        print("Pretrained weights being loaded")
        logger.warn('Pretrained weights being loaded')
        ckpt_path = self.pretrained
        ckpt = _load_checkpoint(
            ckpt_path, logger=logger, map_location='cpu')
        print("ckpt keys: ", ckpt.keys())
        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict_ema']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt

        state_dict = _state_dict
        missing_keys, unexpected_keys = \
            self.load_state_dict(state_dict, False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        x = self.stem(inputs)
        outs = []
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i in self.out_indices:
                outs.append(x)
        return outs


if has_mmseg:
    @seg_BACKBONES.register_module()
    def mobilevigv2_m_feat(pretrained=True, **kwargs):
        model = MobileViG(blocks=[[3,0], [3,3], [9,3], [3,3]],
                        channels=[32, 64, 192, 384],
                        dropout=0.,
                        drop_path=0.1,
                        emb_dims=512,
                        K=[0, 8, 4, 2],
                        conv_branches=1,
                        distillation=True,
                        num_classes=1000,
                        out_indices=[0, 1, 2, 3],
                        pretrained='../Results/MobileViG_V2_M_Semantic_Segmentation.pth')
        model.default_cfg = default_cfgs['mobilevigv2']
        return model

    @seg_BACKBONES.register_module()
    def mobilevigv2_b_feat(pretrained=True, **kwargs):
        model = MobileViG(blocks=[[3,0], [3,3], [9,3], [3,3]],
                        channels=[64, 128, 256, 512],
                        dropout=0.,
                        drop_path=0.1,
                        emb_dims=768,
                        K=[0, 8, 4, 2],
                        conv_branches=1,
                        distillation=True,
                        num_classes=1000,
                        out_indices=[0, 1, 2, 3],
                        pretrained='../Results/MobileViG_V2_B_Semantic_Segmentation.pth')
        model.default_cfg = default_cfgs['mobilevigv2']
        return model


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model