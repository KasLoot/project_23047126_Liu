import torch
import torch.nn as nn


# The autopad is used to detect the padding value for the Convolution layer
def autopad(k, p=None, d=1):
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
  
# This is the activation function used in YOLOv11
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


# The base Conv Block

class Conv(torch.nn.Module):

    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, g=1):
        # in_ch = input channels
        # out_ch = output channels
        # activation = the torch function of the activation function (SiLU or Identity)
        # k = kernel size
        # s = stride
        # p = padding
        # g = groups
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = activation

    def forward(self, x):
        # Passing the input by convolution layer and using the activation function
        # on the normalized output
        return self.relu(self.norm(self.conv(x)))
        
    def fuse_forward(self, x):
        return self.relu(self.conv(x))
    


# The Bottlneck block

class Residual(torch.nn.Module):
    def __init__(self, ch, e=0.5):
        super().__init__()
        self.conv1 = Conv(ch, int(ch * e), torch.nn.SiLU(), k=3, p=1)
        self.conv2 = Conv(int(ch * e), ch, torch.nn.SiLU(), k=3, p=1)

    def forward(self, x):
        # The input is passed through 2 Conv blocks and if the shortcut is true and
        # if input and output channels are same, then it will the input as residual
        return x + self.conv2(self.conv1(x))
        
    

# The C3k Module
class C3K(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, torch.nn.SiLU())
        self.conv2 = Conv(in_ch, out_ch // 2, torch.nn.SiLU())
        self.conv3 = Conv(2 * (out_ch // 2), out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e=1.0),
                                         Residual(out_ch // 2, e=1.0))

    def forward(self, x):
        y = self.res_m(self.conv1(x)) # Process half of the input channels
        # Process the other half directly, Concatenate along the channel dimension
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))
        


# The C3K2 Module

class C3K2(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r):
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r), torch.nn.SiLU())
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch, torch.nn.SiLU())

        if not csp:
            # Using the CSP Module when mentioned True at shortcut
            self.res_m = torch.nn.ModuleList(Residual(out_ch // r) for _ in range(n))
        else:
            # Using the Bottlenecks when mentioned False at shortcut
            self.res_m = torch.nn.ModuleList(C3K(out_ch // r, out_ch // r) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))
        



# Code for SPFF Block
class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, torch.nn.SiLU(), k=1, s=1)
        self.cv2    = Conv(c_ * 4, c2, torch.nn.SiLU(), k=1, s=1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x) # Starting with a Conv Block
        y1 = self.m(x) # First MaxPool layer
        y2 = self.m(y1) # Second MaxPool layer
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1)) # Ending with Conv Block



# Code for the Attention Module

class Attention(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.num_head = num_head
        self.dim_head = ch // num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5

        self.qkv = Conv(ch, ch + self.dim_key * num_head * 2, torch.nn.Identity())

        self.conv1 = Conv(ch, ch, torch.nn.Identity(), k=3, p=1, g=ch)
        self.conv2 = Conv(ch, ch, torch.nn.Identity())

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)

        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
        return self.conv2(x)



# Code for the PSAModule
class PSABlock(torch.nn.Module):
    # This Module has a sequential of one Attention module and 2 Conv Blocks
    def __init__(self, ch, num_head):
        super().__init__()
        self.conv1 = Attention(ch, num_head)
        self.conv2 = torch.nn.Sequential(Conv(ch, ch * 2, torch.nn.SiLU()),
                                         Conv(ch * 2, ch, torch.nn.Identity()))

    def forward(self, x):
        x = x + self.conv1(x)
        return x + self.conv2(x)




# PSA Block Code

class PSA(torch.nn.Module):

    def __init__(self, ch, n):
        super().__init__()
        self.conv1 = Conv(ch, 2 * (ch // 2), torch.nn.SiLU())
        self.conv2 = Conv(2 * (ch // 2), ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(*(PSABlock(ch // 2, ch // 128) for _ in range(n)))

    def forward(self, x):
        # Passing the input to the Conv Block and splitting into two feature maps
        x, y = self.conv1(x).chunk(2, 1)
        # 'n' number of PSABlocks are made sequential, and then passes one them (y) of
        # the feature maps and concatenate with the remaining feature map (x)
        return self.conv2(torch.cat(tensors=(x, self.res_m(y)), dim=1))
  



class DFL(nn.Module):
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x           = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1     = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)




class DarkNet(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], torch.nn.SiLU(), k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(width[1], width[2], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p2.append(C3K2(width[2], width[3], depth[0], csp[0], r=4))
        # p3/8
        self.p3.append(Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p3.append(C3K2(width[3], width[4], depth[1], csp[0], r=4))
        # p4/16
        self.p4.append(Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p4.append(C3K2(width[4], width[4], depth[2], csp[1], r=2))
        # p5/32
        self.p5.append(Conv(width[4], width[5], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p5.append(C3K2(width[5], width[5], depth[3], csp[1], r=2))
        self.p5.append(SPPF(width[5], width[5]))
        self.p5.append(PSA(width[5], depth[4]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5




class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.h1 = C3K2(width[4] + width[5], width[4], depth[5], csp[0], r=2)
        self.h2 = C3K2(width[4] + width[4], width[3], depth[5], csp[0], r=2)
        self.h3 = Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h4 = C3K2(width[3] + width[4], width[4], depth[5], csp[0], r=2)
        self.h5 = Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h6 = C3K2(width[4] + width[5], width[5], depth[5], csp[1], r=2)

    def forward(self, x):
        p3, p4, p5 = x
        p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
        p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
        return p3, p4, p5



def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)





def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv






class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)
        
        self.box = torch.nn.ModuleList(
           torch.nn.Sequential(Conv(x, box,torch.nn.SiLU(), k=3, p=1),
           Conv(box, box,torch.nn.SiLU(), k=3, p=1),
           torch.nn.Conv2d(box, out_channels=4 * self.ch,kernel_size=1)) for x in filters)
        
        self.cls = torch.nn.ModuleList(
            torch.nn.Sequential(Conv(x, x, torch.nn.SiLU(), k=3, p=1, g=x),
            Conv(x, cls, torch.nn.SiLU()),
            Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
            Conv(cls, cls, torch.nn.SiLU()),
            torch.nn.Conv2d(cls, out_channels=self.nc,kernel_size=1)) for x in filters)

    def forward(self, x):
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        if self.training:
            return x

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

    @torch.no_grad()
    def initialize_biases(self):
        for box_branch, cls_branch in zip(self.box, self.cls):
            if box_branch[-1].bias is not None:
                box_branch[-1].bias.data.fill_(1.0)
            if cls_branch[-1].bias is not None:
                cls_branch[-1].bias.data[:self.nc].fill_(0.0)



class YOLO11_v2(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.darknet = DarkNet(
            width=[3, 16, 32, 64, 128, 256],
            depth=[1, 1, 1, 1, 1, 1],
            csp=[False, True],
        )
        self.fpn = DarkFPN(
            width=[3, 16, 32, 64, 128, 256],
            depth=[1, 1, 1, 1, 1, 1],
            csp=[False, True],
        )
        # p3 has 64 channels, p4 has 128, p5 has 256 → total 448
        self.fc1 = torch.nn.Linear((64 + 128 + 256)*1, 128)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        p3, p4, p5 = self.fpn(self.darknet(x))

        # Global average pool each scale: [B, C, H, W] → [B, C]
        p3 = torch.nn.functional.adaptive_avg_pool2d(p3, 1).flatten(1)
        p4 = torch.nn.functional.adaptive_avg_pool2d(p4, 1).flatten(1)
        p5 = torch.nn.functional.adaptive_avg_pool2d(p5, 1).flatten(1)

        # Concatenate along channel dim: [B, 64+128+256] = [B, 448]
        x = torch.cat([p3, p4, p5], dim=1)

        x = self.fc1(x)
        x = torch.nn.functional.rms_norm(x, normalized_shape=x.shape[1:])
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x




class YOLO11_Classification(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        # p3 has 64 channels, p4 has 128, p5 has 256 → total 448
        self.fc1 = torch.nn.Linear((64 + 128 + 256)*1, 128)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        

        x = self.fc1(x)
        x = torch.nn.functional.rms_norm(x, normalized_shape=x.shape[1:])
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x




class YOLO11_Detection(nn.Module):
    """YOLO11-style detection model.

    - Training mode (`model.train()`): returns raw multi-scale head outputs (list of tensors),
      suitable for a detection loss.
    - Eval mode (`model.eval()`): returns center-based boxes + class prediction per box.

    Bounding box format is (cx, cy, w, h) in input-image pixels.
    """

    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 256,
        width=None,
        depth=None,
        csp=None,
    ):
        super().__init__()

        # Defaults match the backbone used by YOLO11_v2 above
        if width is None:
            width = [3, 16, 32, 64, 128, 256]
        if depth is None:
            depth = [1, 1, 1, 1, 1, 1]
        if csp is None:
            csp = [False, True]

        self.num_classes = num_classes
        self.img_size = img_size


        self.head = Head(nc=num_classes, filters=(width[3], width[4], width[5]))

        # p3, p4, p5 come from /8, /16, /32 scales for this backbone/FPN design.
        self.head.stride = torch.tensor([8.0, 16.0, 32.0], dtype=torch.float32)
        self.stride = self.head.stride
        self.head.initialize_biases()

    def _decode_raw_outputs(self, raw_outputs):
        """
        Decode raw training-mode outputs (list of [B, no, H, W]) into:
        - boxes_cxcywh: [B, N, 4] in input-image pixels
        - probs: [B, N, nc]
        """
        if not isinstance(raw_outputs, (list, tuple)) or len(raw_outputs) == 0:
            raise ValueError("Expected non-empty list/tuple of raw detection outputs")

        stride = self.head.stride.to(raw_outputs[0].device, dtype=raw_outputs[0].dtype)
        anchors, strides = (t.transpose(0, 1) for t in make_anchors(raw_outputs, stride))

        x = torch.cat([feat.view(feat.shape[0], self.head.no, -1) for feat in raw_outputs], dim=2)
        box, cls = x.split(split_size=(4 * self.head.ch, self.head.nc), dim=1)

        a, b = self.head.dfl(box).chunk(2, 1)
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        pred = torch.cat(tensors=(box * strides, cls.sigmoid()), dim=1)
        boxes_cxcywh = pred[:, :4, :].transpose(1, 2).contiguous()  # [B, N, 4]
        probs = pred[:, 4:, :].transpose(1, 2).contiguous()  # [B, N, nc]
        return boxes_cxcywh, probs

    def forward(self, p3, p4, p5, return_scores: bool = True):
        feats = [p3, p4, p5]
        y = self.head(feats)

        if isinstance(y, (list, tuple)):
            boxes_cxcywh, probs = self._decode_raw_outputs(y)
        else:
            # Inference mode from Head: y is [B, 4 + nc, N]
            boxes_cxcywh = y[:, :4, :].transpose(1, 2).contiguous()  # [B, N, 4]
            probs = y[:, 4:, :].transpose(1, 2).contiguous()  # [B, N, nc]

        scores, class_ids = probs.max(dim=2)  # [B, N]
        if return_scores:
            return boxes_cxcywh, probs, class_ids, scores
        return boxes_cxcywh, probs, class_ids



class YOLO11_ALL(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.darknet = DarkNet(
            width=[3, 16, 32, 64, 128, 256],
            depth=[1, 1, 1, 1, 1, 1],
            csp=[False, True],
        )
        self.fpn = DarkFPN(
            width=[3, 16, 32, 64, 128, 256],
            depth=[1, 1, 1, 1, 1, 1],
            csp=[False, True],
        )
        self.classification_head = YOLO11_Classification(num_classes=num_classes, dropout=dropout)
        self.detection_head = YOLO11_Detection(num_classes=num_classes, img_size=256, width=[3, 16, 32, 64, 128, 256], depth=[1, 1, 1, 1, 1, 1], csp=[False, True])

    def forward(self, x):
        p3, p4, p5 = self.fpn(self.darknet(x))

        # Keep feature maps for detection
        det_p3, det_p4, det_p5 = p3, p4, p5

        # Global average pool each scale: [B, C, H, W] → [B, C]
        p3 = torch.nn.functional.adaptive_avg_pool2d(p3, 1).flatten(1)
        p4 = torch.nn.functional.adaptive_avg_pool2d(p4, 1).flatten(1)
        p5 = torch.nn.functional.adaptive_avg_pool2d(p5, 1).flatten(1)

        # Concatenate along channel dim: [B, 64+128+256] = [B, 448]
        x = torch.cat([p3, p4, p5], dim=1)

        class_logits = self.classification_head(x)
        boxes_cxcywh, det_probs, class_ids, scores = self.detection_head(det_p3, det_p4, det_p5, return_scores=True)

        return class_logits, boxes_cxcywh, det_probs, class_ids, scores









# class YOLO(torch.nn.Module):
#     def __init__(self, width, depth, csp, num_classes):
#         super().__init__()
#         self.net = DarkNet(width, depth, csp)
#         self.fpn = DarkFPN(width, depth, csp)

#         img_dummy = torch.zeros(1, width[0], 256, 256)
#         self.head = Head(num_classes, (width[3], width[4], width[5]))
#         self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
#         self.stride = self.head.stride
#         self.head.initialize_biases()

#     def forward(self, x):
#         x = self.net(x)
#         x = self.fpn(x)
#         return self.head(list(x))

#     def fuse(self):
#         for m in self.modules():
#             if type(m) is Conv and hasattr(m, 'norm'):
#                 m.conv = fuse_conv(m.conv, m.norm)
#                 m.forward = m.fuse_forward
#                 delattr(m, 'norm')
#         return self




# class YOLOv11:
#   def __init__(self):
    
#     self.dynamic_weighting = {
#       'n':{'csp': [False, True], 'depth' : [1, 1, 1, 1, 1, 1], 'width' : [3, 16, 32, 64, 128, 256]},
#       's':{'csp': [False, True], 'depth' : [1, 1, 1, 1, 1, 1], 'width' : [3, 32, 64, 128, 256, 512]},
#       'm':{'csp': [True, True], 'depth' : [1, 1, 1, 1, 1, 1], 'width' : [3, 64, 128, 256, 512, 512]},
#       'l':{'csp': [True, True], 'depth' : [2, 2, 2, 2, 2, 2], 'width' : [3, 64, 128, 256, 512, 512]},
#       'x':{'csp': [True, True], 'depth' : [2, 2, 2, 2, 2, 2], 'width' : [3, 96, 192, 384, 768, 768]},
#     }
#   def build_model(self, version, num_classes):
#     csp = self.dynamic_weighting[version]['csp']
#     depth = self.dynamic_weighting[version]['depth']
#     width = self.dynamic_weighting[version]['width']
#     return YOLO(width, depth, csp, num_classes)