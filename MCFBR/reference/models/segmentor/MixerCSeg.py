import torch
from torch import nn
from models.decoder import SRFModule
from models.encoder import VSSEncoder
import torch.nn.functional as F



class MixerCSeg(nn.Module):
    def __init__(self, backbone, embed_dims, args=None):
        super().__init__()
        self.args = args
        self.backbone = backbone    
        self.decoder = SRFModule(embed_dims, mid_dim=8, size=(args.load_width, args.load_height))

    def forward(self, samples):
        outs = self.backbone(samples)
        out = self.decoder(outs)

        return out

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc


class StructureAwareFocalLossWithLogits(nn.Module):
    def __init__(
        self,
        alpha=0.75,
        gamma=2.0,
        edge_weight=1.0,
        fg_weight=0.5,
        boundary_weight=0.2,
        smooth=1.0,
        eps=1e-7
    ):
        super(StructureAwareFocalLossWithLogits, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.edge_weight = edge_weight
        self.fg_weight = fg_weight
        self.boundary_weight = boundary_weight
        self.smooth = smooth
        self.eps = eps

    def get_morph_edge(self, y_true):
        """
        从真实标签中提取边界区域。
        dilation - erosion 可以得到裂缝边缘附近区域。
        """
        dilate = F.max_pool2d(y_true, kernel_size=3, stride=1, padding=1)
        erode = -F.max_pool2d(-y_true, kernel_size=3, stride=1, padding=1)
        edge = (dilate - erode).clamp(0, 1)

        return edge

    def get_sobel_edge(self, x):
        """
        用 Sobel 算子提取软边缘。
        x 可以是 sigmoid 后的预测图，也可以是真实标签。
        """
        b, c, h, w = x.shape

        kernel_x = torch.tensor(
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]],
            dtype=x.dtype,
            device=x.device
        ).unsqueeze(0)

        kernel_y = torch.tensor(
            [[[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]]],
            dtype=x.dtype,
            device=x.device
        ).unsqueeze(0)

        kernel_x = kernel_x.repeat(c, 1, 1, 1)
        kernel_y = kernel_y.repeat(c, 1, 1, 1)

        grad_x = F.conv2d(x, kernel_x, padding=1, groups=c)
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=c)

        edge = torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)

        edge = edge / (edge.amax(dim=(-2, -1), keepdim=True) + self.eps)

        return edge

    def boundary_dice_loss(self, pred_prob, y_true):
        pred_edge = self.get_sobel_edge(pred_prob)
        true_edge = self.get_sobel_edge(y_true)

        dims = (-2, -1)

        inter = (pred_edge * true_edge).sum(dims)
        union = pred_edge.sum(dims) + true_edge.sum(dims)

        dice = (2 * inter + self.smooth) / (union + self.smooth + self.eps)

        return 1 - dice.mean()

    def forward(self, y_pred, y_true):
        y_true = y_true.type_as(y_pred)

        bce = F.binary_cross_entropy_with_logits(
            y_pred,
            y_true,
            reduction="none"
        )

        prob = torch.sigmoid(y_pred)

        pt = prob * y_true + (1 - prob) * (1 - y_true)

        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)

        focal_weight = alpha_t * (1 - pt).pow(self.gamma)

        edge = self.get_morph_edge(y_true)

        structure_weight = (
            1.0
            + self.edge_weight * edge
            + self.fg_weight * y_true
        )

        focal_loss = structure_weight * focal_weight * bce
        focal_loss = focal_loss.mean()

        boundary_loss = self.boundary_dice_loss(prob, y_true)

        return focal_loss + self.boundary_weight * boundary_loss


class bce_dice(nn.Module):
    def __init__(self, args):
        super(bce_dice, self).__init__()

        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()

        self.focal_fn = StructureAwareFocalLossWithLogits(
            alpha=getattr(args, "Focal_alpha", 0.75),
            gamma=getattr(args, "Focal_gamma", 2.0),
            edge_weight=getattr(args, "Focal_edge_weight", 1.0),
            fg_weight=getattr(args, "Focal_fg_weight", 0.5),
            boundary_weight=getattr(args, "Focal_boundary_weight", 0.2)
        )

        self.args = args

    def forward(self, y_pred, y_true):
        y_true = y_true.type_as(y_pred)

        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        focal = self.focal_fn(y_pred, y_true)

        return (
            self.args.BCELoss_ratio * bce
            + self.args.DiceLoss_ratio * dice
            + self.args.FocalLoss_ratio * focal
        )



def build_MixerCSeg(args):
    device = torch.device(args.device)
    args.device = torch.device(args.device)

    embed_dim=[16,32,64,128]

    depths = [1,1,1,1]
    state_dim=[8,8,16,16]

    backbone = VSSEncoder(
        in_dim=3,
        embed_dim=embed_dim,
        depths=depths,
        mlp_ratio=2.,
        state_dim=state_dim,
        )
    model = MixerCSeg(backbone, embed_dim, args).to(device)

    criterion = bce_dice(args)
    criterion.to(device)
    
    return model, criterion

