#-*-coding:utf8-*-
import torch
from solver.nms import box_nms
from model_index.modules.cnn.index_vgg_backbone import VGGBackbone_index,VGGBackboneBN_index
from model_index.modules.cnn.index_cnn_heads import DetectorHead, DescriptorHead


class MagicPoint_index(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, config, input_channel=1, grid_size=8, using_bn=True, device='cpu'):
        super(MagicPoint_index, self).__init__()
        self.nms = config['nms']
        self.det_thresh = config['det_thresh']
        self.topk = config['topk']
        if using_bn:
            self.backbone = VGGBackboneBN_index(config['backbone']['vgg'], input_channel, device=device)
        else:
            self.backbone = VGGBackbone_index(config['backbone']['vgg'], input_channel, device=device)

        self.detector_head = DetectorHead(input_channel=128, grid_size=grid_size,using_bn=using_bn)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
        """
        if isinstance(x, dict):
            feat_map = self.backbone(x['img'])
        else:
            feat_map = self.backbone(x)
        outputs = self.detector_head(feat_map)

        prob = outputs['prob']
        if self.nms is not None:
            prob = [box_nms(p.unsqueeze(dim=0),
                            self.nms,
                            min_prob=self.det_thresh,
                            keep_top_k=self.topk).squeeze(dim=0) for p in prob]
            prob = torch.stack(prob)
            outputs.setdefault('prob_nms',prob)

        pred = prob[prob>=self.det_thresh]
        outputs.setdefault('pred', pred)

        return outputs

