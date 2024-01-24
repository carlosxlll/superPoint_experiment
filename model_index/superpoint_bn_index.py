import torch
import torch.nn as nn
import sys
sys.path.append("/root/workspace/code/mine/superPoint_experiment")
from solver.nms import box_nms
from model_index.modules.cnn.index_vgg_backbone import VGGBackbone,VGGBackboneBN
from model_index.modules.cnn.index_cnn_heads import DetectorHead, DescriptorHead

class SuperPointBNNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, config, input_channel=1, grid_size=8, device='cpu', using_bn=True):
        super(SuperPointBNNet, self).__init__()
        self.nms = config['nms']
        self.det_thresh = config['det_thresh']
        self.topk = config['topk']
        if using_bn:
            self.backbone = VGGBackboneBN(config['backbone']['vgg'], input_channel, device=device)
        else:
            self.backbone = VGGBackbone(config['backbone']['vgg'], input_channel, device=device)
        ##
        self.detector_head = DetectorHead(input_channel=config['det_head']['feat_in_dim'],
                                          grid_size=grid_size, using_bn=using_bn)
        self.descriptor_head = DescriptorHead(input_channel=config['des_head']['feat_in_dim'],
                                              output_channel=config['des_head']['feat_out_dim'],
                                              grid_size=grid_size, using_bn=using_bn)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        if isinstance(x, dict):
            feat_map = self.backbone(x['img'])
        else:
            feat_map = self.backbone(x)
        det_outputs = self.detector_head(feat_map)

        prob = det_outputs['prob']
        if self.nms is not None:
            prob = [box_nms(p.unsqueeze(dim=0),
                            self.nms,
                            min_prob=self.det_thresh,
                            keep_top_k=self.topk).squeeze(dim=0) for p in prob]
            prob = torch.stack(prob)
            det_outputs.setdefault('prob_nms',prob)

        pred = prob[prob>=self.det_thresh]
        det_outputs.setdefault('pred', pred)

        desc_outputs = self.descriptor_head(feat_map)
        return {'det_info':det_outputs, 'desc_info':desc_outputs}



def test_all():
    def test_superpoint_bn():
        import yaml
        with open('/root/workspace/code/mine/superPoint_my/config/superpoint_train.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config = config['model']
        model = SuperPointBNNet(config)
        print(model)
        model.load_state_dict(torch.load('/root/workspace/code/mine/superPoint_my/superpoint_bn.pth'))
        print('Done')
    
        # 创建输入示例，这里使用随机生成的图像张量
        batch_size, channels, height, width = 2, 1, 256, 256
        input_tensor = torch.rand((batch_size, channels, height, width))

        # 调用模型的前向传播函数
        output = model(input_tensor)
        output_prob = output['det_info']
        output_desc = output['desc_info']

        # 打印输出的形状，以确保与预期一致
        print("Output logits and prob Shape:", output_prob['logits'].shape, output_prob['prob'].shape)
        print("Output Desc_raw and desc Shape:", output_desc['desc_raw'].shape, output_desc['desc'].shape)
    
    test_superpoint_bn()

if __name__=='__main__':
    test_all()
