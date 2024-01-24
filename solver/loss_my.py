import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append('/root/workspace/code/mine/superPoint_experiment/')
from utils.keypoint_op import warp_points
from utils.tensor_op import pixel_shuffle_inv

def detector_loss(keypoint_map, logits, valid_mask=None, grid_size=8, device='cpu'):
    '''
    :param keypoint_map: 特征点图(B,H,W)
    :param logits: 特征点图预测(B,65,H/8,W/8)
    :param valid_mask: 有效掩码(B,H,W)
    :param grid_size: 网格大小
    :param device: 设备
    '''
    labels = keypoint_map.unsqueeze(1).float() # (B,1,H,W)
    labels = pixel_shuffle_inv(labels, grid_size) # (B,64,H/8,W/8)
    B,C,h,w = labels.shape
    labels = torch.cat([2*labels, torch.ones(B,1,h,w,device=device)], dim=1) # (B,65,H/8,W/8)
    labels = torch.argmax(labels+torch.zeros(labels.shape,device=device).uniform_(0,1), dim=1) # (B,H/8,W/8)

    # 有效掩码
    valid_mask = torch.ones_like(keypoint_map,device=device) if valid_mask is None else valid_mask
    valid_mask = valid_mask.unsqueeze(1) # (B,1,H,W)
    valid_mask = pixel_shuffle_inv(valid_mask, grid_size) # (B,64,H/8,W/8)
    valid_mask = torch.prod(valid_mask, dim=1).unsqueeze(dim=1).type(torch.float32) # 乘积(B,1,H/8,W/8)

    # 交叉熵损失
    ce_loss = F.cross_entropy(logits, labels, reduction='none') # (B,H/8,W/8)
    valid_mask = valid_mask.squeeze(dim=1) # (B,H/8,W/8)
    loss = torch.divide(torch.sum(ce_loss*valid_mask, dim=(1,2)),
                        torch.sum(valid_mask+1e-6, dim=(1,2))) # (B,)
    loss = torch.mean(loss) # 一维

    return loss

def descriptor_loss(config, descriptors, warped_descriptors, homographies,
                    valid_mask=None,device='cpu'):
    '''
    :param config: 配置文件
    :param descriptors: 描述子(B,256,H/8,W/8)
    :param warped_descriptors: 透视图像描述子(B,256,H/8,W/8)
    :param homographies: 透视矩阵(B,3,3)
    :param valid_mask: 有效掩码(B,H,W)
    :param device: 设备
    '''
    # 读取配置
    grid_size = config['grid_size']
    positive_margin = config['loss']['positive_margin']
    negative_margin = config['loss']['negative_margin']
    lambda_d = config['loss']['lambda_d']
    lambda_loss = config['loss']['lambda_loss']

    (B,C,h,w) = descriptors.shape
    # 坐标网格
    coord_cells = torch.stack(torch.meshgrid(torch.arange(h, device=device),
                            torch.arange(w, device=device),indexing='ij'), dim=-1) # (h,w,2)
    coord_cells = coord_cells * grid_size + grid_size // 2 # (h,w,2)
    warped_coord_cells = warp_points(coord_cells.reshape(-1,2), homographies, device=device) # (B*h*w,2)
    coord_cells = torch.reshape(coord_cells, [1,1,1,h,w,2]).type(torch.float32) # (1,1,1,h,w,2)
    warped_coord_cells = torch.reshape(warped_coord_cells, [B,h,w,1,1,2]) # (B,h,w,1,1,2)
    cell_distances = torch.norm(coord_cells-warped_coord_cells, dim=-1,p=2) # (B,h,w,h,w)
    s = (cell_distances<(grid_size-0.5)).type(torch.float32) # (B,h,w,h,w)

    descriptors = torch.reshape(descriptors, [B, -1, h,w,1,1]) # (B,256,h,w,1,1)
    descriptors = F.normalize(descriptors, p=2, dim=1) # (B,256,h,w,1,1)
    warped_descriptors = torch.reshape(warped_descriptors, [B, -1, 1,1,h,w]) # (B,256,1,1,h,w)
    warped_descriptors = F.normalize(warped_descriptors, p=2, dim=1) # (B,256,1,1,h,w)
    dot_product_desc = torch.sum(descriptors*warped_descriptors, dim=1) # (B,h,w,h,w)
    dot_product_desc = F.relu(dot_product_desc) # (B,h,w,h,w)

    # l2bnorm损失
    dot_product_desc = torch.reshape(F.normalize(torch.reshape(dot_product_desc,[B,h,w,h*w]), p=2, dim=3), 
                                    [B,h,w,h,w]) # (B,h,w,h,w)
    dot_product_desc = torch.reshape(F.normalize(torch.reshape(dot_product_desc,[B,h*w,h,w]), p=2, dim=1), 
                                    [B,h,w,h,w]) # (B,h,w,h,w)
    
    positive_dist = torch.maximum(torch.tensor(0.,device=device), 
                                  positive_margin-dot_product_desc) # (B,h,w,h,w)
    negative_dist = torch.maximum(torch.tensor(0.,device=device),
                                  dot_product_desc-negative_margin) # (B,h,w,h,w)
    
    loss = lambda_d * s * positive_dist + (1-s) * negative_dist # (B,h,w,h,w)

    # 有效掩码
    




def precision_recall(pred, keypoint_map, valid_mask):
    pass

def loss_func(config, data, prob, desc=None, prob_warp=None,
              desc_warp=None, device='cpu'):
    '''
    :param config: 配置文件
    :param data: 原始数据{'raw': {'img': (B,1,H,W), 'kpts_map': (B,H,W), 'mask': (B,H,W)}, 
                         'warp': 同raw, 
                         'homography': (B,3,3)}
    :param prob: 原始图像特征点{'prob': (B,H,W,), 'prob_nms': (B,H,W,), 'preb': 一维, 'logits': (B,65,H/8,W/8)} 
    :param desc: 原始图像描述子{'desc_raw': (B,256,H/8,W/8), 'desc': (B,256,H,W)}
    :param prob_warp: 对应透视图像特征点{'prob': (B,H,W,), 'prob_nms': (B,H,W,), 'preb': 一维, 'logits': (B,65,H/8,W/8)}
    :param desc_warp: 对应透视图像描述子{'desc_raw': (B,256,H/8,W/8), 'desc': (B,256,H,W)}
    :param device: 设备
    '''
    # 检测点损失
    det_loss = detector_loss(keypoint_map=data['raw']['kpts_map'],
                             logits=prob['logits'],
                             valid_mask=data['raw']['mask'],
                             grid_size=config['grid_size'],
                             device=device)
    # 如果没有描述子，直接返回检测点损失
    if desc is None or prob_warp is None or desc_warp is None:
        return det_loss
    
    # 对应透视图像检测点损失
    det_loss_warp = detector_loss(keypoint_map=data['warp']['kpts_map'],
                                  logits=prob_warp['logits'],
                                  valid_mask=data['warp']['mask'],
                                  grid_size=config['grid_size'],
                                  device=device)
    # 描述子损失
    desc_loss = descriptor_loss(config=config,
                                descriptors=desc['desc_raw'],
                                warped_descriptors=desc_warp['desc_raw'],
                                homographies=data['homography'],
                                valid_mask=data['warp']['mask'],
                                device=device)
    
    # 返回检测点损失+对应透视图像检测点损失+描述子损失
    loss = det_loss + det_loss_warp + desc_loss
    a, b, c = det_loss.item(), det_loss_warp.item(), desc_loss.item()
    print('det_loss: {:.4f}, det_loss_warp: {:.4f}, desc_loss: {:.4f}, total: {:.4f}'.format(a, b, c, a+b+c))
    return loss


def test_all():
    def test_det_loss():
        keypoint_map = torch.randn(10,480,640)
        logits = torch.randn(10,65,60,80)
        loss = detector_loss(keypoint_map, logits)
        print(loss)
    

    def test_desc_loss():
        config = {'grid_size': 8, 
                  'loss': {'positive_margin': 1, 'negative_margin': 0.2, 'lambda_d': 0.05, 'lambda_loss': 1000}}
        descriptors = torch.randn(10,256,2,2)
        warped_descriptors = torch.randn(10,256,2,2)
        homographies = torch.randn(10,3,3)
        loss = descriptor_loss(config, descriptors, warped_descriptors, homographies)
        # print(loss)

    test_det_loss()
    test_desc_loss()

if __name__ == '__main__':
    test_all()