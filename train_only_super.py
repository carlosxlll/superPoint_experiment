#-*-coding:utf8-*-
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import yaml
import argparse
from tqdm import tqdm
from dataset.coco import COCODataset
from torch.utils.data import DataLoader
from model.superpoint_bn import SuperPointBNNet
from solver.loss import loss_func
import wandb
import cv2

# 加载监测图片
def load_img(resize_shape ,img_path, device='cpu'):
        img = cv2.imread(img_path, 0)#Gray image
        img = cv2.resize(img, resize_shape[::-1])
        img_tensor = torch.as_tensor(img.copy(), dtype=torch.float, device=device)
        img_tensor = img_tensor/255.
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        img_show = (img_tensor * 255).cpu().detach().numpy().squeeze().astype(int).astype(np.uint8)
        img_show = cv2.merge((img_show, img_show, img_show))
        return img_tensor, img_show

# 训练
def train_eval(model, dataloader, config, device):
    # 优化器

    optimizer = torch.optim.Adam(model.parameters(), lr=config['solver']['base_lr'])
    # wandb 初始化
    wandb.init(project='superpoint', name=config['solver']['experiment_name'], notes=config['solver']['experiment_detail'])
    # 加载监测图片
    img_tensor, img_show = load_img(config['data']['resize'],img_path=config['solver']['img_watch_path'], device=device)
    # 评估步长
    save_iter = int(0.25*len(dataloader['train']))#half epoch

    
    try:
        # 开始训练
        print('Start training...')
        for epoch in range(config['solver']['epoch']):
            # 学习率衰减
            # scheduler = StepLR(optimizer, step_size=config['solver']['lr_decay_epoch'], gamma=config['solver']['lr_decay'])
            # scheduler.step()
            model.train()
            mean_loss, mean_det_loss, mean_det_loss_warp, mean_weighted_des_loss, mean_positive_sum, mean_negative_sum = [], [], [], [], [], []
            for i, data in tqdm(enumerate(dataloader['train'])):
                print('epoch/all_epoch: {}/{}, iter/all_iter: {}/{}'.format(epoch, config['solver']['epoch'], i, len(dataloader['train'])))
                # 'det_info': {prob、prob_nms: (B, H, W), 'preb':一维，'logits': (B, 65, H/8, W/8)}
                # 'desc_info': {'desc_raw': (B, 256, H/8, W/8), desc: (B, 256, H, W)}
                raw_outputs = model(data['raw'])
                warp_outputs = model(data['warp'])
                # 原始图像特征点、描述子，warp图像特征点、描述子
                prob, desc, prob_warp, desc_warp = None, None, None, None
                prob, desc, prob_warp, desc_warp = raw_outputs['det_info'],raw_outputs['desc_info'],warp_outputs['det_info'],warp_outputs['desc_info']

                # 加载loss
                loss, det_loss, det_loss_warp, weighted_des_loss, positive_sum, negative_sum = loss_func(config['solver'], data, prob, desc, prob_warp, desc_warp, device)
                mean_loss.append(loss.item())
                mean_det_loss.append(det_loss.item())
                mean_det_loss_warp.append(det_loss_warp.item())
                mean_weighted_des_loss.append(weighted_des_loss.item())
                mean_positive_sum.append(positive_sum.item())
                mean_negative_sum.append(negative_sum.item())
                # wandb.log({'loss': loss.item(), 
                #            'det_loss': det_loss.item(), 
                #            'det_loss_warp': det_loss_warp.item(), 
                #            'weighted_des_loss': weighted_des_loss.item(),
                #            'positive_sum': positive_sum.item(),
                #            'negative_sum': negative_sum.item()}, step=(epoch*len(dataloader['train'])+i) )
                
                # 反向传播
                model.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 监测
                if (i%500==0):
                    model.eval()
                    # 记录loss
                    wandb.log({'lr':optimizer.state_dict()['param_groups'][0]['lr'],
                               'near500_mean_loss':np.mean(mean_loss),
                               'near500_mean_det_loss':np.mean(mean_det_loss), 
                               'near500_mean_det_loss_warp':np.mean(mean_det_loss_warp),
                               'near500_mean_weighted_des_loss':np.mean(mean_weighted_des_loss),
                               'near500_mean_positive_sum':np.mean(mean_positive_sum),
                               'near500_mean_negative_sum':np.mean(mean_negative_sum)}, step=(epoch*len(dataloader['train'])+i) )
                    mean_loss, mean_det_loss, mean_det_loss_warp, mean_weighted_des_loss, mean_positive_sum, mean_negative_sum = [], [], [], [], [], []
                    # # 记录图片
                    # prob = model(img_tensor)['det_info']['prob_nms']
                    # prob = prob.cpu().detach().numpy().squeeze()
                    # keypoints = np.where(prob > 0.015)
                    # keypoints = np.stack(keypoints).T
                    # img_show_now = img_show.copy()
                    # for kp in keypoints:
                    #     cv2.circle(img_show_now, (int(kp[1]), int(kp[0])), radius=1, color=(0, 255, 0))
                    # wandb.log({'img': wandb.Image(img_show_now)}, step=(epoch*len(dataloader['train'])+i) )
                    
                    model.train()

                # 评估
                if (i%save_iter==0 and i!=0) or ((i+1)==len(dataloader['train'])):
                    print('***************eval***************')
                    print('start eval...')
                    model.eval()
                    # 记录loss
                    mean_loss, mean_det_loss, mean_det_loss_warp, mean_weighted_des_loss, mean_positive_sum, mean_negative_sum = do_eval(model, dataloader['test'], config, device)
                    wandb.log({'eval_loss':mean_loss, 
                               'eval_det_loss':mean_det_loss, 
                               'eval_det_loss_warp':mean_det_loss_warp,
                               'eval_weighted_des_loss':mean_weighted_des_loss,
                               'eval_positive_sum':mean_positive_sum,
                               'eval_negative_sum':mean_negative_sum}, step=(epoch*len(dataloader['train'])+i))
                    mean_loss, mean_det_loss, mean_det_loss_warp, mean_weighted_des_loss , mean_positive_sum, mean_negative_sum = [], [], [], [], [], []
                    # 记录图片
                    prob = model(img_tensor)['det_info']['prob_nms']
                    prob = prob.cpu().detach().numpy().squeeze()
                    keypoints = np.where(prob > 0.015)
                    keypoints = np.stack(keypoints).T
                    img_show_now = img_show.copy()
                    for kp in keypoints:
                        cv2.circle(img_show_now, (int(kp[1]), int(kp[0])), radius=1, color=(0, 255, 0))
                    wandb.log({'img': wandb.Image(img_show_now)}, step=(epoch*len(dataloader['train'])+i) )
                    
                    model.train()
                    # 计算保存路径
                    save_dir = os.path.join(config['solver']['save_dir'], config['solver']['experiment_name'])
                    os.makedirs(save_dir, exist_ok=True)  # 创建文件夹，如果已经存在则不会报错
                    save_path = os.path.join(save_dir,'{}_{}_{}.pth'.format(config['solver']['model_name'],epoch, i//save_iter))
                    # 保存模型
                    torch.save(model.state_dict(), save_path)


    except KeyboardInterrupt:
        torch.save(model.state_dict(), "./export/key_interrupt_model.pth")

@torch.no_grad()
def do_eval(model, dataloader, config, device):
    mean_loss = []
    mean_det_loss = []
    mean_det_loss_warp = []
    mean_weighted_des_loss = []
    mean_positive_sum = []
    mean_negative_sum = []

    truncate_n = max(int(0.1 * len(dataloader)), 100)  # 0.1 of test dataset for eval
    for ind, data in tqdm(enumerate(dataloader)):
        if ind>truncate_n:
            break
        prob, desc, prob_warp, desc_warp = None, None, None, None
        raw_outputs = model(data['raw'])
        warp_outputs = model(data['warp'])
        prob, desc, prob_warp, desc_warp = raw_outputs['det_info'], raw_outputs['desc_info'], warp_outputs['det_info'],warp_outputs['desc_info']
        # compute loss
        loss, det_loss, det_loss_warp, weighted_des_loss, positive_sum, negative_sum = loss_func(config['solver'], data, prob, desc, prob_warp, desc_warp, device)

        mean_loss.append(loss.item())
        mean_det_loss.append(det_loss.item())
        mean_det_loss_warp.append(det_loss_warp.item())
        mean_weighted_des_loss.append(weighted_des_loss.item())
        mean_positive_sum.append(positive_sum.item())
        mean_negative_sum.append(negative_sum.item())

    mean_loss = np.mean(mean_loss)
    mean_det_loss = np.mean(mean_det_loss)
    mean_det_loss_warp = np.mean(mean_det_loss_warp)
    mean_weighted_des_loss = np.mean(mean_weighted_des_loss)
    mean_positive_sum = np.mean(mean_positive_sum)
    mean_negative_sum = np.mean(mean_negative_sum)

    return mean_loss, mean_det_loss, mean_det_loss_warp, mean_weighted_des_loss, mean_positive_sum, mean_negative_sum


if __name__=='__main__':

    torch.multiprocessing.set_start_method('spawn')

    # 配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='./config/superpoint_train.yaml')
    args = parser.parse_args()
    config_file = args.config
    assert (os.path.exists(config_file))
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # 创建保存模型的文件夹，设备
    if not os.path.exists(config['solver']['save_dir']):
        os.makedirs(config['solver']['save_dir'])
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    print('using... device' , device)

    # dataloader
    data_loaders = None
    datasets = {k: COCODataset(config['data'], is_train=True if k == 'train' else False, device=device)
                for k in ['test', 'train']}
    data_loaders = {k: DataLoader(datasets[k],
                                    config['solver']['{}_batch_size'.format(k)],
                                    collate_fn=datasets[k].batch_collator,
                                    shuffle=True) for k in ['train', 'test']}

    model = SuperPointBNNet(config['model'], device=device, using_bn=config['model']['using_bn'])

    # 加载预训练模型
    if os.path.exists(config['model']['pretrained_model']):
        pre_model_dict = torch.load(config['model']['pretrained_model'])
        model_dict = model.state_dict()
        for k,v in pre_model_dict.items():
            if k in model_dict.keys() and v.shape==model_dict[k].shape:
                model_dict[k] = v
        model.load_state_dict(model_dict)
    model.to(device)
    train_eval(model, data_loaders, config, device)
    print('Done')
