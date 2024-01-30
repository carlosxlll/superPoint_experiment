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
from model_index.magic_point_index import MagicPoint_index
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


def train_eval(model, dataloader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['solver']['base_lr'])
    # wandb 初始化
    wandb.init(project='superpoint', name=config['solver']['experiment_name'], notes=config['solver']['experiment_detail'])
    # 加载监测图片
    img_tensor, img_show = load_img(config['data']['resize'],img_path=config['solver']['img_watch_path'], device=device)
    save_iter = int(0.5*len(dataloader['train']))#half epoch
    try:
        # start training
        print("***************train***************")
        print('Start training...')
        for epoch in range(config['solver']['epoch']):
            model.train()
            mean_loss = []
            for i, data in tqdm(enumerate(dataloader['train'])):
                
                print('epoch/all_epoch: {}/{}, iter/all_iter: {}/{}'.format(epoch, config['solver']['epoch'], i, len(dataloader['train'])))
                prob, desc, prob_warp, desc_warp = None, None, None, None
                data['raw'] = data['warp']
                data['warp'] = None
                raw_outputs = model(data['raw'])
                prob = raw_outputs #train magicpoint

                ##loss
                loss = loss_func(config['solver'], data, prob, desc,
                                 prob_warp, desc_warp, device)

                mean_loss.append(loss.item())

                #reset
                model.zero_grad()
                loss.backward()
                optimizer.step()


                # if (i%500==0):
                #     wandb.log({'lr':optimizer.state_dict()['param_groups'][0]['lr'],
                #                'near500_mean_loss':np.mean(mean_loss)}, step=(epoch*len(dataloader['train'])+i) )
                #     mean_loss = []

                if (i%1==0):
                    wandb.log({'lr':optimizer.state_dict()['param_groups'][0]['lr'],
                               'near500_mean_loss':np.mean(mean_loss)}, step=(epoch*len(dataloader['train'])+i) )
                    mean_loss = []

                ##do evaluation
                if (i%save_iter==0 and i!=0) or (i+1)==len(dataloader['train']):
                    print('***************eval***************')
                    print('start eval...')
                    model.eval()
                    eval_loss = do_eval(model, dataloader['test'], config, device)
                    wandb.log({'eval_loss':eval_loss}, step=(epoch*len(dataloader['train'])+i))
                    mean_loss = []

                    # 记录图片
                    prob = model(img_tensor)['prob_nms']
                    prob = prob.cpu().detach().numpy().squeeze()
                    keypoints = np.where(prob > 0.015)
                    keypoints = np.stack(keypoints).T
                    img_show_now = img_show.copy()
                    for kp in keypoints:
                        cv2.circle(img_show_now, (int(kp[1]), int(kp[0])), radius=1, color=(0, 255, 0))
                    wandb.log({'img': wandb.Image(img_show_now)}, step=(epoch*len(dataloader['train'])+i) )

                    model.train()

                    # 保存模型
                    save_dir = os.path.join(config['solver']['save_dir'], config['solver']['experiment_name'])
                    os.makedirs(save_dir, exist_ok=True)  # 创建文件夹，如果已经存在则不会报错
                    save_path = os.path.join(save_dir,'{}_{}_{}.pth'.format(config['solver']['model_name'],epoch, i))
                    torch.save(model.state_dict(), save_path)
                    print('save model to {}'.format(save_path)) 
                    print('switch to train mode...')
                    print("***************train***************")

    except KeyboardInterrupt:
        save_dir = os.path.join(config['solver']['save_dir'], config['solver']['experiment_name'])
        os.makedirs(save_dir, exist_ok=True)  # 创建文件夹，如果已经存在则不会报错
        save_path = os.path.join(save_dir,'{}_key_interrupt_model.pth'.format(config['solver']['model_name']))
        torch.save(model.state_dict(), save_path)

@torch.no_grad()
def do_eval(model, dataloader, config, device):
    mean_loss = []
    truncate_n = max(int(0.1 * len(dataloader)), 100)  # 0.1 of test dataset for eval

    for ind, data in tqdm(enumerate(dataloader)):
        if ind>truncate_n:
            break
        prob, desc, prob_warp, desc_warp = None, None, None, None
        data['raw'] = data['warp']
        data['warp'] = None
        raw_outputs = model(data['raw'])
        prob = raw_outputs

        # compute loss
        loss = loss_func(config['solver'], data, prob, desc,
                         prob_warp, desc_warp, device)

        mean_loss.append(loss.item())
    mean_loss = np.mean(mean_loss)

    return mean_loss


if __name__=='__main__':

    torch.multiprocessing.set_start_method('spawn')

    # 读取配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='./config/3.magicBN_index_coco.yaml')
    args = parser.parse_args()
    config_file = args.config
    print('loading config file from {}'.format(config_file))
    assert (os.path.exists(config_file))
    with open(config_file, 'r') as fin:
        config = yaml.safe_load(fin)

    # 创建保存模型的文件夹，设备
    if not os.path.exists(config['solver']['save_dir']):
        os.makedirs(config['solver']['save_dir'])
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print('using... device' , device)

    ##Make Dataloader
    data_loaders = None
    datasets = {k: COCODataset(config['data'], is_train=True if k == 'train' else False, device=device)
                for k in ['test', 'train']}
    data_loaders = {k: DataLoader(datasets[k],
                                    config['solver']['{}_batch_size'.format(k)],
                                    collate_fn=datasets[k].batch_collator,
                                    shuffle=True) for k in ['train', 'test']}

    ##Make model
    model = MagicPoint_index(config['model'], using_bn=True, device=device)

    ##Load Pretrained Model
    if os.path.exists(config['model']['pretrained_model']):
        print('loading pretrained model from {}'.format(config['model']['pretrained_model']))
        model.load_state_dict(torch.load(config['model']['pretrained_model']))
        # pre_model_dict = torch.load(config['model']['pretrained_model'])
        # model_dict = model.state_dict()
        # for k,v in pre_model_dict.items():
        #     if k in model_dict.keys() and v.shape==model_dict[k].shape:
        #         model_dict[k] = v
        # model.load_state_dict(model_dict)
    model.to(device)
    train_eval(model, data_loaders, config)
    print('Done')
