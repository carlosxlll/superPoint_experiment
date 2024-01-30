#-*-coding:utf-8-*-
import os
import yaml
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from dataset.patch import PatchesDataset
from dataset.synthetic_shapes import SyntheticShapes
from model.magic_point import MagicPoint
from model.superpoint_bn import SuperPointBNNet
from model_index.superpoint_bn_index import SuperPointBNNet_index
import cv2
import solver.detector_evaluation as ev
from utils.plt import plot_imgs

def export_detections_repeatability(config, device='cuda:7'):

    output_dir = os.path.join(config['data']['export_dir_root'], config['data']['experiment_name'], config['data']['export_dir'], config['data']['alteration'])
    print('Exporting to {}'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = device
    if config['data']['name']=='synthetic':
        dataset_ = SyntheticShapes(config['data'], task='training', device=device)
    elif config['data']['name'] == 'hpatches':
        dataset_ = PatchesDataset(config['data'],device=device)

    p_dataloader = DataLoader(dataset_, batch_size=1, shuffle=False, collate_fn=dataset_.batch_collator)

    if config['model']['name'] == 'superpoint_bn_raw':
        net = SuperPointBNNet(config['model'], device=device, using_bn=config['model']['using_bn'])
    elif config['model']['name'] == 'superpoint_bn_index':
        net = SuperPointBNNet_index(config['model'], device=device, using_bn=config['model']['using_bn'])
    elif config['model']['name'] == 'magicpoint':
        net = MagicPoint(config['model'], device=device)
        

    net.load_state_dict(torch.load(config['model']['pretrained_model'], map_location=device))
    net.to(device).eval()

    with torch.no_grad():
        for i, data in tqdm(enumerate(p_dataloader)):
            prob1 = net(data['img'])
            prob2 = net(data['warp_img'])
            ##
            pred = {'prob':prob1['det_info']['prob_nms'], 'warp_prob':prob2['det_info']['prob_nms'],
                    'homography': data['homography']}

            if not ('name' in data):
                pred.update(data)
            #to numpy
            pred = {k:v.cpu().numpy().squeeze() for k,v in pred.items()}
            filename = data['name'] if 'name' in data else str(i)
            filepath = os.path.join(output_dir, '{}.npz'.format(filename))
            np.savez_compressed(filepath, **pred)

    print(config['data']['alteration'], ' Done!')


def get_true_keypoints(exper_name, prob_thresh=0.5):
    def warp_keypoints(keypoints, H):
        warped_col0 = np.add(np.sum(np.multiply(keypoints, H[0, :2]), axis=1), H[0, 2])
        warped_col1 = np.add(np.sum(np.multiply(keypoints, H[1, :2]), axis=1), H[1, 2])
        warped_col2 = np.add(np.sum(np.multiply(keypoints, H[2, :2]), axis=1), H[2, 2])
        warped_col0 = np.divide(warped_col0, warped_col2)
        warped_col1 = np.divide(warped_col1, warped_col2)
        new_keypoints = np.concatenate([warped_col0[:, None], warped_col1[:, None]],
                                       axis=1)
        return new_keypoints

    def filter_keypoints(points, shape):
        """ Keep only the points whose coordinates are
        inside the dimensions of shape. """
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) & \
               (points[:, 1] >= 0) & (points[:, 1] < shape[1])
        return points[mask, :]

    true_keypoints = []
    for i in range(5):
        path = os.path.join(exper_name, str(i) + ".npz")
        data = np.load(path)
        shape = data['warped_prob'].shape

        # Filter out predictions
        keypoints = np.where(data['prob'] > prob_thresh)
        keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
        warped_keypoints = np.where(data['warped_prob'] > prob_thresh)
        warped_keypoints = np.stack([warped_keypoints[0], warped_keypoints[1]], axis=-1)

        # Warp the original keypoints with the true homography
        H = data['homography']
        true_warped_keypoints = warp_keypoints(keypoints[:, [1, 0]], H)
        true_warped_keypoints[:, [0, 1]] = true_warped_keypoints[:, [1, 0]]
        true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)
        true_keypoints.append((true_warped_keypoints[:, 0], true_warped_keypoints[:, 1]))

    return true_keypoints

def draw_keypoints(img, corners, color=(0, 255, 0), radius=3, s=3):
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners).T:
        cv2.circle(img, tuple(s*np.flip(c, 0)), radius, color, thickness=-1)
    return img

def select_top_k(prob, thresh=0, num=300):
    pts = np.where(prob > thresh)
    idx = np.argsort(prob[pts])[::-1][:num]
    pts = (pts[0][idx], pts[1][idx])
    return pts

def compute_repeatability(experiments, experiments_picture, confidence_thresholds=[0.015, 0.015, 0.015]):
    # 遍历列表，创建每个文件夹
    for path in experiments_picture:
        os.makedirs(path, exist_ok=True)

    ## show keypoints
    for i in range(4):
        for e, thresh, e_picture in zip(experiments, confidence_thresholds, experiments_picture):
            path = os.path.join(e, str(i) + ".npz")
            d = np.load(path)
            img = np.round(d['img']*255).astype(int).astype(np.uint8)
            warp_img = np.round(d['warp_img']*255).astype(int).astype(np.uint8)

            points1 = select_top_k(d['prob'], thresh=thresh)
            im1 = draw_keypoints(img, points1, (0, 255, 0))/255.

            points2 = select_top_k(d['warp_prob'], thresh=thresh)
            im2 = draw_keypoints(warp_img, points2, (0, 255, 0))/255.

            plot_imgs(i, e_picture, [im1, im2], ylabel=e, dpi=200, cmap='gray',
                      titles=[str(len(points1[0])) + ' points', str(len(points2[0])) + ' points'])

    ## compute repeatability
    print('Computing repeatability...')
    print('Confidence thresholds: {}'.format(1))
    for exp, thresh in zip(experiments, confidence_thresholds):
        repeatability = ev.compute_repeatability(exp, keep_k_points=300, distance_thresh=1) # TODO: change Distance threshold
        print('experiment: {}, thresh: {}'.format(exp, thresh))
        print('> {}: {}'.format(exp, repeatability))
    print('Confidence thresholds: {}'.format(3))
    for exp, thresh in zip(experiments, confidence_thresholds):
        repeatability = ev.compute_repeatability(exp, keep_k_points=300, distance_thresh=3) # TODO: change Distance threshold
        print('experiment: {}, thresh: {}'.format(exp, thresh))
        print('> {}: {}'.format(exp, repeatability))

if __name__ == '__main__':
    # 6.1保存检测结果
    with open('./config/6.1.detection_repeatability.yaml', 'r', encoding='utf8') as fin:
        config = yaml.safe_load(fin)
    for i in range(3):
        if i == 0 : config['data']['alteration'] = 'v'
        elif i == 1 : config['data']['alteration'] = 'i'
        elif i==2 : config['data']['alteration'] = 'all'
        export_detections_repeatability(config, device='cuda:7') # TODO: change device
  
    # 6.2保存检测结果的图片和计算重复率
    output_dir = os.path.join(config['data']['export_dir_root'], config['data']['experiment_name'], config['data']['export_dir'])
    print('Exporting to {}'.format(output_dir))
    experiments = [os.path.join(output_dir, 'v'), os.path.join(output_dir, 'i'), os.path.join(output_dir, 'all')]
    experiments_picture = [os.path.join(output_dir, 'v', 'picture'), os.path.join(output_dir, 'i', 'picture'), os.path.join(output_dir, 'all', 'picture')]
    compute_repeatability(experiments, experiments_picture, confidence_thresholds=[0.015, 0.015, 0.015])
 
    print('Done!')