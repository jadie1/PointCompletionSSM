import os
import sys
import importlib
import argparse
import logging
import munch
import yaml
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from dataset import MeshDataset
from utils.vis_utils import plot_single_pcd
from utils.train_utils import *

def test(test_set):
    data_dir = os.path.join('data', args.dataset)
    if test_set == 'test':
        dataset_test = MeshDataset(os.path.join(data_dir, 'test_meshes/'), npoints=args.num_input_points, scale_factor=args.scale_factor)
    elif test_set == 'train':
        dataset_test = MeshDataset(os.path.join(data_dir, 'train_meshes/'), npoints=args.num_input_points, scale_factor=args.scale_factor)
    else:
        print("Error: Test set unrecognized.")
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    device = 'cuda:0'
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = model_module.Model(args)
    net.to(device)
    net.load_state_dict(torch.load(args.best_model_path, map_location=args.device)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()

    metrics = ['cd_p', 'cd_t', 'f1']
    test_loss_meters = {m: AverageValueMeter() for m in metrics}
    idx_to_plot = [0,1]
    
    logging.info('Testing...')
    if args.save_vis:
        pic_dir = os.path.join(log_dir, test_set, 'pics')
        save_gt_path = os.path.join(pic_dir, 'gt')
        save_partial_path = os.path.join(pic_dir, 'partial')
        save_coarse_pic_path = os.path.join(pic_dir, 'coarse_completion')
        save_dense_pic_path = os.path.join(pic_dir, 'dense_completion')
        os.makedirs(save_gt_path, exist_ok=True)
        os.makedirs(save_partial_path, exist_ok=True)
        os.makedirs(save_coarse_pic_path, exist_ok=True)
        os.makedirs(save_dense_pic_path, exist_ok=True)
    if args.save_predictions:
        pred_dir = os.path.join(log_dir, test_set, 'predictions')
        save_coarse_path = os.path.join(pred_dir, 'coarse')
        save_dense_path = os.path.join(pred_dir, 'dense')
        os.makedirs(save_coarse_path, exist_ok=True)
        os.makedirs(save_dense_path, exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            
            pc, gt, names = data
            pc = pc.to(device)
            gt = gt.to(device)
            inputs = pc.contiguous()
            result_dict = net(inputs, gt, is_training=False)
            for k, v in test_loss_meters.items():
                v.update(result_dict[k].mean().item())

            # for j, l in enumerate(label):
            #     for ind, m in enumerate(metrics):
            #         test_loss_cat[int(l), ind] += result_dict[m][int(j)]

            if i % args.step_interval_to_print == 0:
                logging.info('test [%d/%d]' % (i, dataset_length / args.batch_size))

            if args.save_vis:
                for j in range(args.batch_size):
                    idx = i * args.batch_size + j
                    if idx in idx_to_plot:
                        pic = names[j]+'.png'
                        plot_single_pcd(result_dict['out1'][j].cpu().numpy(), os.path.join(save_coarse_pic_path, pic))
                        plot_single_pcd(result_dict['out2'][j].cpu().numpy(), os.path.join(save_dense_pic_path, pic))
                        plot_single_pcd(gt[j].cpu().numpy(), os.path.join(save_gt_path, pic))
                        plot_single_pcd(pc[j].cpu().numpy(), os.path.join(save_partial_path, pic))
            if args.save_predictions:
                for j in range(len(names)):
                    np.savetxt(os.path.join(save_coarse_path, names[j]+'.particles'), result_dict['out1'][j].cpu().numpy()*args.scale_factor)
                    np.savetxt(os.path.join(save_dense_path, names[j]+'.particles'), result_dict['out2'][j].cpu().numpy()*args.scale_factor)

        logging.info('Overview results:')
        overview_log = ''
        for metric, meter in test_loss_meters.items():
            overview_log += '%s: %f ' % (metric, meter.avg)
        logging.info(overview_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-t', '--test_set', help='train or test', default='test')
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    if not args.best_model_path:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.best_model_path)
    log_dir = os.path.dirname(args.best_model_path)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])

    test(arg.test_set)
