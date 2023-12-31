import os
import sys
import yaml
import argparse
import logging
import math
import importlib
import datetime
import random
import munch
import time
import torch
import torch.optim as optim
import warnings
import shutil
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from dataset import MeshDataset
from utils.train_utils import *

def train():
    logging.info(str(args))
    metrics = ['cd_p', 'cd_t']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    data_dir = os.path.join('data', args.dataset)
    dataset = MeshDataset(os.path.join(data_dir, 'train_meshes/'), npoints=args.num_input_points, subsample=args.data_size, missing_percent=args.missing_percent, set_type='train')
    scale_factor = dataset.get_scale_factor()
    dataset_test = MeshDataset(os.path.join(data_dir, 'test_meshes/'), npoints=args.num_input_points, scale_factor=scale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = args.device
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = model_module.Model(args)
    net.to(device)
    if hasattr(model_module, 'weights_init'):
        net.apply(model_module.weights_init)

    cascade_gan = (args.model_name == 'cascade')
    net_d = None
    if cascade_gan:
        net_d = model_module.Discriminator(args)
        net_d.to(device)
        net_d.apply(model_module.weights_init)

    lr = args.lr
    if cascade_gan:
        lr_d = lr / 2

    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer(net.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    else:
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)

    if cascade_gan:
        optimizer_d = optim.Adam(net_d.parameters(), lr=lr_d, weight_decay=0.00001, betas=(0.5, 0.999))

    alpha = None
    if args.varying_constant:
        varying_constant_epochs = [int(ep.strip()) for ep in args.varying_constant_epochs.split(',')]
        varying_constant = [float(c.strip()) for c in args.varying_constant.split(',')]
        assert len(varying_constant) == len(varying_constant_epochs) + 1

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.load_state_dict(ckpt['net_state_dict'])
        if cascade_gan:
            net_d.load_state_dict(ckpt['D_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    epochs_since_best_cd_t = 0
    for epoch in range(args.start_epoch, args.nepoch):
        start_time = time.time()
        torch.cuda.empty_cache()
        train_loss_meter.reset()
        net.train()
        if cascade_gan:
            net_d.train()

        if args.varying_constant:
            for ind, ep in enumerate(varying_constant_epochs):
                if epoch < ep:
                    alpha = varying_constant[ind]
                    break
                elif ind == len(varying_constant_epochs)-1 and epoch >= ep:
                    alpha = varying_constant[ind+1]
                    break

        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            if cascade_gan:
                optimizer_d.zero_grad()

            pc, gt, names = data
            pc = pc.to(device)
            gt = gt.to(device)
            inputs = pc.contiguous() 
            out, loss, net_loss = net(inputs, gt, alpha=alpha)

            if cascade_gan:
                d_fake = generator_step(net_d, out, net_loss, optimizer)
                discriminator_step(net_d, inputs, d_fake, optimizer_d)
            else:
                train_loss_meter.update(net_loss.mean().item())
                net_loss.backward()
                optimizer.step()

            if i % args.step_interval_to_print == 0:
                logging.info(exp_name + ' train [%d: %d/%d]  loss_type: %s, fine_loss: %f total_loss: %f lr: %f' %
                             (epoch, i, len(dataset) / args.batch_size, args.loss, loss.mean().item(), net_loss.mean().item(), lr) + ' alpha: ' + str(alpha) + ' time: ' + str(time.time()-start_time)[:4] + ' track: ' + str(epochs_since_best_cd_t) )

        if epoch % args.epoch_interval_to_save == 0:
            save_model('%s/network.pth' % log_dir, net, net_d=net_d)
            logging.info("Saving net...")

        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            best_cd_t = val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses, device)
            if args.early_stop:
                if best_cd_t:
                    epochs_since_best_cd_t = 0
                else:
                    epochs_since_best_cd_t += 1
                if epochs_since_best_cd_t > args.early_stop_patience:
                    print("Early stopping epoch:", epoch)
                    break

    best_cd_t = val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses, device)
    return scale_factor


def val(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses, device):
    best_cd_t = False
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            pc, gt, names = data
            pc = pc.to(device)
            gt = gt.to(device)
            inputs = pc.contiguous()
            result_dict = net(inputs, gt, is_training=False)
            for k, v in val_loss_meters.items():
                v.update(result_dict[k].mean().item())
            # print(result_dict['out1'].shape)
            # input(result_dict['out2'].shape)

        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                save_model('%s/best_%s_network.pth' % (log_dir, loss_type), net)
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
                if loss_type == 'cd_t':
                    best_cd_t = True
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

        logging.info(curr_log)
        logging.info(best_log)
    return best_cd_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    if 'missing_percent' not in args:
        args['missing_percent'] = 0
    if 'data_size' not in args:
        args['data_size'] = -1

    print_time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        exp_name = args.model_name
        log_dir = os.path.join(args.work_dir, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
    # Update yaml in log dir
    scale_factor =  train()
    args['best_model_path'] = log_dir+'/best_cd_p_network.pth'
    args['scale_factor'] = scale_factor
    with open(os.path.join(log_dir, os.path.basename(config_path)), 'w') as f:
        yaml.dump(args, f)




