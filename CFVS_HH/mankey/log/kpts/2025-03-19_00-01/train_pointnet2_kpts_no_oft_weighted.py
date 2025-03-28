import os
'''HYPER PARAMETER'''
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np

import datetime
import logging
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm

import config.parameter as parameter
from dataproc.xyzrot_pcd_supervised_db import SpartanSupvervisedKeypointDBConfig, SpartanSupervisedKeypointDatabase
from dataproc.supervised_keypoint_pcd_loader import SupervisedKeypointDatasetConfig, SupervisedKeypointDataset

from models.loss import RMSELoss
from models.utils import compute_rotation_matrix_from_ortho6d
from models.loss import OFLoss, WeightedOFLoss

from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    #parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='32 24batch size in training')
    parser.add_argument('--model', default='pointnet2_kpts', help='model name [default: pointnet_cls]')
    parser.add_argument('--out_channel', default=9, type=int)
    parser.add_argument('--epoch', default=600, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()

def construct_dataset(is_train: bool) -> (torch.utils.data.Dataset, SupervisedKeypointDatasetConfig):
    # Construct the db
    db_config = SpartanSupvervisedKeypointDBConfig()
    db_config.keypoint_yaml_name = 'peg_in_hole.yaml'
    # db_config.pdc_data_root = '/tmp2/r09944001/data/pdc'
    db_config.pdc_data_root = '/home/hcis/YongZhe/CFVS_HH/PDM_dataset'
    if is_train:
        # db_config.config_file_path = '/tmp2/r09944001/robot-peg-in-hole-task/mankey/config/insertion_20220616_coarse_noiseaug3_rotscale.txt'
        db_config.config_file_path = '/home/hcis/YongZhe/CFVS_HH/mankey/config/USB_coarse.txt'
    else:
        # db_config.config_file_path = '/tmp2/r09944001/robot-peg-in-hole-task/mankey/config/insertion_20220616_coarse_noiseaug3_rotscale.txt'
        db_config.config_file_path = '/home/hcis/YongZhe/CFVS_HH/mankey/config/USB_coarse.txt'
    database = SpartanSupervisedKeypointDatabase(db_config)

    # Construct torch dataset
    config = SupervisedKeypointDatasetConfig()
    config.network_in_patch_width = 256
    config.network_in_patch_height = 256
    config.network_out_map_width = 64
    config.network_out_map_height = 64
    config.image_database_list.append(database)
    config.is_train = is_train
    dataset = SupervisedKeypointDataset(config)
    return dataset, config

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
        

def test(model, loader, out_channel, criterion_rmse, criterion_cos, criterion_bce, criterion_kptof):
    '''
    xyz_error = []
    heatmap_error = []
    step_size_error = []
    '''
    kptof_error = []
    mask_error = []
    network = model.eval()

    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points = data[parameter.pcd_key].numpy()
        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        kpt_of_gt = data[parameter.kpt_of_key]
        pcd_centroid = data[parameter.pcd_centroid_key]
        pcd_mean = data[parameter.pcd_mean_key]
        heatmap_target = data[parameter.heatmap_key]
        
        if not args.use_cpu:
            points = points.cuda()
            kpt_of_gt = kpt_of_gt.cuda()
            pcd_centroid = pcd_centroid.cuda()
            pcd_mean = pcd_mean.cuda()
            heatmap_target = heatmap_target.cuda()
            
        kpt_of_pred, trans_of_pred, rot_of_pred, mean_kpt_pred, rot_mat_pred, confidence = network(points)

        # loss computation
        loss_kptof = criterion_kptof(kpt_of_pred, kpt_of_gt, heatmap_target).sum()
        loss_mask = criterion_rmse(confidence, heatmap_target)
        
        kptof_error.append(loss_kptof.item())
        mask_error.append(loss_mask.item())
        
    kptof_error = sum(kptof_error) / len(kptof_error)
    mask_error = sum(mask_error) / len(mask_error)
    
    return kptof_error, mask_error

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)


    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('kpts')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # Construct the dataset
    train_dataset, train_config = construct_dataset(is_train=True)
    # Random split
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, valid_set_size])
    # And the dataloader
    trainDataLoader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    validDataLoader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    '''MODEL LOADING'''
    out_channel = args.out_channel
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_pointnet2_kpts_no_oft_weighted.py', str(exp_dir))

    network = model.get_model()
    criterion_rmse = RMSELoss()
    criterion_cos = torch.nn.CosineSimilarity(dim=1)
    criterion_bce = torch.nn.BCELoss()
    criterion_kptof = WeightedOFLoss()
    network.apply(inplace_relu)

    if not args.use_cpu:
        network = network.cuda()
        criterion_rmse = criterion_rmse.cuda()
        criterion_cos = criterion_cos.cuda()
        criterion_bce = criterion_bce.cuda()
        criterion_kptof = criterion_kptof.cuda()
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        network.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_kptof_error = 99.9
    best_mask_error = 99.9

    '''TRANING'''
    logger.info('Start training...')
    writer = SummaryWriter(log_dir)
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        train_kptof_error = []
        train_mask_error = []
        network = network.train()

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points =  data[parameter.pcd_key].numpy()
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            heatmap_target = data[parameter.heatmap_key]
            kpt_of_gt = data[parameter.kpt_of_key]
            pcd_centroid = data[parameter.pcd_centroid_key]
            pcd_mean = data[parameter.pcd_mean_key]
 
            if not args.use_cpu:
                points = points.cuda()
                kpt_of_gt = kpt_of_gt.cuda()
                pcd_centroid = pcd_centroid.cuda()
                pcd_mean = pcd_mean.cuda()
                heatmap_target = heatmap_target.cuda()

            kpt_of_pred, trans_of_pred, rot_of_pred, mean_kpt_pred, rot_mat_pred, confidence = network(points)

            loss_kptof = criterion_kptof(kpt_of_pred, kpt_of_gt, heatmap_target).sum()
            loss_mask = criterion_rmse(confidence, heatmap_target)
            loss = loss_kptof + loss_mask
            loss.backward()
            optimizer.step()
            global_step += 1
            
            train_kptof_error.append(loss_kptof.item())
            train_mask_error.append(loss_mask.item())

        train_kptof_error = sum(train_kptof_error) / len(train_kptof_error)
        train_mask_error = sum(train_mask_error) / len(train_mask_error)
        
        log_string('Train Keypoint Offset Error: %f' % train_kptof_error)
        log_string('Train Mask Error: %f' % train_mask_error)
        
        writer.add_scalar('Training kpt loss', train_kptof_error, epoch)
        writer.add_scalar('Training heatmap loss', train_mask_error, epoch)

        with torch.no_grad():
            kptof_error, mask_error = test(network.eval(), validDataLoader, out_channel, criterion_rmse, criterion_cos, criterion_bce, criterion_kptof)
            
            log_string('Test Keypoint offset Error: %f, Mask Error: %f' % (kptof_error, mask_error))
            log_string('Best Keypoint offset Error: %f, Mask Error: %f' % (best_kptof_error, best_mask_error))

            writer.add_scalar('testing kpt loss', kptof_error, epoch)
            writer.add_scalar('testing heatmap loss', mask_error, epoch)
            
            if (kptof_error + mask_error) < (best_kptof_error + best_mask_error): 
                best_kptof_error = kptof_error
                best_mask_error = mask_error
                best_epoch = epoch + 1
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model_e_' + str(best_epoch) + '.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'kptof_error': kptof_error,
                    'mask_error': mask_error,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
