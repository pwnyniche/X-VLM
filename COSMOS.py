import argparse
import os
import sys
import math

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models import load_pretrained
from models.model_cosmos import XVLM

from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 500   
    step_size = 100

    for i, (image, text1, text2, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # images = torch.cat([image0, image1], dim=0)
        image, targets = image.to(device), targets.to(device)   
        
        text_inputs1 = tokenizer(text1, padding="longest", return_tensors="pt", max_length=512,truncation=True).to(device)  
        text_inputs2 = tokenizer(text2, padding="longest", return_tensors="pt", max_length=512,truncation=True).to(device)
        loss = model(image, text_inputs1.input_ids, text_inputs1.attention_mask, 
            text_inputs2.input_ids, text_inputs2.attention_mask, 
            targets=targets, train=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device):
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 500

    for (image, text1, text2, targets) in metric_logger.log_every(data_loader, print_freq, header):
        image, targets = image.to(device), targets.to(device)   
        text_inputs1 = tokenizer(text1, padding="longest", return_tensors="pt", max_length=512,truncation=True).to(device)  
        text_inputs2 = tokenizer(text2, padding="longest", return_tensors="pt", max_length=512,truncation=True).to(device)

        prediction = model(image, text_inputs1.input_ids, text_inputs1.attention_mask, 
            text_inputs2.input_ids, text_inputs2.attention_mask, 
            targets=targets, train=False)
 
        _, pred_class = prediction.max(1)
        accuracy = (targets == pred_class).sum() / targets.size(0)
        
        metric_logger.meters['acc'].update(accuracy.item(), n=image.size(0))
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    
    
def main(args, config):
    # Namespace(checkpoint='4m_base_model_state_step_199999.th', config='configs/NLVR.yaml', 
    # output_dir='output/nlvr', device='cuda', seed=42, world_size=1, dist_url='env://', 
    # distributed=True, load_nlvr_pretrain=False, bs=-1, evaluate=False)
    # print(config)
    # {'train_file': ['data/finetune/nlvr_train.json'], 'val_file': ['data/finetune/nlvr_dev.json'], 
    # 'test_file': ['data/finetune/nlvr_test.json'], 'image_root': 'images/nlvr2/', 
    # 'vision_config': 'configs/config_swinB_384.json', 'use_clip_vit': False, 
    # 'use_swin': True, 'image_res': 384, 'patch_size': 32, 'use_roberta': False, 
    # 'text_config': 'configs/config_bert.json', 'text_encoder': 'data/bert-base-uncased', 
    # 'batch_size': 10, 'optimizer': {'opt': 'adamW', 'lr': 3e-05, 'weight_decay': 0.01, 'lr_mult': 2}, 'schedular': {'sched': 'linear', 'lr': 3e-05, 'epochs': 10, 'num_warmup_steps': 0.1}}
    
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.bs > 0:
        config['batch_size'] = args.bs // world_size

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('cosmos', config)
    datasets = [train_dataset, val_dataset, test_dataset]

    train_dataset_size = len(train_dataset)
    train_batch_size = config['batch_size']
    world_size = utils.get_world_size()

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {train_batch_size} x {world_size}")
        print(f"### test data {len(test_dataset)}", flush=True)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)     
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']] * 3,
                                                          num_workers=[4, 4, 4], is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])


    print("Creating model")
    model = XVLM(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    

    if config['use_roberta']:
        tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

    print("### output_dir, ", args.output_dir, flush=True)
    start_time = time.time()

    if args.evaluate:
        print("Start evaluating")
        val_stats = evaluate(model, val_loader, tokenizer, device)
        test_stats = evaluate(model, test_loader, tokenizer, device)

        if utils.is_main_process():
            log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

            print(log_stats)

        dist.barrier()

    else:
        print("Start training")
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(train_batch_size*world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        max_epoch = config['schedular']['epochs']

        best = 0
        best_epoch = 0

        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler)
            val_stats = evaluate(model, val_loader, tokenizer, device)
            test_stats = evaluate(model, test_loader, tokenizer, device)

            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                if float(val_stats['acc']) > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        # 'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = float(val_stats['acc'])
                    best_epoch = epoch

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            dist.barrier()

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d" % best_epoch)

            os.system(f"cat {args.output_dir}/log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', default='./configs/COSMOS.yaml')
    parser.add_argument('--output_dir', default='output/cosmos')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)