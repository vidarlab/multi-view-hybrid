import argparse
import os
import logging

import torch
import torch.nn as nn

from model import MultiImageHybrid

from loss import MutualDistillationLoss
from engine import TrainerEngine, Evaluator
import numpy as np
from datasets import HotelsDataset
from utils import str2bool

def main(logger):

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_train = HotelsDataset(args.data_dir, split='train', n=args.n, train=True)
    dataset_val = HotelsDataset(args.data_dir, split='val', n=args.n, classes=dataset_train.classes, train=False)

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    model = MultiImageHybrid(args.architecture, num_classes=dataset_train.num_classes, n=args.n)
    model.cuda()

    print('Number of classes: ', dataset_train.num_classes)

    optimizer_model = torch.optim.SGD(params=model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer_model, max_lr=args.lr,
                                                    epochs=args.num_epochs,
                                                    div_factor=10,
                                                    steps_per_epoch=len(dataset_train) // args.batch_size,
                                                    final_div_factor=1000,
                                                    pct_start=5 / args.num_epochs, anneal_strategy='cos')

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    if args.use_mutual_distillation_loss:
        md_loss = MutualDistillationLoss(temp=args.md_temp, lambda_hyperparam=args.md_lambda)
    else:
        md_loss = None

    evaluator = Evaluator(model=model, n=args.num_images)
    trainer = TrainerEngine(model=model, lr_scheduler_model=scheduler, criterion=criterion, optimizer_model=optimizer_model, 
                            evaluator=evaluator, md_loss=md_loss, grad_clip_norm=args.grad_clip_norm, logger=logger,
                            save_dir=args.save_dir)

    trainer.train(loader_train, args.num_epochs, loader_val)
    logger.info('Training done!\n')

    best_weights = torch.load(f'{args.save_dir}/best.pth')
    model.load_state_dict(best_weights['model'])

    logger.info('Evaluating on test set:\n')
    dataset_test = HotelsDataset(args.data_dir, split='test', n=args.n, classes=dataset_train.classes, train=False)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    score_dict = evaluator.evaluate(loader_test)
    for view_type in score_dict:
        for metric in score_dict[view_type]:
            logger.info(f'Test {view_type} {metric}: {score_dict[view_type][metric]}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Multiview Model Training", allow_abbrev=False
    )
    parser.add_argument("--data_dir", default='/', type=str, help="location to images")
    parser.add_argument("--architecture", default='vit_small_r26_s32_224', type=str, help="model architecture")
    parser.add_argument('--pretrained_weights', default=True, type=str2bool, help='use pretrained weights')
    parser.add_argument('--save_dir', default='output', type=str, help='save location for model weights and log')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--momentum", default=.9, type=float, help="momentum")
    parser.add_argument("--num_epochs", default=50, type=int, help="number of epochs for training")
    parser.add_argument("--num_images", default=2, type=int, help="number of images per input")
    parser.add_argument("--num_workers", default=8, type=int, help="num dataloading workers")
    parser.add_argument('--use_mutual_distillation_loss', default=True, type=str2bool, help='use mutual distillation loss')
    parser.add_argument("--md_temp", default=4., type=float, help='mutual distillation temperature')
    parser.add_argument("--md_lambda", default=.1, type=float, help='mutual distillation temperature lambda hyperparm')
    parser.add_argument("--grad_clip_norm", default=80., type=float, help='grad clip norm value')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logfile = f'{args.save_dir}/log.txt'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(logfile), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    for param in vars(args):
        logger.info(f'{param}: {getattr(args, param)}')

    PARAMS = {}
    for arg in vars(args):
        PARAMS[arg] = getattr(args, arg)

    args.n = args.num_images
    main(logger=logger)