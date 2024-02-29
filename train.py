"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import shutil
import importlib
from argparse import ArgumentParser

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import SSD, SSDLite, ResNet, MobileNetV2
from src.utils import generate_dboxes, Encoder, coco_classes
from src.transform import SSDTransformer
from src.loss import Loss
from src.process import train, evaluate, cognata_eval
from src.dataset import collate_fn, CocoDataset, Cognata, prepare_cognata, train_val_split
from torchinfo import summary 

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def prepare(dataset, params, rank, world_size):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, sampler=sampler, **params)
    
    return dataloader

def get_args():
    parser = ArgumentParser(description="Implementation of SSD")
    parser.add_argument("--data-path", type=str, default="/coco",
                        help="the root folder of dataset")
    parser.add_argument("--save-folder", type=str, default="trained_models",
                        help="path to folder containing model checkpoint file")
    parser.add_argument("--log-path", type=str, default="tensorboard/SSD")

    parser.add_argument("--model", type=str, default="ssd", choices=["ssd", "ssdlite"],
                        help="ssd-resnet50 or ssdlite-mobilenetv2")
    parser.add_argument("--epochs", type=int, default=65, help="number of total epochs to run")
    parser.add_argument("--batch-size", type=int, default=32, help="number of samples for each iteration")
    parser.add_argument("--multistep", nargs="*", type=int, default=[43, 54],
                        help="epochs at which to decay learning rate")
    parser.add_argument("--amp", action='store_true', help="Enable mixed precision training")

    parser.add_argument("--lr", type=float, default=2.6e-3, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum argument for SGD optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="momentum argument for SGD optimizer")
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument("--dataset", default='Cognata', type=str)
    parser.add_argument("--config", default='config', type=str)
    parser.add_argument("--save-path", default='SSD.pth', type=str)
    
    args = parser.parse_args()
    return args


def main(rank, opt, world_size):
    setup(rank, world_size)

    train_params = {"batch_size": opt.batch_size,
                    "num_workers": opt.num_workers,
                    "collate_fn": collate_fn}

    test_params = {"batch_size": opt.batch_size,
                   "num_workers": opt.num_workers,
                   "collate_fn": collate_fn}

    config = importlib.import_module('config.' + opt.config)
    image_size = config.model['image_size']
    num_classes = len(coco_classes)
    if opt.model == "ssd":
        dboxes = generate_dboxes(config.model, model="ssd")
    else:
        dboxes = generate_dboxes(model="ssdlite")
    if opt.dataset == 'Cognata':
        folders = config.dataset['folders']
        cameras = config.dataset['cameras']
        files, label_map, label_info = prepare_cognata(opt.data_path, folders, cameras)
        files = train_val_split(files)
        train_set = Cognata(label_map, label_info, files['train'], SSDTransformer(dboxes, image_size, val=False))
        test_set = Cognata(label_map, label_info, files['val'], SSDTransformer(dboxes, image_size, val=True))
        num_classes = len(label_map.keys())
        print(label_map)
        print(label_info)
    elif opt.dataset == 'Coco':
        train_set = CocoDataset(opt.data_path, 2017, "train", SSDTransformer(dboxes, image_size, val=False))
        test_set = CocoDataset(opt.data_path, 2017, "val", SSDTransformer(dboxes, image_size, val=True))
    if opt.model == "ssd":
        model = SSD(config.model, backbone=ResNet(config.model), num_classes=num_classes)
    else:
        model = SSDLite(backbone=MobileNetV2(), num_classes=len(coco_classes))
    train_loader = prepare(train_set, train_params, rank, world_size)
    test_loader = prepare(test_set, test_params, rank, world_size)

    encoder = Encoder(dboxes)

    opt.lr = opt.lr * (opt.batch_size / 32)
    criterion = Loss(dboxes)

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=opt.multistep, gamma=0.1)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

        if opt.amp:
            from apex import amp
            from apex.parallel import DistributedDataParallel as DDP
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP
        # It is recommended to use DistributedDataParallel, instead of DataParallel
        # to do multi-GPU training, even if there is only a single node.
        model = model.to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank)


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    checkpoint_path = os.path.join(opt.save_folder, opt.save_path)

    writer = SummaryWriter(opt.log_path)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        first_epoch = checkpoint["epoch"] + 1
        model.module.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        first_epoch = 0

    for epoch in range(first_epoch, opt.epochs):
        train(model, train_loader, epoch, writer, criterion, optimizer, scheduler, opt.amp)
        if opt.dataset == 'Cognata':
            cognata_eval(model, test_loader, epoch, writer, encoder, opt.nms_threshold)
        else:
            evaluate(model, test_loader, epoch, writer, encoder, opt.nms_threshold)
        checkpoint = {"epoch": epoch,
                      "model_state_dict": model.module.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict()}
        if rank == 0:
            torch.save(checkpoint, checkpoint_path)
        torch.distributed.barrier()
    cleanup()


if __name__ == "__main__":
    opt = get_args()
    world_size = opt.num_gpus
    torch.multiprocessing.spawn(
        main,
        args=(opt, world_size),
        nprocs=world_size
    )
