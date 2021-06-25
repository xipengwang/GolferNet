import os
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


from torch.utils.tensorboard import SummaryWriter

def read_labels(label_root):
    data = []
    class_map = None
    for json_file in glob.glob(os.path.join(label_root, '*.json')):
        with open(json_file, 'r') as f:
            labels = json.load(f)
            for label in labels:
                idx = label['datasetObjectId']
                content = label['consolidatedAnnotation']['content']
                class_map_ = content['Human-Pose-metadata']['class-map']
                if class_map is None or len(class_map_.items()) > len(class_map.items()):
                    class_map = class_map_
                annotations = content['Human-Pose']['annotations']
                centers = []
                for annotation in annotations:
                    class_id = annotation['class_id']
                    center = (annotation['top']+annotation['height']/2.0, annotation['left']+annotation['width']/2.0)
                    centers.append((int(class_id),center))
                data.append({idx:centers})
    return data, class_map


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10240'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()

class GolfPoseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.labels, self.class_map = read_labels(os.path.join(self.root_dir, 'labels'))
        self.nclass = len(self.class_map.items())
        imgs_tmp = glob.glob(os.path.join(self.root_dir, 'images/*.png'))
        self.imgs = []
        for label in self.labels:
            key, val = next(iter(label.items()))
            for img in imgs_tmp:
                if int(key) == int(os.path.basename(img).split('.')[0]):
                    self.imgs.append(img)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx]
        img_name = self.imgs[idx]
        img_bgr = cv.imread(img_name)
        image = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        np.transpose(image, (2, 0, 1))
        #  "width": 960,
        #  "height": 540,
        #  "depth": 3
        #  TODO: crop images
        width = int(960/4)
        height = int(540/4)
        target = np.zeros((self.nclass, height, width))
        for key, val in label.items():
            for (c, center) in val:
                target[c, int(center[0]/4), int(center[1]/4)] = 1

        for c in range(self.nclass):
            target[c, :, :] = cv.GaussianBlur(target[c, :, :], (5, 5), 0)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': target}
        return (sample, img_name)

def train(rank, model, state, args):
    writer = SummaryWriter(comment=f'_RANK_{rank}_LR_{args.lr}_BS_{args.batch_size}')
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset = GolfPoseDataset(args.data,
                              transform=transform)

    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        dataset,  [n_train, n_val], generator=torch.Generator().manual_seed(1))

    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}
    device = state['device']
    if state['use_cuda']:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                      })
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)

    torch.manual_seed(rank)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)

    model.to(device)
    train_sampler = None
    if state['use_cuda'] and state['world_size'] > 1:
        kwargs['shuffle'] = False
        setup(rank, state['world_size'])
        model = DDP(model, device_ids=[rank])
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_subset,
            num_replicas=state['world_size'],
            rank=rank
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_subset,
            num_replicas=state['world_size'],
            rank=rank
        )
        kwargs.update({'sampler': train_sampler})

    train_loader = torch.utils.data.DataLoader(dataset=train_subset, **kwargs)
    kwargs['batch_size'] = n_val
    val_loader = torch.utils.data.DataLoader(dataset=val_subset, **kwargs)

    for batch_idx, (sample, image_name) in enumerate(train_loader):
        print(image_name)

    print('----')
    for batch_idx, (sample, image_name) in enumerate(val_loader):
        print(image_name)

    return


    global_step = 0
    for epoch in range(args.iters):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        global_step = train_epoch(epoch=epoch, model=model, device=device, data_loader=train_loader, optimizer=optimizer, scheduler=scheduler, args=args,
                                  global_step=global_step, writer=writer, rank=rank, val_loader=val_loader)
    if state['use_cuda']:
        cleanup()

def train_epoch(*, epoch, model, device, data_loader, optimizer, scheduler, args, global_step, writer, rank, val_loader):
    model.train()
    pid = os.getpid()
    for batch_idx, (sample, image_name) in enumerate(data_loader):
        data = sample['image']
        target = sample['label']
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = model.calc_loss(combined_hm_preds=output, heatmaps=target.to(device))
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(f'{pid}\tTrain Epoch: {epoch} \
            [{batch_idx * len(data)}/{len(data_loader.dataset)} \
            ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            writer.add_scalar('Loss/train', loss.item(), global_step)

        im = data[0, :, :, :].permute(1, 2, 0).numpy()
        cols, rows, channels = im.shape
        im = cv.pyrDown(im, dstsize=(cols // 2, rows // 2))
        cols, rows, channels = im.shape
        im = cv.pyrDown(im, dstsize=(cols // 2, rows // 2))
        plt.imsave('./plots/train-im.png', im)
        heatmap = output[0, 0, :, :].permute(1, 2, 0).detach().numpy()
        min_v = np.min(heatmap)
        max_v = np.max(heatmap)
        heatmap = (heatmap - min_v) / (max_v - min_v) * 255
        # heatmap *= 255
        cv.imwrite('./plots/train-heatmap.png', heatmap)

        if True or global_step % 10 == 0:
            val_epoch(epoch=epoch, model=model, device=device, data_loader=val_loader, optimizer=optimizer, args=args,
                      global_step=global_step, writer=writer, rank=rank)
            for tag, value in model.named_parameters():
                pass
                # print(tag)
                # tag = tag.replace('.', '/')
                # writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            if rank == 0:
                pass
                # torch.save(model.state_dict(), args.model+f'-epoch-{epoch}-step-{global_step}')
            scheduler.step()
        global_step += 1
    return global_step

def val_epoch(*, epoch, model, device, data_loader, optimizer, args, global_step, writer, rank):
    model.eval()
    pid = os.getpid()
    for batch_idx, (sample, image_name) in enumerate(data_loader):
        data = sample['image']
        target = sample['label']
        output = model(data.to(device))
        loss = model.calc_loss(combined_hm_preds=output, heatmaps=target.to(device))

        im = data[0, :, :, :].permute(1, 2, 0).numpy()
        cols, rows, channels = im.shape
        im = cv.pyrDown(im, dstsize=(cols // 2, rows // 2))
        cols, rows, channels = im.shape
        im = cv.pyrDown(im, dstsize=(cols // 2, rows // 2))
        plt.imsave('./plots/val-im.png', im)
        heatmap = output[0, 0, :, :].permute(1, 2, 0).detach().numpy()
        min_v = np.min(heatmap)
        max_v = np.max(heatmap)
        heatmap = (heatmap - min_v) / (max_v - min_v) * 255
        # heatmap *= 255
        cv.imwrite('./plots/val-heatmap.png', heatmap)

        if batch_idx % args.log_interval == 0:
            print(f'{pid}\tVal Epoch: {epoch} \
            [{batch_idx * len(data)}/{len(data_loader.dataset)} \
            ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            writer.add_scalar('Loss/val', loss.item(), global_step)
    model.train()


def infer(model, state, args):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=False,
                       transform=transform)
    kwargs = {'batch_size': args.batch_size,
              'shuffle': False}
    if state['use_cuda']:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                      })
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               **kwargs)
    test_epoch(epoch=epoch, model=model, state=state, data_loader=train_loader, args=args)

def test_epoch(*, epoch, model, state, data_loader, args):
    device = state['device']
    model.to(device)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += model.calc_loss(output, target)

    test_loss /= len(data_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}')
