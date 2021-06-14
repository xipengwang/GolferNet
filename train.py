import os
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
        self.imgs = glob.glob(os.path.join(self.root_dir, '*.png'))
        self.imgs.sort()
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.imgs[idx]
        img_bgr = cv.imread(img_name)
        image = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        np.transpose(image, (2, 0, 1))
        keypoint = None
        if '0' in img_name:
            keypoint = torch.tensor([110-14, 317-14])
        elif '1' in img_name:
            keypoint = torch.tensor([105-14, 306-14])
        elif '2' in img_name:
            keypoint = torch.tensor([102-14, 296-14])
        elif '3' in img_name:
            keypoint = torch.tensor([100-14, 286-14])
        target = np.zeros((1, 128, 128))
        target[0, int(keypoint[1]/4), int(keypoint[0]/4)] = 1
        target[0, :, :] = cv.GaussianBlur(target[0, :, :], (5, 5), 0)
        # plt.imshow(np.transpose(target,(1,2,0)), cmap='gray')
        # plt.show()

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
    n_val = 2
    n_train = 2
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

    global_step = 0
    for epoch in range(args.iters):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        global_step = train_epoch(epoch=epoch, model=model, device=device, data_loader=train_loader, optimizer=optimizer, args=args,
                                  global_step=global_step, writer=writer, rank=rank, val_loader=val_loader)
    if state['use_cuda']:
        cleanup()

def train_epoch(*, epoch, model, device, data_loader, optimizer, args, global_step, writer, rank, val_loader):
    model.train()
    pid = os.getpid()
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_idx, (sample, image_name) in enumerate(data_loader):
        data = sample['image']
        target = sample['label']
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = model.calc_loss(combined_hm_preds=output, heatmaps=target.to(device))
        loss = loss.mean(dim=1).mean(dim=0)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'{pid}\tTrain Epoch: {epoch} \
            [{batch_idx * len(data)}/{len(data_loader.dataset)} \
            ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            writer.add_scalar('Loss/train', loss.item(), global_step)

        if True or global_step % (10 * args.batch_size) == 0:
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
        global_step += 1
    return global_step

def val_epoch(*, epoch, model, device, data_loader, optimizer, args, global_step, writer, rank):
    model.eval()
    pid = os.getpid()
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_idx, (sample, image_name) in enumerate(data_loader):
        data = sample['image']
        target = sample['label']
        output = model(data.to(device))
        plt.figure()
        f, ax = plt.subplots(2, 1)
        ax[0].imshow(data[0, :, :, :].permute(1, 2, 0))
        ax[1].imshow(output[0, 0, :, :].permute(1, 2, 0).detach().numpy() , cmap='gray')
        # plt.imshow(np.transpose(target,(1,2,0)), cmap='gray')
        plt.show()
        loss = model.calc_loss(combined_hm_preds=output, heatmaps=target.to(device))
        loss = loss.mean(dim=1).mean(dim=0)
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
            test_loss += criterion(output, target)

    test_loss /= len(data_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}')
