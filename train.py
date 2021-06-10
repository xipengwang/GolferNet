import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10240'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, model, state, args):
    writer = SummaryWriter(comment=f'_RANK_{rank}_LR_{args.lr}_BS_{args.batch_size}')
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST('./data', train=True, download=True,
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
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)

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
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'{pid}\tTrain Epoch: {epoch} \
            [{batch_idx * len(data)}/{len(data_loader.dataset)} \
            ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            writer.add_scalar('Loss/train', loss.item(), global_step)

        if global_step % (10 * args.batch_size) == 0:
            val_epoch(epoch=epoch, model=model, device=device, data_loader=val_loader, optimizer=optimizer, args=args,
                      global_step=global_step, writer=writer, rank=rank)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            if rank == 0:
                torch.save(model.state_dict(), args.model+f'-epoch-{epoch}-step-{global_step}')
        global_step += 1
    return global_step

def val_epoch(*, epoch, model, device, data_loader, optimizer, args, global_step, writer, rank):
    model.eval()
    pid = os.getpid()
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_idx, (data, target) in enumerate(data_loader):
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
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
