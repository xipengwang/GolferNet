import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

def train(rank, model, state, args):
    writer = SummaryWriter(comment=f'_RANK_{rank}_LR_{args.lr}_BS_{args.batch_size}')
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}
    if state['use_cuda']:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                      })
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               **kwargs)
    torch.manual_seed(rank)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if state['world'] == 1:
        model = nn.DataParallel(model)
    global_step = 0
    for epoch in range(args.iters):
        global_step = train_epoch(epoch=epoch, model=model, state=state, data_loader=train_loader, optimizer=optimizer, args=args,
                    global_step=global_step, writer=writer)

def train_epoch(*, epoch, model, state, data_loader, optimizer, args, global_step, writer):
    device = state['device']
    model.to(device)
    model.train()
    pid = os.getpid()
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # torch.save(model.state_dict(), args.model+f'-epoch-{epoch}-batch-{batch_idx}')
            print(f'{pid}\tTrain Epoch: {epoch} \
            [{batch_idx * len(data)}/{len(data_loader.dataset)} \
            ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            writer.add_scalar('Loss/train', loss.item(), global_step)
        global_step += 1
    return global_step

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
