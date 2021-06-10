#!/usr/bin/env python3

import argparse
import os
import sys

import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp

from model import Model
from train import train

def parse(args):
    parser = argparse.ArgumentParser(description='GolferNet aimed to detect golfer poses.')
    subparsers = parser.add_subparsers(help='sub-command', dest='command')
    subparsers.required = True

    devcount = max(1, torch.cuda.device_count())

    parser_train = subparsers.add_parser('train', help='train a network')
    parser_train.add_argument('model', type=str, help='path to output model or checkpoint to resume from')
    parser_train.add_argument('--num-processes', type=int, default=1, metavar='N',help='how many training processes to use (default: 1)')
    parser_train.add_argument('--lr', metavar='value', help='learning rate', type=float, default=0.01)
    parser_train.add_argument('--momentum', type=float, default=0.5, metavar='value', help='SGD momentum (default: 0.5)')
    parser_train.add_argument('--batch-size', metavar='size', type=int, help='batch size', default=2*devcount)
    parser_train.add_argument('--iters', metavar='number', type=int, help='number of iterations to train for', default=100)
    parser_train.add_argument('--val-percent', metavar='number', type=float, help='percentage of data for validation', default=0.2)
    parser_train.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')

    parser_infer = subparsers.add_parser('infer', help='run inference')
    parser_infer.add_argument('model', type=str, help='path to model')
    parser_infer.add_argument('--batch-size', metavar='size', type=int, help='batch size', default=2*devcount)

    return parser.parse_args(args)

def load_model(args, verbose=False):
    if args.command != 'train' and not os.path.isfile(args.model):
        raise RuntimeError(f'Model file {args.model} does not exist!')

    model = None
    state = {}
    _, ext = os.path.splitext(args.model)
    if ext != '.torch':
        raise RuntimeError(f'Model file {args.model} should end with .torch')

    if args.command == 'train' and (not os.path.exists(args.model)):
        if verbose:
            print('Initializing model...')
        model = Model()
    else:
        if verbose:
            print(f'Loading model from {os.path.basename(args.model)}...')
        model, state = Model.load(args.model)
    if verbose:
        print(model)
    state['model_path'] = args.model
    return model, state

def main(args=None):
    args = parse(args or sys.argv[1:])

    model, state = load_model(args, verbose=True)
    state['use_cuda'] = torch.cuda.is_available()
    state['device'] =  torch.device("cuda" if state['use_cuda'] else "cpu")
    if model:
        model.share_memory()
    world_size = args.num_processes
    state['world_size'] = world_size
    ngpu = torch.cuda.device_count()
    state['ngpu'] = ngpu
    if ngpu > 0 and world_size > ngpu:
        raise RuntimeError(f'Set number of process smaller than number of GPUs')
    print(f'Number of process: {world_size}')
    if args.command == 'infer':
        raise RuntimeError('Not implemented')
    else:
        mp.spawn(train,
                 args=(model, state, args),
                 nprocs=world_size,
                 join=True)

if __name__ == '__main__':
    main()
