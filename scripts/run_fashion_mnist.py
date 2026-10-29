"""New LeNet pilot, independent of the missing historical training code.

No dataset transfer without --download. Official test is read only if
--evaluate-test is set, which should happen after configuration selection.
"""
import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys
import time
import urllib.request
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from optimizer_research import Config
from optimizer_research.torch_optimizer import AdaSymTorch


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1,6,5,padding=2), nn.Tanh(), nn.AvgPool2d(2), nn.Conv2d(6,16,5), nn.Tanh(), nn.AvgPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(16*5*5,120),nn.Tanh(),nn.Linear(120,84),nn.Tanh(),nn.Linear(84,10))

    def forward(self, x):
        return self.classifier(self.features(x))


def load_idx(path):
    with gzip.open(path,'rb') as f:
        magic, count = struct.unpack('>II', f.read(8))
        if magic == 2051:
            rows,cols = struct.unpack('>II',f.read(8))
            a = np.frombuffer(f.read(),dtype=np.uint8).copy().reshape(count,rows,cols)
        elif magic == 2049:
            a = np.frombuffer(f.read(),dtype=np.uint8).copy().reshape(count)
        else:
            raise ValueError('unexpected IDX format')
    return a


def official_data(root, download, include_test):
    names = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz']
    if include_test:
        names += ['t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
    hashes = {}
    for name in names:
        path = root/name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'{path} absent. Pass --download to fetch official Fashion-MNIST.')
            root.mkdir(parents=True,exist_ok=True)
            urllib.request.urlretrieve('https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'+name,path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    values = [load_idx(root/name) for name in names]
    return values, hashes


def tensor_data(images, labels):
    return TensorDataset(torch.tensor(images,dtype=torch.float32).unsqueeze(1)/255.,torch.tensor(labels,dtype=torch.long))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0,0,0.
    for x,y in loader:
        x,y = x.to(device),y.to(device)
        logits = model(x)
        total += len(y)
        correct += int((logits.argmax(1)==y).sum())
        loss_sum += float(nn.functional.cross_entropy(logits,y,reduction='sum'))
    return dict(loss=loss_sum/total,accuracy=correct/total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--evaluate-test',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='functional check with artificial pixels, no download')
    parser.add_argument('--data',type=Path,default=Path('data/downloads/fashion-mnist'))
    parser.add_argument('--output',type=Path,default=Path('results/runs/fashion-mnist'))
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--split-seed',type=int,default=42)
    parser.add_argument('--train-size',type=int,default=10000)
    parser.add_argument('--val-size',type=int,default=5000)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--method',choices=['adamw','adamw_scaled','linear','sqrt','historical','sigmoid'],default='linear')
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    if min(args.epochs,args.train_size,args.val_size,args.batch_size) < 1:
        parser.error('sizes and epochs must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    hashes = {}
    if args.smoke:
        generator = torch.Generator().manual_seed(args.split_seed)
        train_data = TensorDataset(torch.rand(32,1,28,28,generator=generator),torch.randint(10,(32,),generator=generator))
        val_data = TensorDataset(torch.rand(16,1,28,28,generator=generator),torch.randint(10,(16,),generator=generator))
        test_data = None
        args.epochs = 1
    else:
        values, hashes = official_data(args.data,args.download,args.evaluate_test)
        if args.train_size+args.val_size > len(values[0]):
            parser.error('train-size + val-size exceeds the training dataset')
        ids = np.random.default_rng(args.split_seed).permutation(len(values[0]))
        ti, vi = ids[:args.train_size], ids[args.train_size:args.train_size+args.val_size]
        hashes['split_indices'] = hashlib.sha256(np.concatenate([ti,vi]).astype('<i8').tobytes()).hexdigest()
        train_data = tensor_data(values[0][ti],values[1][ti])
        val_data = tensor_data(values[0][vi],values[1][vi])
        test_data = tensor_data(values[2],values[3]) if args.evaluate_test else None
    generator = torch.Generator().manual_seed(args.seed+100000)
    train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,generator=generator)
    val_loader = DataLoader(val_data,batch_size=args.batch_size)
    model = LeNet().to(args.device)
    config = Config(lr=args.lr)
    if args.method in ('adamw','adamw_scaled'):
        config = replace(config,alpha=0,lr=args.lr*(np.sqrt(config.cap) if args.method=='adamw_scaled' else 1))
        optimizer = torch.optim.AdamW(model.parameters(),lr=config.lr,betas=(config.beta1,config.beta2),eps=config.eps,weight_decay=config.weight_decay)
    else:
        config = replace(config,rule='linear' if args.method=='sigmoid' else args.method,schedule='sigmoid' if args.method=='sigmoid' else 'constant')
        optimizer = AdaSymTorch(model.parameters(),config)
    start = time.perf_counter()
    curve, best_loss, best = [],float('inf'),None
    total_steps,step = args.epochs*len(train_loader),0
    for epoch in range(args.epochs):
        model.train()
        for x,y in train_loader:
            x,y = x.to(args.device),y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x),y)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            loss.backward()
            if isinstance(optimizer,AdaSymTorch):
                optimizer.set_progress(step/max(total_steps-1,1))
            optimizer.step()
            step += 1
        metrics = evaluate(model,val_loader,args.device)
        curve.append(dict(epoch=epoch+1,validation=metrics,diagnostics=getattr(optimizer,'diagnostics',{})))
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if str(args.device).startswith('cuda'):
        torch.cuda.synchronize()
    record = dict(method=args.method,smoke=args.smoke,seed=args.seed,split_seed=args.split_seed,config=config.dictionary(),dataset_hashes=hashes,train_size=len(train_data),val_size=len(val_data),batch_size=args.batch_size,epochs=args.epochs,curve=curve,elapsed_seconds=time.perf_counter()-start,versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__),device=args.device)
    if test_data is not None:
        model.load_state_dict(best)
        record['test_at_best_validation'] = evaluate(model,DataLoader(test_data,batch_size=args.batch_size),args.device)
    args.output.mkdir(parents=True,exist_ok=True)
    path = args.output/f'{args.method}_seed{args.seed}.json'
    path.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'output':str(path),'last_validation':curve[-1]['validation'],'smoke':args.smoke}))


if __name__=='__main__':
    main()
