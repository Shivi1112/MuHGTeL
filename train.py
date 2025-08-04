import os
from tqdm import *
import random
import argparse
import numpy as np
from joblib import dump, load
import torch
import torch.optim as optim
from utils import *
from data.Task import *
from models.Model import *
from models.baselines import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch_geometric.data import Data
# from explainability import *
# from att_explain import *
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1, help = 'Number of epochs to train.')
parser.add_argument('--lr', type=float, default = 0.001, help = 'learning rate.')
parser.add_argument('--model', type=str, default="MuHGTeL", help = 'Transformer, RETAIN, StageNet, KAME, GCT, DDHGNN, TRANS, MuHGTeL')
parser.add_argument('--dev', type=int, default = 7)
parser.add_argument('--seed', type=int, default = 42)
parser.add_argument('--dataset', type=str, default = "mimic3", choices=['mimic3', 'mimic4'])
parser.add_argument('--batch_size', type=int, default = 128)
parser.add_argument('--pe_dim', type=int, default = 4, help = 'dimensions of spatial encoding')
parser.add_argument('--devm', type=bool, default = False, help = 'develop mode')

fileroot = {
   'mimic3': 'path/mimic3',
   'mimic4': 'path/mimic4',
}

args = parser.parse_args()
print('batch_size', args.batch_size)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print('{}--{}'.format(args.dataset, args.model))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
if args.dataset == 'mimic4':
   task_dataset = load_dataset(args.dataset, root = fileroot[args.dataset], task_fn=diag_prediction_mimic4_fn, dev= args.devm)
elif args.dataset == 'mimic3':
   task_dataset = load_dataset(args.dataset, root = fileroot[args.dataset], task_fn=diag_prediction_mimic3_fn, dev= args.devm)
else:
    task_dataset = load_dataset(args.dataset, root = fileroot[args.dataset])
    
Tokenizers = get_init_tokenizers(task_dataset)
label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('conditions'))
c_tokenzier = Tokenizers['cond_hist']


if args.model == 'Transformer':
    train_loader , val_loader, test_loader = seq_dataloader(task_dataset, batch_size = args.batch_size)
    model  = Transformer(Tokenizers,len(task_dataset.get_all_tokens('conditions')),device)

elif args.model == 'RETAIN':
    train_loader , val_loader, test_loader = seq_dataloader(task_dataset, batch_size = args.batch_size)
    model  = RETAIN(Tokenizers,len(task_dataset.get_all_tokens('conditions')),device)

elif args.model == 'KAME':
    train_loader , val_loader, test_loader = seq_dataloader(task_dataset, batch_size = args.batch_size)
    Tokenizers.update(get_parent_tokenizers(task_dataset))
    model  = KAME(Tokenizers, len(task_dataset.get_all_tokens('conditions')), device)

elif args.model == 'StageNet':
    train_loader , val_loader, test_loader = seq_dataloader(task_dataset, batch_size = args.batch_size)
    model  = StageNet(Tokenizers, len(task_dataset.get_all_tokens('conditions')), device)

elif args.model == 'MuHGTeL':
    data_path = '../TRANS/logs/{}_{}.pkl'.format(args.dataset, args.pe_dim)
    if os.path.exists(data_path):
        print('Loading created data')
        mdataset = load(data_path)
    else:
        print('creating dataset')
        mdataset = MMDataset(task_dataset,Tokenizers, dim = 128, device = device, trans_dim=args.pe_dim)
        dump(mdataset,data_path)
    trainset, validset, testset = split_dataset(mdataset)
    train_loader , val_loader, test_loader = mm_dataloader(trainset, validset, testset, batch_size=args.batch_size)
    
    model = MuHGTeL(Tokenizers, 128, len(task_dataset.get_all_tokens('conditions')),
                    device,graph_meta=graph_meta, pe=args.pe_dim)
    

co_graph = build_co_graph(task_dataset, Tokenizers['cond_hist'])
model.co_graph = co_graph.to(device)
torch.save(co_graph, "co_graph.pt") #1 for cat_att
model.co_graph = torch.load("co_graph.pt", weights_only=False).to(device)


ckptpath = './logs/trained_{}_{}.ckpt'.format(args.model, args.dataset)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=args.lr, 
    weight_decay=1e-4  # L2 regularization
)

# Remove verbose=True for older PyTorch versions
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

early_stopping = EarlyStopping(patience=21, min_delta=0.001)
best = float('inf')

pbar = tqdm(range(args.epochs))
for epoch in pbar:
    model = model.to(device)

    train_loss = train(train_loader, model, label_tokenizer, optimizer, device)
    val_loss = valid(val_loader, model, label_tokenizer, device)

    # Update learning rate based on validation loss
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    
    # Manual verbose logging since verbose=True is not available
    if old_lr != new_lr:
        print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
    
    pbar.set_description(f"Epoch {epoch + 1}/{args.epochs} - train loss: {train_loss:.4f} - valid loss: {val_loss:.4f} - lr: {new_lr:.6f}")
    
    # Save best model
    if val_loss < best:
        best = val_loss
        torch.save(model.state_dict(), ckptpath)
        print(f"New best model saved at epoch {epoch + 1} with val_loss: {val_loss:.4f}")
    
    # Early stopping check
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        print(f"Best validation loss: {best:.4f}")
        break


#for limited gpu memory
if args.model == 'MuHGTeL':
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    device = torch.device('cpu')
    model = MuHGTeL(Tokenizers, 128, len(task_dataset.get_all_tokens('conditions')),
                device, graph_meta=graph_meta, pe=args.pe_dim)
best_model = torch.load(ckptpath)
model.load_state_dict(best_model)
model = model.to(device)
model.co_graph = torch.load("co_graph.pt", weights_only=False).to(device)

y_t_all, y_p_all = [], []
y_true, y_prob = test(test_loader, model, label_tokenizer)
print(code_level(y_true, y_prob))
print(visit_level(y_true, y_prob))






