import copy
import os

import json
import itertools
import random
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import multiprocessing as mp

import sys
sys.path.append('./src/')

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import argparse
from model import *
from dataset import TUDataset, EgonetLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
import pytorch_lightning as pl
from multiprocessing import Process, Queue
import subprocess
import torch_geometric.datasets

import json   
import sys  
from argparse import Namespace
  

def get_arg_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--num_gpus", default=4, type=int)

    parser.add_argument("--project", default='GKNN')
    parser.add_argument("--dataset", default='MUTAG')
    parser.add_argument("--fold", default=0, type=int) #selected fold
    parser.add_argument("--folds", default=10, type=int) #total number of folds
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--logger", default='tensorboard', choices=['tensorboard', 'wandb'])
    parser.add_argument("--wandb_entity", default=None) # your W&B entity, if --logger wandb
    parser.add_argument("--run_test", default=False, type=lambda x: (str(x).lower() == 'true')) # evaluate on test (only for final configs)

    parser.add_argument("--nodes", default=6, type=int) 
    parser.add_argument("--labels", default=7, type=int) 
    parser.add_argument("--hidden", default=16, type=int) 
    
    parser.add_argument("--mlp_layers", default=2, type=int) 
    parser.add_argument("--activation", default='relu') 
    parser.add_argument("--layers", default=2, type=int) 
    parser.add_argument("--hops", default=1, type=int) 
    parser.add_argument("--kernel", default='wl', type=str) 
    parser.add_argument("--normalize", default=True, type=bool) 
    
    
    parser.add_argument("--pooling", default='add', type=str) 
    
    
    parser.add_argument("--sparsity", default=0, type=float) 
    parser.add_argument("--jsd_weight", default=1e3, type=float) # weight of the regulariser chosen below
    parser.add_argument("--reg_type", default='jsd_code', choices=['jsd_code', 'jsd_paper', 'logdet', 'none'])
    parser.add_argument("--logdet_eps", default=1e-4, type=float) # epsilon in -(1/m) log det(C + eps I), reg_type logdet
    parser.add_argument("--max_cc", default=True, type=bool)
    parser.add_argument("--mask", default=False, type=bool)

    parser.add_argument("--max_epochs", default=1000, type=int) 
    parser.add_argument("--lr", default=5e-3, type=float) 
    parser.add_argument("--lr_graph", default=1e-2, type=float) 
    
    
    parser.add_argument("--optimize_masks", default=True)
    
    parser.add_argument("--batch_size", default=16, type=int) 
#     parser.add_argument("--num_gpus", default=20, type=int) #num filters
    parser.add_argument("--run", default=0) #run id
    parser.add_argument("--debug", default=False, type=lambda x: (str(x).lower() == 'true')) #debug mode
    return parser 

def run_training_process_with_validation(run_params):
    print("######################################### NEW TRAIN on FOLD %d ######################################" % run_params.fold)
    pl.seed_everything(run_params.seed)

    if run_params.dataset == 'ENZYMES':
        dataset = torch_geometric.datasets.TUDataset('./data',run_params.dataset,hops=run_params.hops)
    else:
        dataset = TUDataset('./data',run_params.dataset,hops=run_params.hops)
        
    # dataset = TUDataset('./data',run_params.dataset)
    yy = [int(d.y) for d in dataset]
    fold = run_params.fold

    
    ###### Load or generate splits
    os.makedirs('./data/folds', exist_ok=True)
    if not os.path.isfile('./data/folds/%s_folds_%d.txt' % (run_params.dataset, run_params.folds)):
        print('GENERATING %d FOLDS FOR %s' % (run_params.folds, run_params.dataset) ) 
        skf = StratifiedKFold(n_splits=run_params.folds, random_state=1, shuffle=True)
        folds = list(skf.split(np.arange(len(yy)),yy))

        folds_split = []
        for fold in range(run_params.folds):
          train_i_split, val_i_split = train_test_split([int(i) for i in folds[fold][0]],
                                                stratify=[n for n in np.asarray(yy)[folds[fold][0]]],
                                                test_size=int(len(list(folds[fold][0]))*0.1),
                                                random_state=0)
          test_i_split = [int(i) for i in folds[fold][1]]
          folds_split.append([train_i_split,val_i_split,test_i_split])

        with open('./data/folds/%s_folds_%d.txt' % (run_params.dataset, run_params.folds), 'w') as f:
            f.write(json.dumps(folds_split))

    fold = run_params.fold
    with open('./data/folds/%s_folds_%d.txt' % (run_params.dataset, run_params.folds), 'r') as f:
        folds = json.loads(f.read())
    train_i_split,val_i_split,test_i_split = folds[fold]
    
    train_dataset = dataset[train_i_split]
    val_dataset = dataset[val_i_split]
    
    test_dataset = dataset[test_i_split]

    train_loader = EgonetLoader(train_dataset[:], batch_size=run_params.batch_size, shuffle=True)
    val_loader = EgonetLoader(val_dataset, batch_size=2000, shuffle=False)
    test_loader = EgonetLoader(test_dataset, batch_size=2000, shuffle=False)
 
    run_params.loss = torch.nn.CrossEntropyLoss()

    class MyDataModule(pl.LightningDataModule):
        def setup(self,stage=None):
            pass
        def train_dataloader(self):
            return train_loader
        def val_dataloader(self):
            return val_loader
        def test_dataloader(self):
            return test_loader
    
    run_params.in_features = train_dataset.num_features
    run_params.labels = train_dataset.num_features
    run_params.num_classes = train_dataset.num_classes
    
    model = Model(run_params)

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=300,
        verbose=False,
        mode='min')

    if run_params.mask:
        run_params.max_epochs = run_params.max_epochs//2
        
    run_name = '%s_%s_w%g_fold%d_seed%d' % (run_params.dataset, run_params.reg_type,
                                            run_params.jsd_weight, run_params.fold, run_params.seed)
    if run_params.logger == 'wandb':
        import wandb  # only needed when W&B is used
        logger = WandbLogger(project=run_params.project, log_model=True, name=run_name,
                             entity=run_params.wandb_entity)
    else:
        logger = TensorBoardLogger('./lightning_logs', name=run_name)

    # Trainer.from_argparse_args was removed in Lightning 2.0
    trainer = pl.Trainer(max_epochs=run_params.max_epochs,
                         accelerator="cpu",  # kernels use NumPy conversions
                         devices=1,
                         logger=logger,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         log_every_n_steps=1)

    trainer.fit(model, datamodule=MyDataModule())

    print("TRAINING FINISHED")
    print("################# VALIDATING (best checkpoint) #####################")
    trainer.validate(model=model, datamodule=MyDataModule(), ckpt_path="best", weights_only=False)  # own checkpoint; hparams contain the loss module
    if run_params.run_test:
        # only for the configurations selected on validation
        print("################# TESTING (best checkpoint) #####################")
        trainer.test(model=model, datamodule=MyDataModule(), ckpt_path="best", weights_only=False)  # own checkpoint; hparams contain the loss module
    if run_params.logger == 'wandb':
        wandb.finish()

if __name__ == "__main__":
    params = get_arg_parser().parse_args()
    run_training_process_with_validation(params)


      