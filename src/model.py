from argparse import Namespace
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from grakel.kernels import (WeisfeilerLehman, VertexHistogram, 
                            WeisfeilerLehmanOptimalAssignment, 
                            Propagation, GraphletSampling, 
                            RandomWalkLabeled, PyramidMatch)
from layers import *
from cam import CAMKernel
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.cluster.vq import kmeans2
import time



def H(x):
    # clamp avoids 0*log(0) = NaN; same mathematical definition
    return -(x * x.clamp_min(1e-12).log()).sum(-1)

def JSD(x):
    return H(x.mean(0)) - H(x).mean(0)

EPS = 1e-8

def jsd_code(R, batch=None):
    """JSD as in the OFFICIAL GKNN code (lcosmo/GKNN and gdl-unive/Graph-Kernel-Convolution,
    src/model.py: `-jsd_weight * JSD(x.softmax(-1))`). Used for both GKNN and CAM.

    q_v = softmax_i(R_vi) is a distribution over the m FILTERS for each node v;
    JSD = H(mean_v q_v) - mean_v H(q_v), over all nodes of the mini-batch
    (graphs are not separated). Range [0, log m]. Large when each node is
    claimed by few filters and different nodes by different filters.

    Differs from the paper (Cosmo et al., Eq. 4), which normalises each filter
    over the VERTICES of a graph and uses sums instead of means: see jsd_paper.
    Note: softmax depends on the response scale, which includes the learned
    per-filter `Model.mask` scale, so JSD can also grow by sharpening that scale.
    """
    return JSD(R.softmax(-1))

def jsd_paper(R, batch):
    """JSD as in the paper: p_i(v) = R_vi / sum_{u in V(G)} R_ui, normalised over the
    VERTICES of each graph separately. JSD is taken across the m filter distributions
    of each graph, then averaged over the graphs in the batch."""
    R = R.clamp_min(0) + EPS                        # p_i(v) requires R >= 0
    S = global_add_pool(R, batch)                   # [G, m]  sum_u R_ui per graph
    P = R / S[batch]                                # [N, m]  p_i(v)
    H_i = -global_add_pool(P * P.clamp_min(1e-12).log(), batch)      # [G, m]  H(p_i) per graph
    M = P.mean(-1, keepdim=True)                                     # [N, 1]  mixture (1/m) sum_i p_i(v)
    H_M = -global_add_pool(M * M.clamp_min(1e-12).log(), batch)[:, 0]  # [G]
    return (H_M - H_i.mean(-1)).mean()

LOGDET_EPS = 1e-4
CONST_STD = 1e-6


def correlation_matrix(R, const_std=CONST_STD):
    """Empirical Pearson correlation; inactive columns have zero rows/cols."""
    Rc = R - R.mean(0, keepdim=True)
    norm = Rc.norm(dim=0, keepdim=True)
    active = Rc.detach().std(dim=0, unbiased=False) > const_std
    U = torch.where(active[None, :], Rc / norm.clamp_min(EPS), torch.zeros_like(Rc))
    return U.t() @ U, active


def logdet_penalty(R, batch=None, eps=LOGDET_EPS, const_std=CONST_STD):
    """Collective diversity loss: -logdet(C + eps I) / m.

    Note: inactive filters are measured, not automatically prevented.
    """
    m = R.shape[-1]
    if m < 2:
        return R.sum() * 0
    C, _ = correlation_matrix(R, const_std)
    C = C.double()
    A = 0.5*(C+C.t()) + eps*torch.eye(m, dtype=C.dtype, device=C.device)
    L, info = torch.linalg.cholesky_ex(A)
    if int(info) == 0:
        logdet = 2*torch.log(torch.diagonal(L)).sum()
    else:
        sign, logdet = torch.linalg.slogdet(A)
        if sign <= 0:
            raise RuntimeError('Non-positive determinant in logdet_penalty')
    return (-logdet/m).to(R.dtype)




def _cam_or_gkernel(hparams, layer_index, nodes, labels, filters, **kwargs):
    """Preserve original GKNN; choose connected CAM on request."""
    mode = getattr(hparams, "mask_distribution", "original")
    if mode == "original":
        return GKernel(nodes, labels, filters, **kwargs)
    if mode != "cam":
        raise ValueError("Invalid mask_distribution: " + str(mode))
    # Layers >= 2 (cam_multilayer_inputs):
    #  'gknn'        : argmax input labels + the original GKNN input-gradient
    #                  estimator -> same forward/inter-layer path as GKNN; only
    #                  the mask distribution differs. Used by the current pipeline.
    #  'categorical' : (legacy default, kept so older runs/checkpoints reproduce)
    #                  labels sampled from softmax(x/T) with a score-function
    #                  gradient; NOT equivalent to GKNN's argmax labels.
    layer_index = int(layer_index)
    kernel_names = str(getattr(hparams, "kernel", "wl")).split("+")
    if layer_index > 0 and len(kernel_names) != 1:
        raise ValueError("Multilayer CAM currently requires exactly one graph kernel")
    multilayer = str(getattr(hparams, "cam_multilayer_inputs", "categorical"))
    if multilayer not in ("gknn", "categorical"):
        raise ValueError("Invalid cam_multilayer_inputs: " + multilayer)
    deep_gknn = layer_index > 0 and multilayer == "gknn"
    return CAMKernel(
        nodes, labels, filters,
        mc_samples=int(getattr(hparams, "cam_samples", 8)),
        eval_samples=int(getattr(hparams, "cam_eval_samples", 32)),
        initial_stop_prob=float(getattr(hparams, "cam_initial_stop_prob", 0.1)),
        eval_seed=int(getattr(hparams, "cam_eval_seed", 1729)) + layer_index,
        input_label_mode="argmax" if layer_index == 0 or deep_gknn else "categorical",
        input_temperature=float(getattr(hparams, "cam_input_temperature", 1.0)),
        input_gradient="gknn" if deep_gknn else None,
        **kwargs,
    )


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        print(hparams)
        
        try: 
            in_features, hidden, num_classes, labels = hparams.in_features,  hparams.hidden,  hparams.num_classes, hparams.labels
        except:
            hparams = Namespace(**hparams)
            in_features, hidden, num_classes, labels = hparams.in_features,  hparams.hidden,  hparams.num_classes, hparams.labels

        #self.hparams=hparams
        self.save_hyperparameters(hparams)
       
#         assert(layers==1) #only works with 1 layer at the moment
        self.conv_layers = nn.ModuleList()
        self.vq_layers =nn.ModuleList()
    
        self.conv_layers.append(_cam_or_gkernel(hparams, 0, hparams.nodes, labels, hidden,max_cc=self.hparams.max_cc,hops=hparams.hops, kernels=hparams.kernel, store_fit=True))
        
        n_kernels = len(hparams.kernel.split('+'))
        for i in range(1, hparams.layers):
            self.conv_layers.append(_cam_or_gkernel(hparams, i, hparams.nodes, hidden, hidden,max_cc=self.hparams.max_cc,hops=hparams.hops, kernels=hparams.kernel))
            
            self.vq_layers.append(SoftAss(hidden, hidden))
        
        activation = nn.LeakyReLU
        if hparams.activation == 'sigmoid':
          activation = nn.Sigmoid
          
#         print(hidden*n_kernels*hparams.layers)
#         print(type(hidden))
#         print(type(n_kernels))
#         print(type(hparams.layers))
#         print(type(num_classes))
        if hparams.mlp_layers==2:
            self.fc = nn.Sequential(nn.Linear(hidden*n_kernels*hparams.layers,hidden),activation(),nn.Linear(hidden,hidden),activation(),nn.Linear(hidden,num_classes))
        else:
            self.fc = nn.Sequential(nn.Linear(hidden*n_kernels,hidden),activation(),nn.Linear(hidden,num_classes))
        
        self.eye = torch.eye(hidden)
        self.lin = nn.Linear(hidden,hidden)
        
        self.automatic_optimization = False
        
        reg_type = getattr(hparams, 'reg_type', 'jsd_code')
        reg_weight = hparams.jsd_weight
        logdet_eps = getattr(hparams, 'logdet_eps', LOGDET_EPS)
        def _regularizers(x, batch):
            if reg_type == 'jsd_code':      # maximise diversity -> minus sign
                return -reg_weight * jsd_code(x, batch)
            if reg_type == 'jsd_paper':     # maximise diversity -> minus sign
                return -reg_weight * jsd_paper(x, batch)
            if reg_type == 'logdet':
                return reg_weight * logdet_penalty(x, batch, eps=logdet_eps)
            if reg_type == 'none':
                return x.sum() * 0
            raise ValueError('unknown reg_type %s' % reg_type)
        self.regularizers = _regularizers
            
        self.mask=nn.Parameter(torch.ones(hidden).float())
        
    def prefit(self,data):
        for l in self.conv_layers:
          if l.store_fit:
            with torch.no_grad():
              l.prefit(data.x,data.edge_index)
    
        
    def one_hot_embedding(self, labels):
        self.eye = self.eye.to(labels.device)
        return self.eye[labels] 

    def forward(self, data):
    
        # PyG stores Data properties outside __dict__.
        nidx = getattr(data, 'nidx', None)
            
        batch=data.batch
        edge_index=data.edge_index
        x=data.x
        
        egonets_data = [data.egonets_idx,data.egonets_edg]
        
        loss = x.sum().detach()*0
        
        responses = []
        for l,vq in zip(self.conv_layers,[None]+list(self.vq_layers)): #only works with one layer
#             if vq!=None:
#                 x = vq(x)
               
            x = l(x,edge_index,node_indexes=nidx,egonets_data=egonets_data,batch=batch,optimize=self.hparams.optimize_masks)
                
            if self.mask is not None:
                x = x*self.mask[None,:].repeat(1,x.shape[-1]//self.mask.shape[-1])
            
            responses.append(x)
        x = torch.cat(responses,-1)
#         x = responses[-1]
        
        pooling_op = None
        if self.hparams.pooling=='add': 
            pooling_op=global_add_pool
        if self.hparams.pooling=='max': 
            pooling_op=global_max_pool
        if self.hparams.pooling=='mean': 
            pooling_op=global_mean_pool
        
        return self.fc(pooling_op(x,batch)), responses, loss
    
    def configure_optimizers(self):
#         assert(len(self.conv_layers)==1)
        
        # Deterministic parameter order (the original used sets, whose order varies
        # between processes): same groups and learning rates, but the optimizer
        # state can be restored when a run resumes from a checkpoint.
        graph_params = list(self.conv_layers.parameters())
        graph_ids = {id(p) for p in graph_params}
        cla_params = [p for p in self.parameters() if id(p) not in graph_ids]
        optimizer = torch.optim.Adam([{'params': graph_params,'lr': self.hparams.lr_graph},\
                                      {'params': cla_params,'lr': self.hparams.lr}])
        
        return optimizer
    
    
    def training_step(self, train_batch, batch_idx):  
       
        data=train_batch
        
        optimizer = self.optimizers()
        
        optimizer.zero_grad()
            
        output, responses, vqloss = self(data)
        loss_ce = self.hparams.loss(torch.squeeze(output,-1), data.y)
        
        loss_reg = torch.stack([self.regularizers(x, data.batch) for x in responses]).mean()
#         loss_spa = self.hparams.sparsity * sum([sum([layer.Padd[fi].mean() for fi in range(self.hparams.hidden)]) -\
#                sum([layer.Prem[fi].mean() for fi in range(self.hparams.hidden)]) for layer in self.conv_layers])

        loss = loss_ce + loss_reg #+ vqloss +  loss_spa
        loss.backward()
        optimizer.step()
        
        # for i,p in enumerate(self.conv_layers[0].parameters()):
            # if p.grad is None:
                # n=0
            # else:
                # n = torch.norm(p.grad)
                # self.log('p%d_%s' % (i,p.shape), n, on_step=False, on_epoch=True)
            
            
        acc = 100*torch.mean( (output.argmax(-1)==data.y).float()).detach().cpu()
        self.log('acc', acc, on_step=False, on_epoch=True)
        self.log('loss', loss.item(), on_step=False, on_epoch=True)
        self.log('loss_reg', loss_reg.item(), on_step=False, on_epoch=True)
        with torch.no_grad():
            # Unweighted official-code JSD (>= 0, larger = more diverse filters);
            # logged for every reg_type so diversity is comparable across runs.
            self.log('jsd', torch.stack([jsd_code(x) for x in responses]).mean().item(),
                     on_step=False, on_epoch=True)
            # share of responses < 0 (zeroed by the clamp in jsd_paper, so no gradient there)
            frac_neg = torch.stack([(x < 0).float().mean() for x in responses]).mean()
            # Monitor constant filters separately from the determinant penalty
            frac_const = torch.stack([(~correlation_matrix(x)[1]).float().mean() for x in responses]).mean()
        self.log('frac_neg_resp', frac_neg.item(), on_step=False, on_epoch=True)
        self.log('frac_const_filters', frac_const.item(), on_step=False, on_epoch=True)
        self.log('loss_ce', loss_ce.item(), on_step=False, on_epoch=True)
#         self.log('loss_spa', loss_spa.item(), on_step=False, on_epoch=True)


    # def training_epoch_end(self, training_step_outputs):      
        # self.logger.log_image(key="grads", images=[self.conv_layers[0].P[0].data.cpu(),(self.conv_layers[0].Padd[0]*self.conv_layers[0].temp).sigmoid().data.cpu()])
    
    
    def validation_step(self, train_batch, batch_idx):
        data=train_batch
        with torch.no_grad():
            output, x1, _ = self(data)
#             print(output.shape)
#             print(data.y.shape)
            # val_loss stays cross-entropy only (as in official GKNN): it drives
            # checkpointing/early stopping and is comparable across JSD weights.
            loss = self.hparams.loss(torch.squeeze(output,-1), data.y)
            acc = 100*torch.mean( (output.argmax(-1)==data.y).float()).detach().cpu()
            self.log('val_loss', loss.item(), on_step=False, on_epoch=True)
            self.log('val_acc', acc, on_step=False, on_epoch=True)
            self.log('val_jsd', torch.stack([jsd_code(x) for x in x1]).mean().item(),
                     on_step=False, on_epoch=True)
 
    def test_step(self, train_batch, batch_idx):
        data=train_batch
        with torch.no_grad():
            output, x1, _ = self(data)
            loss = self.hparams.loss(torch.squeeze(output,-1), data.y)
            acc = 100*torch.mean( (output.argmax(-1)==data.y).float()).detach().cpu()
            self.log('test_loss', loss.item(), on_step=False, on_epoch=True)
#             self.log('test_loss', loss.item(), on_step=False, on_epoch=True)
            self.log('test_acc', acc, on_step=False, on_epoch=True)       
            
            
            
class SoftAss(nn.Module):
    def __init__(self, num_words, features_dim, softmax=True):
        super(SoftAss, self).__init__()
        
        if softmax:
            self.normalize = nn.Softmax(-1)
        else: 
            self.normalize = nn.Normalize(dim=-1)
            
        self.dict = nn.Parameter(torch.rand(num_words, features_dim).float(), requires_grad=True)
        self.codebook_init=False        
        
    def reset_codebook(self,x):
        if self.codebook_init:
            centroid, label = kmeans2(x.detach().cpu().numpy(), self.dict.detach().cpu().numpy(), minit='matrix')
        else:
            centroid, label = kmeans2( (x + torch.randn_like(x)*1e-4).detach().cpu().numpy(), self.dict.shape[0])
        self.dict.data = torch.from_numpy(centroid).float().to(x.device)
        self.codebook_init = True
        
        
    def forward(self, x):
        
        if self.training:
            self.reset_codebook(x)
            
#         _dict = self.normalize(self.dict)
        _x = self.normalize(x)

        res = torch.tensordot(_x, self.dict, dims=([1], [1]))

        return res