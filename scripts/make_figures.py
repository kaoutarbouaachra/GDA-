"""Produce interpretable figures from saved runs; no model retraining.

Examples:
  python scripts/make_figures.py --root results/cam_gknn --out figures
  python scripts/make_figures.py --root results/cam_gknn --out figures --dataset MUTAG
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def collect(root, dataset=None):
    items=[]
    for path in Path(root).rglob('summary.json'):
        try:
            data=json.loads(path.read_text(encoding='utf-8'))
            if dataset and data['dataset']!=dataset:
                continue
            items.append((path.parent,data))
        except (KeyError,ValueError):
            pass
    return items


def savefig(fig,out,name):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def overview_rows(items):
    rows=[]
    for path,run in items:
        row={k:run.get(k) for k in ('dataset','model','reg_type','jsd_weight','fold','seed','best_epoch',
                                    'best_val_loss','total_train_seconds')}
        row['directory']=str(path)
        for split in ['val','test']:
            metrics=run.get('outputs',{}).get(split,{})
            for key in ['accuracy','balanced_accuracy','cross_entropy','roc_auc','average_precision',
                        'size_only_auc_on_this_split']:
                row[f'{split}_{key}']=metrics.get(key)
            for name in ['layer1_raw','layer1_scaled','layer1_raw_add','layer1_raw_mean',
                         'layer1_scaled_add','layer1_scaled_mean']:
                stats=metrics.get('response_stats',{}).get(name,{})
                for metric in ['mean_abs_corr','effective_rank','participation_rank',
                               'active_filters','constant_filters','top_eigen_fraction']:
                    row[f'{split}_{name}_{metric}']=stats.get(metric)
        rows.append(row)
    return rows


def save_csv(rows,out):
    if not rows:
        return
    fields=list(dict.fromkeys(k for row in rows for k in row.keys()))
    with open(out,'w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def _comparison(items,out):
    import pandas as pd
    df=pd.DataFrame(overview_rows(items))
    if df.empty:
        return
    for dataset,sub in df.groupby('dataset'):
        # The raw points are crucial: do not hide folds or seeds behind an error bar.
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,(metric,title) in zip(axes,[
            ('val_accuracy','Validation accuracy'),
            ('val_layer1_raw_effective_rank','Node-response effective rank'),
            ('val_layer1_raw_mean_abs_corr','Node-response mean |correlation|')]):
            keys=list(dict.fromkeys(sub.model+' / '+sub.reg_type))
            for j,key in enumerate(keys):
                ss=sub.loc[(sub.model+' / '+sub.reg_type)==key,metric]
                vals=pd.to_numeric(ss,errors='coerce').dropna().to_numpy(dtype=float)
                if len(vals)==0:
                    continue
                ax.scatter(j+np.linspace(-0.12,0.12,len(vals)),vals,alpha=0.75,s=22)
                ax.scatter([j],[vals.mean()],marker='D',s=65,edgecolor='black',zorder=10)
            ax.set_xticks(range(len(keys)),keys,rotation=45,ha='right')
            ax.set_title(title)
            ax.grid(alpha=0.2,axis='y')
        fig.suptitle(f'{dataset}: every point is one (fold,seed,hyperparameter) run; diamond=mean')
        fig.tight_layout()
        savefig(fig,out,f'{dataset}_overview')


def heatmap_and_spectra(items,out):
    # Compare a single matched fold/seed; keep identical scale, not arbitrary ordering of filters.
    by={}
    for p,r in items:
        key=(r['dataset'],r['fold'],r['seed'])
        by.setdefault(key,[]).append((p,r))
    candidates=sorted(by.items(),key=lambda x:-len(x[1]))
    if not candidates:
        return
    (dataset,fold,seed),group=candidates[0]
    group=sorted(group,key=lambda x:x[1]['model'])
    fig,axes=plt.subplots(1,len(group),figsize=(3.7*len(group),3.8),squeeze=False)
    for ax,(p,r) in zip(axes[0],group):
        data_file=p/'best_val.npz'
        if not data_file.exists():
            continue
        with np.load(data_file,allow_pickle=False) as data:
            c=data['layer1_raw_correlation']
        cmap=plt.get_cmap('coolwarm').copy()
        cmap.set_bad('lightgray')
        img=ax.imshow(np.ma.masked_invalid(c),vmin=-1,vmax=1,cmap=cmap)
        ax.set_title(f"{r['model']} / {r['reg_type']}\nfold {fold}, seed {seed}")
        ax.set_xlabel('Filter index')
        ax.set_ylabel('Filter index')
    if len(group):
        fig.colorbar(img,ax=axes[0].tolist(),fraction=0.02,label='Pearson correlation')
    savefig(fig,out,f'{dataset}_matched_heatmaps')
    fig,ax=plt.subplots(figsize=(7,4))
    for p,r in group:
        stats=r.get('outputs',{}).get('val',{}).get('response_stats',{}).get('layer1_raw',{})
        eig=np.asarray(stats.get('eigenvalues',[]),dtype=float)
        if len(eig):
            ax.plot(np.arange(1,len(eig)+1),eig/eig.sum(),marker='o',
                    label=r['model']+' / '+r['reg_type'])
    ax.set(xlabel='Eigenvalue index',ylabel='Fraction of correlation trace',
           title=f'{dataset} / matched fold {fold}, seed {seed}')
    ax.legend()
    ax.grid(alpha=0.2)
    savefig(fig,out,f'{dataset}_matched_spectra')


def learning_curves(items,out):
    chosen=items[:12]  # keep figure readable; all histories remain on disk.
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    n=0
    for p,r in chosen:
        ep=p/'epochs.csv'
        if not ep.exists():
            continue
        with ep.open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        for ax,key in zip(axes,['val_loss','val_acc']):
            x=[];y=[]
            for row in rows:
                try:
                    x.append(int(row['epoch']))
                    y.append(float(row[key]))
                except (ValueError,KeyError):
                    continue
            if x:
                ax.plot(x,y,alpha=0.6,label=(r['dataset']+'/'+r['model']+'/'+r['reg_type']+
                                             f"/s{r['seed']}"))
        n+=1
    axes[0].set(xlabel='Epoch',ylabel='Validation cross-entropy')
    axes[1].set(xlabel='Epoch',ylabel='Validation accuracy (%)')
    axes[0].grid(alpha=0.2);axes[1].grid(alpha=0.2)
    if n:
        axes[1].legend(fontsize=7,bbox_to_anchor=(1,1),loc='upper left')
    fig.suptitle('Learning curves; model selection uses validation only')
    savefig(fig,out,'learning_curves')


def mask_gallery(items,out):
    import networkx as nx
    for path,r in items[:6]:
        masks_dir=path/'best_masks'
        files=sorted(masks_dir.glob('masks_*.json'))
        if not files:
            continue
        data=json.loads(files[-1].read_text(encoding='utf-8'))
        if not data['layers']:
            continue
        first=data['layers'][0]
        if 'masks' in first:
            masks=first['masks']
            label='Hard GKNN mask (largest component)'
        elif 'graph_samples' in first:
            masks=[dict(filter=x['filter'],**x['top_graphs'][0]) for x in first['graph_samples']]
            label='CAM most frequent graph in finite sample (not exact graph mode)'
        else:
            continue
        m=min(16,len(masks))
        cols=4;rows=int(np.ceil(m/cols))
        fig,axes=plt.subplots(rows,cols,figsize=(3*cols,2.7*rows),squeeze=False)
        for ax in axes.ravel():
            ax.axis('off')
        for i,msk in enumerate(masks[:m]):
            ax=axes.flat[i]
            g=nx.Graph()
            g.add_nodes_from(range(msk['n_nodes']))
            g.add_edges_from(msk['edges'])
            pos=nx.spring_layout(g,seed=1)
            node_labels={i:str(x) for i,x in enumerate(msk['labels'])}
            nx.draw_networkx(g,pos,ax=ax,labels=node_labels,node_size=280,
                             font_size=8,width=1.1)
            ax.set_title('Filter '+str(i+1))
            ax.axis('off')
        fig.suptitle(f"{r['dataset']} / {r['model']} / fold {r['fold']} seed {r['seed']}\n{label}")
        name=f"{r['dataset']}_{r['model']}_f{r['fold']}_s{r['seed']}_masks"
        savefig(fig,out,name)


def cam_size_figure(items,out):
    for path,r in items:
        if r['model']!='cam':
            continue
        pre=path/'snapshots'/'masks_-001.json'
        best_files=sorted((path/'best_masks').glob('masks_*.json'))
        if not pre.exists() or not best_files:
            continue
        init=json.loads(pre.read_text(encoding='utf-8'))['layers'][0]
        final=json.loads(best_files[-1].read_text(encoding='utf-8'))['layers'][0]
        a=np.asarray(init['size_probabilities'])
        b=np.asarray(final['size_probabilities'])
        if a.shape!=b.shape:
            continue
        f=min(6,len(a));x=np.arange(1,a.shape[1]+1)
        fig,axes=plt.subplots(f,1,figsize=(7,2.1*f),squeeze=False)
        for j in range(f):
            axes[j,0].plot(x,a[j],label='Initialization')
            axes[j,0].plot(x,b[j],label='Best checkpoint')
            axes[j,0].set(xlabel='Mask vertex count',ylabel=f'Filter {j+1} probability')
            axes[j,0].set_ylim(0,1)
            axes[j,0].grid(alpha=0.2)
        axes[0,0].legend()
        fig.suptitle(f"CAM size law: {r['dataset']}, fold {r['fold']}, seed {r['seed']}")
        savefig(fig,out,f"{r['dataset']}_cam_size_f{r['fold']}_s{r['seed']}")
        return


def node_response_figure(items,out):
    """Figure 8 style: same input graph, top responding filter per model.

    The top filter is chosen by its maximum response in that graph, not by
    test labels. Both input topology and node labels are saved in best_val.npz.
    """
    import networkx as nx
    by_key={}
    for p,r in items:
        if (p/'best_val.npz').exists():
            by_key.setdefault((r['dataset'],r['fold'],r['seed']),[]).append((p,r))
    if not by_key:
        return
    _,group=max(sorted(by_key.items()),key=lambda x:len(x[1]))
    group=sorted(group,key=lambda x:x[1]['model'])[:2]
    if len(group)<1:
        return
    fig,axes=plt.subplots(1,len(group),figsize=(5*len(group),4),squeeze=False)
    for ax,(path,r) in zip(axes[0],group):
        with np.load(path/'best_val.npz',allow_pickle=False) as z:
            idxs=z['node_graph']; node_idx=np.flatnonzero(idxs==0)
            labels=z['node_input_label'][node_idx]
            edges=z['edge_index']
            ed=edges[:,(np.isin(edges[0],node_idx)&np.isin(edges[1],node_idx))]
            ed=ed-node_idx[0]
            scores=z['layer1_raw'][node_idx]
            filt=int(scores.max(0).argmax())
            values=scores[:,filt]
        g=nx.Graph()
        g.add_nodes_from(range(len(node_idx)))
        g.add_edges_from(ed.T.tolist())
        pos=nx.spring_layout(g,seed=0)
        nx.draw_networkx_edges(g,pos,ax=ax,alpha=0.6)
        nx.draw_networkx_nodes(g,pos,ax=ax,node_color=values,cmap='viridis',node_size=220)
        nx.draw_networkx_labels(g,pos,ax=ax,labels={i:str(x) for i,x in enumerate(labels)},font_size=7)
        ax.set_title(f"{r['model']} / filter {filt+1}; val graph 0")
        ax.axis('off')
    fig.suptitle('Node responses: colors are not normalized across panels')
    savefig(fig,out,'node_responses_example')




def ablation_figure(items,out):
    # Faithful to Fig.9's single-mask impact, plus pairwise interaction analysis.
    for path,r in items:
        src=path/'best_val_ablation.json'
        if not src.exists():
            continue
        dat=json.loads(src.read_text(encoding='utf-8'))
        single=np.asarray(dat['single_ablation_delta_ce'],dtype=float)
        pair=np.asarray([[np.nan if x is None else x for x in row]
                         for row in dat['pairwise_interaction_delta_ce']],dtype=float)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].bar(np.arange(1,len(single)+1),single)
        axes[0].axhline(0,color='black',linewidth=0.75)
        axes[0].set(xlabel='Filter index',ylabel='CE increase after removal',
                    title='Single-mask ablation (fixed trained classifier)')
        bound=np.nanmax(np.abs(pair)) if np.isfinite(pair).any() else 1.0
        im=axes[1].imshow(np.ma.masked_invalid(pair),cmap='coolwarm',
                          vmin=-max(bound,1e-9),vmax=max(bound,1e-9))
        axes[1].set(title='Pairwise CE nonadditivity',xlabel='Filter index',ylabel='Filter index')
        fig.colorbar(im,ax=axes[1],label='Delta(ij)-Delta(i)-Delta(j)')
        fig.suptitle(f"{r['dataset']} / {r['model']} / seed {r['seed']}; nonadditivity is NOT redundancy proof")
        savefig(fig,out,f"{r['dataset']}_{r['model']}_f{r['fold']}_s{r['seed']}_ablation")
        return


def cam_edge_covariance_figure(items,out):
    for path,r in items:
        if r['model']!='cam':
            continue
        files=sorted((path/'best_masks').glob('masks_*.json'))
        if not files:
            continue
        data=json.loads(files[-1].read_text(encoding='utf-8'))['layers'][0]
        stats=data.get('edge_statistics',[])
        if not stats:
            continue
        cv=stats[0]['edge_covariance_conditional']
        if len(cv)<2:
            continue
        # New vertex t >= 2 has >=2 competing possible parents.
        t=min(3,len(cv)-1)
        matrix=np.asarray(cv[t],dtype=float)
        fig,ax=plt.subplots(figsize=(5,4))
        bound=max(float(np.abs(matrix).max()),1e-10)
        im=ax.imshow(matrix,cmap='coolwarm',vmin=-bound,vmax=bound)
        fig.colorbar(im,ax=ax,label='Conditional edge covariance')
        ax.set(xlabel='Candidate edge (j,t)',ylabel='Candidate edge (k,t)',
               title=f"CAM, filter 1; new vertex t={t+1} (1-indexed)")
        fig.suptitle('Dependence of competing attachments (conditional on vertex existence)')
        savefig(fig,out,f"{r['dataset']}_cam_f{r['fold']}_s{r['seed']}_edge_covariance")
        return


def effort_accuracy_figure(items,out):
    import pandas as pd
    rows=[]
    for p,r in items:
        ep=p/'epochs.csv'
        if not ep.exists():
            continue
        df=pd.read_csv(ep)
        pairs=pd.to_numeric(df.get('train_kernel_evaluated_pairs',pd.Series(dtype=float)),errors='coerce').sum()
        kernel_sec=pd.to_numeric(df.get('train_kernel_seconds',pd.Series(dtype=float)),errors='coerce').sum()
        rows.append(dict(method=r['model']+'/'+r['reg_type'],
                         dataset=r['dataset'],pairs=pairs,seconds=kernel_sec,
                         accuracy=r.get('outputs',{}).get('val',{}).get('accuracy')))
    if not rows:
        return
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for row in rows:
        if row['accuracy'] is None:
            continue
        axes[0].scatter(row['pairs'],row['accuracy'],label=row['method'],alpha=0.75)
        axes[1].scatter(row['seconds'],row['accuracy'],label=row['method'],alpha=0.75)
    axes[0].set(xlabel='GraKeL transform pairs observed during training',ylabel='Val accuracy')
    axes[1].set(xlabel='GraKeL evaluation wall seconds during training',ylabel='Val accuracy')
    axes[0].grid(alpha=0.2);axes[1].grid(alpha=0.2)
    handles,labels=axes[1].get_legend_handles_labels()
    uniq=dict(zip(labels,handles))
    fig.legend(uniq.values(),uniq.keys(),loc='upper center',ncol=min(4,len(uniq)),fontsize=7)
    fig.suptitle('Diagnostic compute budget; not a full machine-level operation count')
    savefig(fig,out,'kernel_budget_vs_accuracy')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='results/cam_gknn')
    p.add_argument('--out',default='figures/cam_gknn')
    p.add_argument('--dataset',default=None)
    opts=p.parse_args()
    items=collect(opts.root,opts.dataset)
    if not items:
        raise SystemExit('No summary.json files found under '+opts.root)
    dest=Path(opts.out);dest.mkdir(parents=True,exist_ok=True)
    save_csv(overview_rows(items),dest/'overview.csv')
    _comparison(items,dest)
    heatmap_and_spectra(items,dest)
    learning_curves(items,dest)
    mask_gallery(items,dest)
    cam_size_figure(items,dest)
    node_response_figure(items,dest)
    ablation_figure(items,dest)
    cam_edge_covariance_figure(items,dest)
    effort_accuracy_figure(items,dest)
    print('Saved',len(items),'runs to',dest)


if __name__=='__main__':
    main()
