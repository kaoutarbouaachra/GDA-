"""A reproducible demo of learned CAM size, connected graphs and edge dependence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from cam import CAMKernel

def main():
    torch.manual_seed(1)
    cam=CAMKernel(nodes=5,labels=3,filters=2,mc_samples=4,eval_samples=16)
    print('P(mask size = 1, ..., 5):\n',cam.size_probabilities().detach().numpy())
    print('P(edge (j, 3) | size >= 4):',cam.edge_marginals(0,3).detach().numpy())
    print('\nSample connected graphs:')
    for i in range(5):
        trace=cam.sample_trace(0)
        print(f'  #{i+1}: vertices={trace.size}, parents={trace.parents}, '
              f'edges={trace.edges}, node_labels={trace.labels}')
        assert torch.allclose(trace.log_prob,cam.trace_log_probability(trace),atol=1e-6)

if __name__=='__main__':
    main()
