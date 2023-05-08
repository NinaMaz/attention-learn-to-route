import torch
import numpy as np
import torch.nn as nn
from utils.graph import batch_knn_graph, batch_complete_bipartite_graph
from torch_geometric.nn.models import GAT


class GATEncoder(GAT):
    def __init__(
        self,
        n_heads,
        embed_dim,
        n_layers,
        k=None,
        **kwargs
    ):
        super().__init__(in_channels=embed_dim, hidden_channels=embed_dim, num_layers=n_layers, heads=n_heads)
        self.k = k

    def forward(self, x, mask, obs):
        batch_size, n_nodes, feature_dim = x.shape

        # if k is none, select it according to the graph size
        if self.k is None:
            self.k = 10 * int(np.ceil(np.log(n_nodes)/np.log(10)))
            print(f"Number of nearest neighbors K is set to {self.k}")

        loc, depot = obs['loc'], obs['depot'] # (batch_size, n_nodes-1, 2), (batch_size, 2)
        loc = torch.cat([depot.view(-1, 1, 2), loc], dim=1) # (batch_size, n_nodes, 2)

        # construct graph
        mask_no_depot = mask.index_fill(1, index=torch.tensor([0], device=mask.device), value=False)
        knn_edges = batch_knn_graph(loc, k=self.k, valid=mask_no_depot)  # directed
        depot_edges = batch_complete_bipartite_graph(mask[:, :1], mask[:, 1:])  # directed
        edge_index = torch.cat([knn_edges, depot_edges, depot_edges.flip(0)], dim=1)  # (2, n_edges)

        # concat features x: (batch_size, n_nodes, feature_dim) -> (batch_size * n_nodes, feature_dim)
        x = x.view(-1, feature_dim)

        # forward
        x = super().forward(x, edge_index=edge_index, edge_weight=None)

        # back to batch
        x = x.view(batch_size, n_nodes, -1)

        return (
            # node embeddings (batch_size, graph_size, embed_dim)
            x,
            # average to get embedding of graph, (batch_size, embed_dim)
            x.sum(dim=1) / np.sqrt(x.size(1))
        )



if __name__ == '__main__':
    Bs = 100
    N = 4000
    n_heads = 5
    d = n_heads * 4
    device = 'cuda'

    x = torch.randn(Bs, N, d).to(device=device)
    mask = (torch.randn(Bs, N) > 0).to(device=device)
    obs = {'loc': torch.randn(Bs, N-1, 2).to(device=device), 'depot': torch.randn(Bs, 2).to(device=device)}

    g = GATEncoder(n_heads, d, 2, 'sym', 5).to(device=device)
    x, g = g(obs, x, mask)
    print(x.shape, g.shape)

    # from .graph_encoder import GraphAttentionEncoderMask
    # g = GraphAttentionEncoderMask(n_heads, d, 2).to(device=device)
    # x, g = g(x, mask)

    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
    print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))