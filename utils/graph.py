import torch



def batch_knn_graph(loc: torch.Tensor, k: int, valid: torch.Tensor = None):
    """
    Given locations "loc" - torch.Tensor of shape [batch_size, num_nodes, 2]
    constructs edges to each node from its k nearest neighbors.
    Merge graphs in a batch into a large graph as subgraphs.
    Node labels are integers chosen so that each node (even not valid) in the large graph is unique.

    Args:
    - loc: tensor of shape [batch_size, num_nodes, 2], representing the locations of each node in 2D space.
    - k: int, specifying the number of nearest neighbors to connect each node to.
    - valid: tensor of shape [batch_size, num_nodes], representing the validity of each node.

    Returns:
    - edge_index: tensor of shape [2, num_edges], where num_edges is the total number of edges
      in the large batched graph.
    """
    # loc: [Bs, Nn, 2], valid: [Bs, Nn]
    Bs, Nn, _ = loc.size()
    device = loc.device
    # Compute pairwise distances between nodes
    dists = torch.cdist(loc, loc) # [Bs, Nn, Nn]
    dists.diagonal(dim1=1, dim2=2).fill_(torch.inf)
    dists.masked_fill_(~valid.view(Bs, 1, Nn), torch.inf)
    dists.masked_fill_(~valid.view(Bs, Nn, 1), torch.inf)
    # Find the indices of the k nearest neighbors for each node
    # top_dists, indices = torch.topk(dists, k=k, largest=False) # [Bs, Nn, k]
    # WARNING: topk produces incorrect results in Pytorch1.13!!! https://github.com/pytorch/pytorch/issues/95455
    dists, indices = torch.sort(dists)
    top_dists, indices = dists[...,:k].contiguous(), indices[...,:k]
    # Create an edge tensor by stacking pairs of nodes that are k-nearest neighbors
    targets = torch.arange(Nn, device=device).view(1, -1, 1).expand_as(indices)
    # print(targets)
    edge_index = torch.stack([indices.reshape(Bs, -1),
                              targets.reshape(Bs, -1)], dim=0) # [2, Bs, Ne], Ne = Nn * k
    # Merging into large graph
    offset = (torch.arange(Bs, device=device) * Nn).view(1, -1, 1) # [1, Bs, 1]
    edge_index += offset
    # Filtering inf and nan (not valid edges)
    valid_edges = top_dists.view(Bs, -1) < torch.inf
    edge_index = edge_index[:, valid_edges] # [2, NE]
    return edge_index


@torch.jit.script
def batch_complete_bipartite_graph(g1_valid: torch.Tensor, g2_valid: torch.Tensor):
    """
    For each element of batch: given nodes of two graphs, constructs edges of a directed graph such that every node in
    the first graph is connected to every node in the second graph. Merge graphs in a batch into a large graph.
    Node labels are integers chosen so that each node (even not valid) in the large graph is unique.

    Args:
    - g1_valid: tensor of shape [batch_size, num_nodes_g1]
    - g2_valid: tensor of shape [batch_size, num_nodes_g2]

    Returns:
    - edge_index: tensor of shape [2, num_edges]
    """
    Bs, Nn1 = g1_valid.size(0), g1_valid.size(1)
    Nn2 = g2_valid.size(1)
    nodes = torch.arange(Bs*(Nn1+Nn2), device=g1_valid.device).view(Bs, Nn1+Nn2)
    g1_nodes = nodes[:, :Nn1].view(Bs, -1, 1).expand(Bs, Nn1, Nn2)
    g2_nodes = nodes[:, Nn1:].view(Bs, 1, -1).expand(Bs, Nn1, Nn2)
    g1_valid = g1_valid.view(Bs, -1, 1).expand(Bs, Nn1, Nn2)
    g2_valid = g2_valid.view(Bs, 1, -1).expand(Bs, Nn1, Nn2)
    valid = torch.stack([g1_valid.reshape(Bs, -1), g2_valid.reshape(Bs, -1)], dim=-1).all(-1) # [Bs, Nn1 * Nn2]
    all_edges = torch.stack([g1_nodes.reshape(Bs, -1), g2_nodes.reshape(Bs, -1)], dim=0) # [2, Bs, Nn1 * Nn2]
    edge_index = all_edges[:, valid] # [2, Ne]
    return edge_index




if __name__ == '__main__':
    torch.manual_seed(123)
    Bs, Nn = 2, 6
    loc = torch.randint(3, (Bs, Nn+1, 2), dtype=torch.float32).to("cuda:1")
    mask = torch.randn((Bs, Nn), device=loc.device) > 0.5
    depot_mask = torch.ones((Bs, 1), dtype=torch.bool, device=loc.device)
    print("loc:\n", loc)
    print("valid:\n", mask)
    mask_no_depot = torch.cat([~depot_mask, mask], dim=1)
    knn_edges = batch_knn_graph(loc, k=3, valid=mask_no_depot)
    bi_edges1 = batch_complete_bipartite_graph(depot_mask, mask)
    bi_edges2 = bi_edges1.flip(0)
    print("knn_edges:\n", knn_edges)
    print("bi_edges:\n", bi_edges1)
    edge_index = torch.cat([knn_edges, bi_edges1, bi_edges2], dim=1)  # (2, n_edges)
    print("all:\n", edge_index, end="\n"*5)

