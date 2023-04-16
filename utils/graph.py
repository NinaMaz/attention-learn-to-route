import torch




def batch_knn_graph(loc: torch.Tensor, k: int, valid: torch.Tensor = None):
    """
    Given locations "loc" - torch.tensor of shape [batch_size, num_nodes, 2]
    constructs edges of a directed graph such that every node is connected to its "k" nearest neighbours.
    Merge graphs in a batch into a large graph as subgraphs.

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
    if valid is not None:
        loc = torch.where(valid.view(Bs, Nn, 1), loc, torch.inf)
    # Compute pairwise distances between nodes
    dists = torch.cdist(loc, loc) # [Bs, Nn, Nn]
    dists.diagonal(dim1=1, dim2=2).fill_(torch.inf)
    # Find the indices of the k nearest neighbors for each node
    top_dists, indices = torch.topk(dists, k=k, largest=False) # [Bs, Nn, k]
    # Create an edge tensor by stacking pairs of nodes that are k-nearest neighbors
    targets = torch.arange(Nn, device=device).view(1, -1, 1).expand_as(indices)
    edge_index = torch.stack([indices.reshape(Bs, -1),
                              targets.reshape(Bs, -1)], dim=0) # [2, Bs, Ne], Ne = Nn * k
    # Merging into large graph
    offset = (torch.arange(Bs, device=device) * Nn).view(1, -1, 1)
    edge_index += offset
    # Filtering inf and nan (not valid edges)
    valid_edges = top_dists.view(Bs, -1) < torch.inf
    edge_index = edge_index[:, valid_edges] # [2, NE]
    return edge_index



def complete_bipartite_graph(g1_nodes, g2_nodes):
    """
    Given nodes of two graphs, constructs edges of a directed graph such that every node in the first graph
    is connected to every node in the second graph.

    Args:
    - g1_nodes: tensor of shape [num_nodes_g1]
    - g2_nodes: tensor of shape [num_nodes_g2]

    Returns:
    - edge_index: tensor of shape [2, num_edges]
    """
    g1_nodes = g1_nodes.view(-1, 1).expand(-1, g2_nodes.size(0))
    g2_nodes = g2_nodes.view(1, -1).expand(g1_nodes.size(0), -1)
    edge_index = torch.stack([g1_nodes.reshape(-1), g2_nodes.reshape(-1)], dim=0)
    return edge_index




if __name__ == '__main__':
    # Define a simple 3-node graph with (x, y) coordinates
    loc = torch.tensor([[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]], dtype=torch.float).to("cuda:1")
    loc = torch.cat([loc, loc], dim=0)

    # valid = (torch.randn(loc.shape[:2]) > 0).to("cuda:1")
    valid = torch.ones(loc.shape[:2], dtype=torch.bool).to("cuda:1")
    valid[0, 1] = False
    valid[1] = False
    valid[1,0] = True
    valid[1,3] = True
    print(valid)

    # Compute edges connecting each node to its 2 nearest neighbors
    # k = loc.shape[1] // 10
    k = 2

    edge_index = batch_knn_graph(loc, k)
    print(edge_index.is_contiguous())
    print(edge_index)

    edge_index = batch_knn_graph(loc, k, valid)
    print(edge_index.is_contiguous())
    print(edge_index)
    # assert((edge_index == edge_index2).all())



    # loc = torch.randn(1000, 1000, 2).to("cuda:1")
    # # valid = (torch.randn(1000, 1000) > 0).to("cuda:1")
    # k = 100
    # from torch.utils.benchmark import Timer
    # timer = Timer(
    #     stmt="my_function(loc, k)",
    #     globals={"my_function": batch_knn_graph, "loc": loc, "k": k}
    # )
    # mean_execution_time = timer.timeit(50).mean
    # print("Mean execution time:", mean_execution_time)


    # test complete_bipartite_graph
    g1_labels = torch.tensor([0, 1, 2, 3])
    g2_labels = torch.tensor([4, 5, 6])
    edge_index = complete_bipartite_graph(g1_labels, g2_labels)
    print(edge_index)


    # test all together
    print("TEST")

    loc = torch.randint(3, (2, 7, 2)).to("cuda:1")
    mask = (torch.randn(loc.shape[:2]) > 0).to("cuda:1")
    print(loc)
    print(mask)
    batch_size, n_nodes, _ = loc.shape
    mask_no_depot = torch.cat([torch.zeros(batch_size, 1, dtype=mask.dtype, device=loc.device), mask[:, 1:]], dim=1)
    depot_nodes = torch.arange(batch_size, device=loc.device) * n_nodes  # (batch_size)
    valid_nodes = mask_no_depot.view(-1).nonzero(as_tuple=False)  # (n_valid_nodes)
    edge_indices = [batch_knn_graph(loc, k=3, valid=mask_no_depot)]
    edge_indices += [complete_bipartite_graph(depot_nodes, valid_nodes)]
    edge_indices += [complete_bipartite_graph(valid_nodes, depot_nodes)]
    edge_index = torch.cat(edge_indices, dim=1)  # (2, n_edges)
    # for t in edge_indices:
    #     print(t)

