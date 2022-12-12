import torch 
from nets.graph_encoder import GraphAttentionEncoder

class KnapsackModel(torch.nn.Module):
    def __init__(self, embedding_dim, n_heads=8, n_encode_layers=2, normalization="batch"):
        super().__init__()

        Encoder = GraphAttentionEncoder
        self.embedder = Encoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
            normalization=normalization,
        )
        # for VRP
        node_dim = 3
        self.init_embed_depot = torch.nn.Linear(2, embedding_dim)
        self.init_embed = torch.nn.Linear(node_dim, embedding_dim)

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        self.out_act = torch.nn.Sigmoid()
    def forward(self, input, bernoulli_prob = 0.5):  
        embeddings, _ = self.embedder(self._init_embed(input))  
        logits = self.layers(embeddings)
        #rand_var = torch.bernoulli(torch.ones_like(input)*bernoulli_prob)
        return self.out_act(logits)

    def _init_embed(self, input):

        features = ("demand",)

        return torch.cat(
            (
                self.init_embed_depot(input["depot"])[:, None, :],
                self.init_embed(
                    torch.cat(
                        (
                            input["loc"],
                            *(input[feat][:, :, None] for feat in features),
                        ),
                        -1,
                    )
                ),
            ),
            1,
        )
