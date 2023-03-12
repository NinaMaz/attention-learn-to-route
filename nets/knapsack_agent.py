import torch
from nets.graph_encoder import GraphAttentionEncoder
from utils.boolean_nonzero import sample_nonzero


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
        # self.layers[-1].bias.data.fill_(10)
        # self.out_act = torch.nn.Sigmoid()

    def forward(self, input, bernoulli_prob = 0.5):
        # loc: [B, N, 2], depot: [B, 2], demand: [B, N]
        embeddings, _ = self.embedder(self._init_embed(input))  # [batch_size, n_nodes, dim]
        logits = self.layers(embeddings)  # [batch_size, n_nodes, dim]
        #rand_var = torch.bernoulli(torch.ones_like(input)*bernoulli_prob)
        mask, _ = sample_nonzero(logits[:,1:,:], dim = 1)
        return logits, mask

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




class KnapsackModelAC(torch.nn.Module):
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

        self.policy_layers = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 1)
        )

        self.val_layers = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 1),
        )
        # self.layers[-1].bias.data.fill_(10)
        # self.out_act = torch.nn.Sigmoid()


    def forward(self, input, bernoulli_prob = 0.5):
        # loc: [B, N, 2], depot: [B, 2], demand: [B, N]
        embeddings, _ = self.embedder(self._init_embed(input))  # [batch_size, n_nodes, dim]
        logits = self.policy_layers(embeddings)  # [batch_size, n_nodes, 1]
        value = self.val_layers(embeddings.mean(dim=1)) # [batch_size, 1]
        #rand_var = torch.bernoulli(torch.ones_like(input)*bernoulli_prob)
        mask, _ = sample_nonzero(logits[:,1:,:], dim = 1)
        return logits, mask, value

    def _init_embed(self, input):
        features = ("demand",)
        return torch.cat(
            (
                self.init_embed_depot(input["depot"])[:, None, :],
                self.init_embed(
                    torch.cat((input["loc"], *(input[feat][:, :, None] for feat in features)), -1)
                )
            ),
            1,
        )
