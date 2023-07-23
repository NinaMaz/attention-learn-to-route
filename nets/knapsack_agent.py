import torch
from nets.graph_encoder import *
from nets.graph_network import *
from nets.agent_gnn import *
from utils.boolean_nonzero import sample_nonzero



class KnapsackModelAC(torch.nn.Module):
    def __init__(self, embedding_dim, encoder_cls, encoder_params):
        super().__init__()

        Encoder = eval(encoder_cls)
        self.embedder = Encoder(**encoder_params)
        # for VRP
        node_dim = 3
        if embedding_dim is not None:
            self.init_embed_depot = torch.nn.Linear(2, embedding_dim)
            self.init_embed = torch.nn.Linear(node_dim, embedding_dim)
            self.embed_fn = self._init_embed
        else:
            print("No embedding")
            self.embed_fn = self._cat_features

        self.policy_layers = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 1)
        )

        self.val_layers = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 1),
        )
        # self.layers[-1].bias.data.fill_(10)
        # self.out_act = torch.nn.Sigmoid()


    def forward(self, input, mask):
        # loc: [B, N-1, 2], depot: [B, 2], demand: [B, N-1], mask: [B, N]
        embeddings, embedding = self.embedder(self.embed_fn(input), mask, obs=input)  # [B, N, dim], [B, N]
        logits = self.policy_layers(embeddings)  # [batch_size, n_nodes, 1]
        value = self.val_layers(embedding) # [batch_size, 1]
        #rand_var = torch.bernoulli(torch.ones_like(input)*bernoulli_prob)
        logits, value = logits.squeeze(2), value.squeeze(1)
        actions, _ = sample_nonzero(logits[:,1:], mask[:,1:], dim = 1)
        return logits, actions, value

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


    def _cat_features(self, input):
        depot = input["depot"]
        return torch.cat([
            torch.cat([depot, torch.full((depot.size(0), 1), -1, device=depot.device)], -1).unsqueeze(1),
            torch.cat([input["loc"], input["demand"].unsqueeze(-1)], -1)
        ], 1)