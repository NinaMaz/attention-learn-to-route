import torch
import math

from algorithms.utils import Trajectory
from nets.graph_encoder import *
from nets.graph_network import *
from nets.agent_gnn import *
from nets.attention_model import *

INF = 1e20

class MultiHeadAttentionLogits(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads, tanh_clipping=10):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads"
        embed_dim = embed_dim // n_heads
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.norm_factor = 1 / math.sqrt(embed_dim)  # See Attention is all you need
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, embed_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, embed_dim))
        self.tanh_clipping = tanh_clipping
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param k: data (batch_size, n_key, input_dim)
        :param mask: mask (batch_size, n_query, n_key) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        batch_size, n_key, input_dim = k.size()
        n_query = q.size(1)

        k = k.view(-1, input_dim)
        q = q.view(-1, input_dim)

        # last dimension can be different for keys and values
        shape_k = (self.n_heads, batch_size, n_key, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(q, self.W_query).view(shape_q)  # (n_heads, batch_size, n_query, embed_dim)
        K = torch.matmul(k, self.W_key).view(shape_k)  # (n_heads, batch_size, n_key, embed_dim)

        # Calculate compatibility (batch_size, n_query, n_key)
        attn_logits = self.norm_factor * torch.matmul(Q, K.transpose(2, 3)).mean(0)

        if self.tanh_clipping > 0:
            attn_logits = torch.tanh(attn_logits) * self.tanh_clipping
        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(batch_size, n_query, n_key)
            attn_logits = torch.where(mask, -INF, attn_logits)  # very small value doesn't produce NaN in softmax
        return attn_logits


class Decoder(torch.nn.Module):
    def __init__(self, dim, n_heads, dim_feedforward, num_layers, dropout, kdim, vdim):
        super().__init__()
        # self.layers = torch.nn.TransformerDecoder(
        #     torch.nn.TransformerDecoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim_feedforward,
        #         batch_first=True, dropout=dropout, norm_first=True),
        #     num_layers=num_layers,
        # )
        self.attn1 = torch.nn.MultiheadAttention(dim, n_heads, dropout=dropout, kdim=kdim, vdim=vdim, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(dim, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_feedforward, dim),
        )
        self.attn2 = MultiHeadAttentionLogits(dim, dim, 1)

    def forward(self, q, k, v, mask=None):  # pre-LN
        q = q + self.attn1(self.norm1(q), k, v, key_padding_mask=mask)[0]
        q = q + self.linear(self.norm2(q))
        attn = self.attn2(q, k, mask=mask)
        return attn


class Agent(torch.nn.Module):
    def __init__(self,
                 encoder_cls,
                 encoder_params,
                 embedding_dim,
                 hidden_dim=128,
                 ff_hidden_dim=256,
                 n_heads=1,
                 num_layers=2,
                 dropout=0.1,
                 node_features_option="once",
                 # encoder_steps_0=10,
                 # encoder_steps=10
        ):
        super().__init__()
        self.enc = eval(encoder_cls)(**encoder_params)
        self.node_features_option = node_features_option

        # self.init_embed_depot = torch.nn.Linear(2, embedding_dim)
        # self.init_embed = torch.nn.Linear(3, embedding_dim)
        # self.emb_fn = self._init_embed
        self.emb_proj = torch.nn.Linear(3, embedding_dim)
        self.emb_fn = self._cat_features

        self.actor = Decoder(dim=hidden_dim, n_heads=n_heads, dim_feedforward=ff_hidden_dim, num_layers=num_layers,
                             dropout=dropout, kdim=hidden_dim, vdim=hidden_dim)
        self.query_proj = torch.nn.Linear(hidden_dim * 2 + 1, hidden_dim)

        self.val_layers = torch.nn.Sequential(
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

        # self.encoder_steps_0 = encoder_steps_0
        # self.encoder_steps = encoder_steps

    # def _init_embed(self, coords, demands):
    #     node_loc, depot_loc = coords[:, 1:], coords[:, :1]
    #     return torch.cat([
    #         self.init_embed_depot(depot_loc),
    #         self.init_embed(torch.cat([node_loc, demands[:, :, None]], -1))
    #     ], 1)

    def _cat_features(self, coords, demands):
        cat = torch.cat([
            coords,
            torch.cat([torch.full((demands.size(0), 1), -1, device=demands.device), demands], 1).unsqueeze(-1)
        ], -1)
        return self.emb_proj(cat)

    def forward(self, obs, mask, greedy=False):
        current_node, used_capacity = obs["current"], obs["used_capacity"]  # [batch_size], [batch_size]

        if "graph_feature" in obs:
            graph_feature = obs["graph_feature"]
            node_features = obs["features"]
            mem_node_features = node_features.clone()
        elif "coords" in obs and "demands" in obs:
            node_features = self.emb_fn(obs.get("coords"), obs.get("demands"))  # [batch_size, graph_size+1, dim]
            mem_node_features, node_features, graph_feature = self.enc(node_features, mask.logical_not(),
                start_pos=current_node)  # [batch_size, graph_size, hidden_dim], [batch_size, hidden_dim]
        elif "features" in obs:
            node_features = obs.get("features")
            mem_node_features, node_features, graph_feature = self.enc(node_features, mask.logical_not(),
                start_pos=current_node)  # [batch_size, graph_size, hidden_dim], [batch_size, hidden_dim]
        else:
            raise RuntimeError

        batch_size, graph_size, _ = node_features.size()


        # assert node_features.shape == torch.Size((batch_size, graph_size, 128)), node_features.shape
        # assert graph_feature.shape == torch.Size((batch_size, 128))

        query = torch.cat([
            graph_feature,
            node_features.gather(1, current_node.view(-1, 1, 1).expand(-1, -1, node_features.size(-1))).squeeze(1),
            used_capacity.unsqueeze(-1),
        ], -1).unsqueeze(1)  # [batch_size, 1, hidden_dim * 2 + 1]
        # assert query.shape == torch.Size([batch_size, 1, 128 * 2 + 1]), query.shape

        query = self.query_proj(query)
        logits = self.actor(query, node_features, node_features, mask=mask).squeeze(1)  # logits: [batch_size, graph_size]
        # assert logits.shape == torch.Size([batch_size, graph_size])

        if greedy:
            action = torch.argmax(logits, dim=-1).view(-1)  # [batch_size]
        else:
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).view(-1)   # [batch_size]
        # assert action.shape == torch.Size([batch_size])

        value = self.val_layers(graph_feature).squeeze(-1)  # [batch_size]

        return logits, action, value, mem_node_features, graph_feature

    def play(self, state, greedy=False):
        # device = next(iter(self.parameters())).device
        traj = Trajectory()
        obs = {"coords": state.coords, "demands": state.demand,
               "current": state.get_current_node(), "used_capacity": state.get_used_capacity()}

        step = 0
        while not state.all_finished():
            step += 1
            mask = state.get_mask()
            done = state.get_finished()

            logits, action, value, node_features, graph_feature = self.forward(obs, mask, greedy)
            # print(mask)
            # print(logits)
            # print(action)

            state = state.update(action)
            cost = state.get_cost()
            # assert(action.requires_grad == False and cost.requires_grad == False and mask.requires_grad == False)

            if not greedy:
                traj.append("obs", obs)
                traj.append("logits", logits)
                traj.append("costs", cost)
                traj.append("values", value)
                traj.append("actions", action)
                traj.append("valid", mask.logical_not())
                traj.append("done", done)

            if self.node_features_option == "update":
                obs = {"features": node_features.detach(), "current": state.get_current_node(), "used_capacity": state.get_used_capacity()}
            elif self.node_features_option == "once":
                obs = {"features": node_features, "current": state.get_current_node(),
                       "graph_feature": graph_feature, "used_capacity": state.get_used_capacity()}
            else:
                obs.update({"current": state.get_current_node(), "used_capacity": state.get_used_capacity()})

            # print(step)
            # print(f"Allocated: {torch.cuda.memory_allocated(logits.device) / 1024 ** 3:.1f} GB")
            # print(f"Cached: {torch.cuda.memory_reserved(logits.device) / 1024 ** 3:.1f} GB")

            # from pytorch_memlab import MemReporter
            # reporter = MemReporter(self)
            # reporter.report(device=torch.device("cuda:1"))

        total_cost = state.get_final_cost()
        # print(total_cost)
        return total_cost, traj

