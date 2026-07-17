import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineRouter(nn.Module):
    """Official MoA-style cosine router.

    The router projects token features to a lower-dimensional routing space,
    then scores them against learnable expert embeddings with cosine similarity.
    """

    def __init__(
        self,
        dim,
        num_experts,
        normalize_input=True,
        proj_dim=256,
        init_temperature=0.5,
        max_logit_scale=100.0,
    ):
        super().__init__()

        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}.")
        if num_experts < 1:
            raise ValueError(f"num_experts must be positive, got {num_experts}.")
        if proj_dim < 1:
            raise ValueError(f"proj_dim must be positive, got {proj_dim}.")
        if init_temperature <= 0:
            raise ValueError(
                f"init_temperature must be positive, got {init_temperature}."
            )

        self.dim = dim
        self.num_experts = num_experts
        self.normalize_input = normalize_input
        self.proj_dim = proj_dim
        self.max_logit_scale = max_logit_scale

        self.cosine_projector = nn.Linear(dim, proj_dim)
        self.sim_matrix = nn.Parameter(torch.empty(proj_dim, num_experts))
        self.logit_scale = nn.Parameter(
            torch.tensor(float(1.0 / init_temperature)).log()
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.cosine_projector.reset_parameters()
        nn.init.normal_(self.sim_matrix, mean=0.0, std=0.01)

    def forward(self, x):
        if x.size(-1) != self.dim:
            raise ValueError(f"Expected x last dim {self.dim}, got {x.size(-1)}.")

        projected = self.cosine_projector(x)
        if self.normalize_input:
            projected = F.normalize(projected, dim=-1)

        sim_matrix = F.normalize(self.sim_matrix, dim=0)
        logits = torch.matmul(projected, sim_matrix)
        scale = self.logit_scale.exp().clamp(max=self.max_logit_scale)
        logits = logits * scale
        gates = torch.softmax(logits, dim=-1)
        return gates, logits
