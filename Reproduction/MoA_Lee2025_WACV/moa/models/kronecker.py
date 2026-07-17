import torch
import torch.nn as nn


def glorot_uniform_(tensor):
    return nn.init.xavier_uniform_(tensor, gain=2 ** 0.5)


class SharedKroneckerRule(nn.Module):
    """Shared official-style MoA Kronecker rule matrices A_t.

    The official implementation parameterizes each A_t as a product of left and
    right factors instead of directly storing a full [P, P, P] tensor.
    """

    def __init__(self, phm_dim, init_std=0.01):
        super().__init__()
        if phm_dim < 1:
            raise ValueError(f"phm_dim must be positive, got {phm_dim}.")
        self.phm_dim = phm_dim
        self.init_std = init_std
        self.left = nn.Parameter(torch.empty(phm_dim, phm_dim, 1))
        self.right = nn.Parameter(torch.empty(phm_dim, 1, phm_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.left, -self.init_std, self.init_std)
        nn.init.uniform_(self.right, -self.init_std, self.init_std)

    @property
    def rule(self):
        return torch.bmm(self.left, self.right)


class MoAKroneckerExpert(nn.Module):
    """Real-valued Kronecker expert used by MoA.

    The effective weight is:

        W = sum_t A_t kron B_t

    where A_t is shared across injected layers and experts, while each expert
    owns a low-rank B_t = U_t V_t.
    """

    def __init__(
        self,
        in_features,
        out_features,
        phm_dim=64,
        rank=1,
        shared_rule=None,
        init_std=0.01,
        dropout=0.5,
    ):
        super().__init__()

        if in_features % phm_dim != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by phm_dim={phm_dim}."
            )
        if out_features % phm_dim != 0:
            raise ValueError(
                f"out_features={out_features} must be divisible by phm_dim={phm_dim}."
            )
        if rank < 1:
            raise ValueError(f"rank must be positive, got {rank}.")

        self.in_features = in_features
        self.out_features = out_features
        self.phm_dim = phm_dim
        self.rank = rank
        self.in_block = in_features // phm_dim
        self.out_block = out_features // phm_dim

        if shared_rule is None:
            shared_rule = SharedKroneckerRule(phm_dim=phm_dim, init_std=init_std)
        self.shared_rule = shared_rule

        self.u = nn.Parameter(torch.empty(phm_dim, self.in_block, rank))
        self.v = nn.Parameter(torch.empty(phm_dim, rank, self.out_block))
        self.init_std = init_std
        self.kdropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for term in range(self.phm_dim):
            glorot_uniform_(self.u.data[term])
            glorot_uniform_(self.v.data[term])

    def low_rank_blocks(self):
        """Return B_t blocks with shape [P, in_block, out_block]."""
        return torch.bmm(self.u, self.v)

    def forward(self, x):
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected x last dim {self.in_features}, got {x.size(-1)}."
            )

        weight = self.effective_weight(apply_dropout=True)
        return torch.matmul(x, weight)

    def effective_weight(self, apply_dropout=False):
        """Materialize H with shape [in_features, out_features]."""
        rule = self.shared_rule.rule
        b_blocks = self.low_rank_blocks()
        weight = torch.einsum(
            "tab,tio->aibo",
            rule,
            b_blocks,
        ).reshape(self.in_features, self.out_features)
        if apply_dropout:
            weight = self.kdropout(weight)
        return weight
