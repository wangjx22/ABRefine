"""Code."""
import torch
from torch.utils.checkpoint import checkpoint


class HieraAttUpdate(torch.nn.Module):
    """Define Class HieraAttUpdate."""

    def __init__(self, Ns, Nh, Nk):
        """Run __init__ method."""
        # code.
        super(HieraAttUpdate, self).__init__()
        self.Ns = Ns
        self.Nh = Nh
        self.Nk = Nk
        self.vt0emb = torch.nn.Sequential(
            torch.nn.Linear(2 * Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, 2 * Nk * Nh),
        )

        self.et0emb = torch.nn.Sequential(
            torch.nn.Linear(6 * Ns + 1, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Nk),
        )

        self.et1emb = torch.nn.Sequential(
            torch.nn.Linear(6 * Ns + 1, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, 3 * Nk),
        )

        self.evm = torch.nn.Sequential(
            torch.nn.Linear(6 * Ns + 1, 2 * Ns),
            torch.nn.ELU(),
            torch.nn.Linear(2 * Ns, 2 * Ns),
            torch.nn.ELU(),
            torch.nn.Linear(2 * Ns, 2 * Ns),
        )

        self.t0t1emb = torch.nn.Sequential(
            torch.nn.Linear(Nh * Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
            torch.nn.ELU(),
            torch.nn.Linear(Ns, Ns),
        )

        self.t2t2emb = torch.nn.Sequential(
            torch.nn.Linear(Nh * Ns, Ns, bias=False),
        )

        self.sdk = torch.sqrt(torch.tensor(Nk).float())


    def forward(self, ft0, ft1, ft0_nn, ft1_nn, d_nn, r_nn):
        """Run forward method."""
        # code.
        N, n, S = ft0_nn.shape
        X_n = torch.cat([
            ft0,
            torch.norm(ft1, dim=1),
        ], dim=1)  # [N, 2*S]

        X_e = torch.cat([
            d_nn.unsqueeze(2),
            X_n.unsqueeze(1).repeat(1, n, 1),
            ft0_nn,
            torch.norm(ft1_nn, dim=2),
            torch.sum(ft1.unsqueeze(1) * r_nn.unsqueeze(3), dim=2),
            torch.sum(ft1_nn * r_nn.unsqueeze(3), dim=2),
        ], dim=2)

        Q = self.vt0emb.forward(X_n).view(N, 2, self.Nh, self.Nk)

        Kt0 = self.et0emb.forward(X_e).view(N, n, self.Nk).transpose(1, 2)

        Kt1 = torch.cat(torch.split(self.et1emb.forward(X_e), self.Nk, dim=2), dim=1).transpose(1, 2)

        V = self.evm.forward(X_e).view(N, n, 2, S).transpose(1, 2)

        Vt1 = torch.cat([
            V[:, 1].unsqueeze(2) * r_nn.unsqueeze(3),
            ft1.unsqueeze(1).repeat(1, n, 1, 1),
            ft1_nn,
        ], dim=1).transpose(1, 2)  # [N, 3, 3*n, S]

        Mt0 = torch.nn.functional.softmax(torch.matmul(Q[:, 0], Kt0) / self.sdk, dim=2)  # [N, Nh, n]
        Mt1 = torch.nn.functional.softmax(torch.matmul(Q[:, 1], Kt1) / self.sdk, dim=2)  # [N, Nh, 3*n]

        Ut0 = torch.matmul(Mt0, V[:, 0]).view(N, self.Nh * self.Ns)
        Ut1 = torch.matmul(Mt1.unsqueeze(1), Vt1).view(N, 3, self.Nh * self.Ns)

        ft0_h = self.t0t1emb.forward(Ut0)
        ft1_h = self.t2t2emb.forward(Ut1)

        ft0_u = ft0 + ft0_h
        ft1_u = ft1 + ft1_h

        return ft0_u, ft1_u


def state_max_pool(ft0, ft1, M):
    """Run state_max_pool method."""
    # code.
    s = torch.norm(ft1, dim=2)  # [N, S]
    ft0_max, _ = torch.max(M.unsqueeze(2) * ft0.unsqueeze(1), dim=0)  # [n, S]
    _, s_ids = torch.max(M.unsqueeze(2) * s.unsqueeze(1), dim=0)  # [n, S]
    ft1_max = torch.gather(ft1, 0, s_ids.unsqueeze(2).repeat((1, 1, ft1.shape[2])))

    return ft0_max, ft1_max


class ResLevelPool(torch.nn.Module):
    """Define Class ResLevelPool."""

    def __init__(self, N0, N1, Nh):
        """Run __init__ method."""
        # code.
        super(ResLevelPool, self).__init__()
        self.faemb = torch.nn.Sequential(
            torch.nn.Linear(2 * N0, N0),
            torch.nn.ELU(),
            torch.nn.Linear(N0, N0),
            torch.nn.ELU(),
            torch.nn.Linear(N0, 2 * Nh),
        )

        self.resatde = torch.nn.Sequential(
            torch.nn.Linear(Nh * N0, N0),
            torch.nn.ELU(),
            torch.nn.Linear(N0, N0),
            torch.nn.ELU(),
            torch.nn.Linear(N0, N1),
        )

        self.resatdevec = torch.nn.Sequential(
            torch.nn.Linear(Nh * N0, N1, bias=False)
        )

    def forward(self, ft0, ft1, M):
        """Run forward method."""
        # code.
        F = (1.0 - M + 1e-6) / (M - 1e-6)

        z = torch.cat([ft0, torch.norm(ft1, dim=1)], dim=1)

        Ms = torch.nn.functional.softmax(self.faemb.forward(z).unsqueeze(1) + F.unsqueeze(2), dim=0).view(M.shape[0], M.shape[1], -1, 2)

        ft0_dot = torch.matmul(torch.transpose(ft0, 0, 1), torch.transpose(Ms[:, :, :, 0], 0, 1))
        ft1_dot = torch.matmul(torch.transpose(torch.transpose(ft1, 0, 2), 0, 1), torch.transpose(Ms[:, :, :, 1], 0, 1).unsqueeze(1))

        ft0_r = self.resatde.forward(ft0_dot.view(Ms.shape[1], -1))
        ft1_r = self.resatdevec.forward(ft1_dot.view(Ms.shape[1], ft1.shape[1], -1))

        return ft0_r, ft1_r


class HieraAttLayer(torch.nn.Module):
    """Define Class HieraAttLayer."""

    def __init__(self, layer_params):
        """Run __init__ method."""
        # code.
        super(HieraAttLayer, self).__init__()

        self.hieraupt = HieraAttUpdate(*[layer_params[k] for k in ['Ns', 'Nh', 'Nk']])

        self.m_nn = torch.arange(layer_params['nn'], dtype=torch.int64)

    def forward(self, Z):
        """Run forward method."""
        # code.
        ft0, ft1, ids_topk, D_topk, R_topk = Z

        ids_nn = ids_topk[:, self.m_nn]

        ft0 = ft0.requires_grad_()
        ft1 = ft1.requires_grad_()
        ft0, ft1 = checkpoint(self.hieraupt.forward, ft0, ft1, ft0[ids_nn], ft1[ids_nn], D_topk[:, self.m_nn], R_topk[:, self.m_nn])

        ft0[0] = ft0[0] * 0.0
        ft1[0] = ft1[0] * 0.0

        return ft0, ft1, ids_topk, D_topk, R_topk


def get_features(X, ids_topk, fv):
    """Run get_features method."""
    # code.
    R_nn = X[ids_topk - 1] - X.unsqueeze(1)
    D_nn = torch.norm(R_nn, dim=2)
    D_nn = D_nn + torch.max(D_nn) * (D_nn < 1e-2).float()
    R_nn = R_nn / D_nn.unsqueeze(2)

    fv = torch.cat([torch.zeros((1, fv.shape[1]), device=fv.device), fv], dim=0)
    ids_topk = torch.cat([torch.zeros((1, ids_topk.shape[1]), dtype=torch.long, device=ids_topk.device), ids_topk], dim=0)
    D_nn = torch.cat([torch.zeros((1, D_nn.shape[1]), device=D_nn.device), D_nn], dim=0)
    R_nn = torch.cat([torch.zeros((1, R_nn.shape[1], R_nn.shape[2]), device=R_nn.device), R_nn], dim=0)

    return fv, ids_topk, D_nn, R_nn
