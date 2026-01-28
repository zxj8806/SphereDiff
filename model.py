import torch
import torch.nn as nn
import torch.nn.functional as F

from cluster_utils import (
    Clustering_Metrics,
    GraphConvSparse,
    ClusterAssignment,
)
from vmfmix.vmf import VMFMixture
from vmfmix.von_mises_fisher import VonMisesFisher, HypersphericalUniform
from torch.distributions.kl import kl_divergence


def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=2e-2, device=None):
    return torch.linspace(beta_start, beta_end, timesteps, device=device)

def compute_diffusion_params(timesteps: int, device):
    betas = linear_beta_schedule(timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_1m = torch.sqrt(1.0 - alphas_cumprod)
    rev_cum = torch.flip(sqrt_1m, dims=[0]).cumsum(0)
    cum_sqrt_1m = torch.flip(rev_cum, dims=[0])
    return sqrt_1m, cum_sqrt_1m


class SphereDiff(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_neurons = kwargs["num_neurons"]
        self.num_features = kwargs["num_features"]
        self.embedding_size = kwargs["embedding_size"]
        self.nClusters = kwargs["nClusters"]
        self.T = kwargs.get("T", 10)

        act = {"ReLU": F.relu, "Sigmoid": F.sigmoid, "Tanh": F.tanh}.get(
            kwargs.get("activation", "ReLU"), F.relu
        )
        self.activation = act

        self.kl_hyp_weight = kwargs.get("kl_hyp_weight", 0.1)
        init_kappa = kwargs.get("init_kappa", 10.0)
        self.align_weight = kwargs.get("align_weight", 0.1)
        self.align_alpha = kwargs.get("align_alpha", 2)
        self.align_num_neg = kwargs.get("align_num_neg", 1)
        self.align_margin = kwargs.get("align_margin", 1.0)
        self.cluster_reg_weight = kwargs.get("cluster_reg_weight", 0.1)
        self.entropy_reg_weight = kwargs.get("entropy_reg_weight", 2e-3)

        self.log_kappa = nn.Parameter(torch.tensor(init_kappa))
        self.z_weight = nn.Parameter(torch.zeros(self.T))  # time weights

        self.base_gcn = GraphConvSparse(self.num_features, self.num_neurons, self.activation)
        self.gcn_mean = GraphConvSparse(self.num_neurons, self.embedding_size, lambda x: x)
        self.gcn_logsigma2 = GraphConvSparse(self.num_neurons, self.embedding_size, lambda x: x)

        self.assignment = ClusterAssignment(self.nClusters, self.embedding_size, kwargs["alpha"])

    def kappa(self):
        return F.softplus(self.log_kappa)

    def encode_once(self, x, adj):
        h = self.base_gcn(x, adj)
        mu = self.gcn_mean(h, adj)
        logs2 = self.gcn_logsigma2(h, adj)
        return mu, logs2

    def decode_diffusion_pairs_stream(self, mu, logs2, row_idx, col_idx, start=1, micro_bs: int = 2000):

        T = self.T
        device = mu.device
        dtype = mu.dtype

        sqrt_1m, cum = compute_diffusion_params(T, device)
        norm = cum[start - 1]
        w_t = F.softmax(self.z_weight[:T], 0)

        B = row_idx.numel()
        out = torch.empty(B, device=device, dtype=dtype)

        d = mu.size(1)

        for s in range(0, B, micro_bs):
            e = min(s + micro_bs, B)
            rr = row_idx[s:e]
            cc = col_idx[s:e]

            mu_r = mu[rr]
            mu_c = mu[cc]
            ls2_r = logs2[rr]
            ls2_c = logs2[cc]

            acc = None
            for tau in range(start, T + 1):
                eps_r = torch.randn((rr.numel(), d), device=device, dtype=dtype)
                eps_c = torch.randn((cc.numel(), d), device=device, dtype=dtype)

                z_r = mu_r + eps_r * torch.exp(0.5 * ls2_r)
                z_c = mu_c + eps_c * torch.exp(0.5 * ls2_c)

                z_r = F.normalize(z_r, 2, 1)
                z_c = F.normalize(z_c, 2, 1)

                sim = (z_r * z_c).sum(dim=1)  # [b]
                w = sqrt_1m[tau - 1] * w_t[tau - 1]
                acc = sim * w if acc is None else acc + sim * w

            out[s:e] = torch.clamp(acc / norm, 0, 1)

        return out


    @staticmethod
    def align_loss_pairs(z, row_pos, col_pos, row_neg, col_neg, alpha=2, margin=1.0):
        dp = z[row_pos] - z[col_pos]
        pos = dp.norm(2, 1).pow(alpha).mean()
        dn = z[row_neg] - z[col_neg]
        neg = F.relu(margin - dn.norm(2, 1)).pow(alpha).mean()
        return pos + neg

    def train(
        self,
        features,
        adj_norm,
        adj_label,
        y,
        norm,
        optimizer="Adam",
        epochs=1000,
        lr=5e-3,
        kappa_lr=1e-3,
        save_path="./results/",
        dataset="Cora",
        run_id: str = None,
        pos_per_step: int = 50_000,
        neg_ratio: float = 1.0,
        steps_per_epoch: int = 4,
        pair_micro_bs: int = 10_000
    ):
        import os, time, json
        import numpy as np

        os.makedirs(save_path, exist_ok=True)
        base_dir = os.path.join(save_path, dataset)
        os.makedirs(base_dir, exist_ok=True)

        if run_id is None:
            run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            need_resume = False
        else:
            need_resume = os.path.isfile(os.path.join(base_dir, run_id, "epoch0.ckpt"))
            
        run_dir = os.path.join(base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        base_params = [p for n, p in self.named_parameters() if n != "log_kappa"]
        optim_cls = torch.optim.Adam if optimizer == "Adam" else torch.optim.SGD
        opt = optim_cls(
            [{"params": base_params, "lr": lr}, {"params": [self.log_kappa], "lr": kappa_lr}],
            **({"momentum": 0.9} if optimizer == "SGD" else {}),
        )

        device = getattr(features, "device", next(self.parameters()).device)
        b_acc = 0.0

        if adj_label.is_sparse:
            row_all, col_all = adj_label.coalesce().indices()
        else:
            row_all, col_all = adj_label.nonzero(as_tuple=True)
        mask_offdiag = row_all != col_all
        row_all = row_all[mask_offdiag]
        col_all = col_all[mask_offdiag]
        undup_mask = row_all < col_all
        pos_row_full = row_all[undup_mask]
        pos_col_full = col_all[undup_mask]
        num_pos_total = pos_row_full.numel()
        N = adj_label.size(0)

        vmf = VMFMixture(n_cluster=self.nClusters, max_iter=100)
        bce = nn.BCELoss(reduction='mean')

        for ep in range(epochs):
            mu, logs2 = self.encode_once(features, adj_norm)
            z = mu
            z_u = F.normalize(z, 2, 1)

            kappa_b = torch.full((z_u.size(0), 1), self.kappa().item(), device=z.device)
            qz = VonMisesFisher(z_u, kappa_b)
            pz = HypersphericalUniform(z_u.size(1) - 1)
            loss_kl = kl_divergence(qz, pz).mean()

            p = self.assignment(z_u)
            centers = F.normalize(self.assignment.cluster_centers, 2, 1)
            intra = ((p[:, :, None] * (z_u[:, None, :] - centers[None, :, :]).pow(2))).sum() / z_u.size(0)
            inter = torch.pdist(centers, 2).mean()
            loss_clu = intra / (inter + 1e-9)
            loss_ent = (p * torch.log(p + 1e-9)).sum() / p.size(0)

            loss_rec_total = 0.0
            loss_aln_total = 0.0
            steps = max(1, steps_per_epoch)

            for _ in range(steps):
                k = min(pos_per_step, num_pos_total)
                idx = torch.randint(0, num_pos_total, (k,), device=device)
                pr = pos_row_full[idx]
                pc = pos_col_full[idx]

                num_neg = int(k * neg_ratio)
                nr = torch.randint(0, N, (num_neg,), device=device)
                nc = torch.randint(0, N, (num_neg,), device=device)
                mask = nr != nc
                nr = nr[mask][:num_neg]
                nc = nc[mask][:num_neg]

                pos_score = self.decode_diffusion_pairs_stream(mu, logs2, pr, pc, start=1, micro_bs=pair_micro_bs)
                neg_score = self.decode_diffusion_pairs_stream(mu, logs2, nr, nc, start=1, micro_bs=pair_micro_bs)

                loss_rec = bce(pos_score, torch.ones_like(pos_score)) + bce(neg_score, torch.zeros_like(neg_score))
                loss_rec_total += loss_rec

                loss_aln = self.align_loss_pairs(z_u, pr, pc, nr, nc,
                                                 alpha=self.align_alpha, margin=self.align_margin)
                loss_aln_total += loss_aln

            loss_rec_avg = loss_rec_total / steps
            loss_aln_avg = loss_aln_total / steps

            loss = (
                loss_rec_avg
                + self.align_weight * loss_aln_avg
                + self.cluster_reg_weight * loss_clu
                + self.entropy_reg_weight * loss_ent
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                vmf.fit(z_u.detach().cpu().numpy())
                y_pred = vmf.labels_ if hasattr(vmf, "labels_") else vmf.predict(z_u.detach().cpu().numpy())

            print(
                f"[Epoch {ep+1}] "
                f"loss={loss.item():.4f} rec={loss_rec_avg.item():.4f} "
                f"rec={loss_rec_avg.item():.4f} "
                f"aln={loss_aln_avg.item():.4f} "
                f"clu={loss_clu.item():.4f} ent={loss_ent.item():.4f} "
            )

        return y_pred, y
