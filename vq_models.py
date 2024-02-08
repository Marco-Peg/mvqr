import json
import os
from functools import partial
from abc import abstractmethod
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from typing import Union, Sequence
from torch import nn

from rcpm.densities import DensityDataset
from rcpm.manifolds import Product
import rcpm.densities as densities


def init_uniform(minval, maxval, shape, dtype=torch.float32):
    return torch.empty(shape, dtype=dtype).uniform_(minval, maxval)

NLS = {
    "relu": torch.nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "silu": nn.SiLU,
    "lrelu": nn.LeakyReLU,
}

class MLP(nn.Module):
    """
    A Simple MLP.

    Structure is:
    FC -> [BN] -> ACT -> [DROPOUT] -> ... FC -> [BN] -> ACT -> [DROPOUT] -> FC

    Note that BatchNorm and Dropout are optional and the MLP always ends with an FC.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Union[str, Sequence[int]],
        nl: Union[str, nn.Module] = "silu",
        skip: bool = False,
        batchnorm: bool = False,
        dropout: float = 0,
    ):
        """
        :param in_dim: Input feature dimension.
        :param hidden_dims: Hidden dimensions. Will be converted to FC layers. Last
        entry is the output dimension. If a string is provided, it will be parsed as
        a comma-separated list of integer values, e.g. '12,34,46,7'.
        :param nl: Non-linearity to use between FC layers.
        :param skip: Whether to use a skip-connection (over all layers). This
        requires that in_dim==out_dim, thus if skip==True and hidden_dims[-1]!=in_dim
        then the last hidden layer will be changed to produce an output of size in_dim.
        :param batchnorm: Whether to use Batch Normalization
        (before each non-linearity).
        :param dropout: Whether to use dropout (after each non-linearity).
        Zero means no dropout, otherwise means dropout probability.
        """
        super().__init__()

        if isinstance(hidden_dims, str):
            try:
                hidden_dims = tuple(map(int, hidden_dims.strip(", ").split(",")))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "When hidden_dims is a string it must be a comma-separated "
                    "sequence of integers, e.g. '11,22,34,45'"
                ) from e

        if not hidden_dims:
            raise ValueError(f"got {hidden_dims=} but must have at least one")

        if isinstance(nl, nn.Module):
            non_linearity = nl
        else:
            if nl not in NLS:
                raise ValueError(f"got {nl=} but must be one of {[*NLS.keys()]}")
            non_linearity = NLS[nl]

        if not 0 <= dropout < 1:
            raise ValueError(f"got {dropout=} but must be in [0, 1)")

        # Split output dimension from the hidden dimensions
        *hidden_dims, out_dim = hidden_dims
        if skip and out_dim != in_dim:
            out_dim = in_dim

        layers = []
        fc_dims = [in_dim, *hidden_dims]
        for d1, d2 in zip(fc_dims[:-1], fc_dims[1:]):
            layers.append(nn.Linear(d1, d2, bias=True))
            if batchnorm:
                layers.append(nn.BatchNorm1d(num_features=d2))
            layers.append(non_linearity())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        # Always end with FC
        layers.append(nn.Linear(fc_dims[-1], out_dim, bias=True))

        self.fc_layers = nn.Sequential(*layers)
        self.skip = skip

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        z = self.fc_layers(x)

        if self.skip:
            z += x

        return z

class CConvexABC(nn.Module):

    def __init__(self, manifold, cost_gamma,
                 min_zero_gamma, device=torch.device("cpu"),
                 th_dtype=torch.float32, keepdim=False):
        super().__init__()
        self.manifold = manifold
        self.device = device
        self.th_dtype = th_dtype
        # soft min params
        self.cost_gamma = cost_gamma
        self.min_zero_gamma = min_zero_gamma
        self.keepdim = keepdim

    @abstractmethod
    def forward(self, U_batch):
        raise NotImplementedError


class CConvex(CConvexABC):

    def __init__(self, manifold, n_components,
                 init_alpha_mode, init_alpha_linear_scale,
                 init_alpha_minval, init_alpha_range, cost_gamma,
                 min_zero_gamma,
                 init_points="grid", fixed_points=False,
                 device=torch.device("cpu"),
                 th_dtype=torch.float32, alpha_dim=1, keepdim=False):
        super().__init__(manifold, cost_gamma,
                         min_zero_gamma, device, th_dtype, keepdim)
        self.n_components = n_components
        self.init_alpha_mode = init_alpha_mode
        self.init_alpha_linear_scale = init_alpha_linear_scale
        self.init_alpha_minval = init_alpha_minval
        self.init_alpha_range = init_alpha_range
        self.alpha_dim = alpha_dim
        self.fixed_points = fixed_points

        if init_points == "uniform":
            init_distr = densities.get_uniform(self.manifold)
            self.sample_points = init_distr.sample(self.n_components)
            # self.sample_points = self.manifold.projx(self.sample_points.T).T
        elif init_points == "grid":
            self.sample_points = self.manifold.grid(self.n_components)
        else:
            raise ValueError(f"init_points={init_points} not recognized")

        self.sample_points = self.sample_points.to(
            self.th_dtype).to(self.device).clone().detach()
        self.sample_points = nn.Parameter(self.sample_points)
        if fixed_points:
            self.sample_points.requires_grad_(False)

        # values of the function on sampled points
        if self.init_alpha_mode == 'linear':
            alphas = self.init_alpha_linear_scale * self.mus[:, 0].dot(self.mus)
            self.alphas = nn.Parameter(
                torch.Tensor(alphas).repeat(1, self.alpha_dim).to(self.th_dtype).to(
                    self.device))
        elif self.init_alpha_mode == 'uniform':
            self.alphas = nn.Parameter(torch.Tensor(
                np.random.uniform(
                    low=self.init_alpha_minval,
                    high=self.init_alpha_minval + self.init_alpha_range,
                    size=(self.n_components, self.alpha_dim))).to(self.th_dtype).to(
                self.device))
        elif self.init_alpha_mode == "constant":
            self.alphas = nn.Parameter(torch.Tensor(
                np.ones((self.n_components, self.alpha_dim))).to(self.th_dtype).to(
                self.device))

        else:
            assert False

    def post_hook(self, optimizer, args, kwargs):

        if not self.fixed_points:
            # project points on the manifold
            with torch.no_grad():
                # ensure points lays on the manifold
                new_val = self.manifold.projx(self.sample_points)
                self.sample_points.copy_(new_val)
    
    def get_points_params(self):
        if self.fixed_points:
            return []
        return [self.sample_points,]

    def forward(self, U_batch, training=False, vmap=False, gamma=None):
        # \Phi(x) = - min_{i}(c(x,y_i) + \Psi(y_i))
        batched = U_batch.dim() == 1 or vmap
        if batched:
            U_batch = U_batch.unsqueeze(0)
        assert U_batch.shape[1] == self.manifold.D

        costs = self.manifold.cost(U_batch, self.sample_points).unsqueeze(-1) - \
                self.alphas
                
        if gamma is None:
            gamma = self.cost_gamma

        if self.cost_gamma is not None and self.cost_gamma > 0.:
            F = gamma * torch.logsumexp(
                -costs / gamma, dim=1)
        else:
            F = - torch.min(costs, dim=1).values

        if self.min_zero_gamma is not None and self.min_zero_gamma > 0.:
            Fz = torch.stack((F, torch.zeros_like(F)), dim=-1)
            F = self.min_zero_gamma * torch.logsumexp(
                -Fz / self.min_zero_gamma, dim=-1)
        if F.shape[1] == 1 and self.keepdim == False:
            F = F.squeeze(1)
        if batched:
            F = F.squeeze(0)
        return F


class CConvexStack(nn.Module):

    def __init__(self, manifold, n_cc, n_components, init_alpha_mode,
                 init_alpha_linear_scale,
                 init_alpha_minval, init_alpha_range, cost_gamma, min_zero_gamma,
                 fixed_points=False, init_points="grid",
                 device=torch.device("cpu"), th_dtype=torch.float32):
        super().__init__()
        self.n_cc = n_cc
        self.cc_stack = list()
        for i_n in range(n_cc):
            if fixed_points:
                n_comp = n_components
            else:
                n_comp = int(n_components / n_cc)
            self.cc_stack.append(CConvex(manifold, n_comp,
                                         init_alpha_mode,
                                         init_alpha_linear_scale,
                                         init_alpha_minval, init_alpha_range,
                                         cost_gamma, min_zero_gamma,
                                         fixed_points=fixed_points,
                                         init_points=init_points,
                                         device=device, th_dtype=th_dtype,
                                         alpha_dim=1,
                                         keepdim=True))
            self.add_module(f"cc_{i_n}", self.cc_stack[-1])

    def post_hook(self, optimizer, args, kwargs):
        for cc in self.cc_stack:
            cc.post_hook(optimizer, args, kwargs)
    
    def get_points_params(self):
        params = []
        for cc in self.cc_stack:
            params += cc.get_points_params()
        return params

    def forward(self, U_batch):
        p_list = [cc_f(U_batch) for cc_f in self.cc_stack]
        p = torch.cat(p_list, 1)

        return p

class CConvexStackMulti(nn.Module):

    def __init__(self, manifold, n_cc, n_components, init_alpha_mode,
                 init_alpha_linear_scale,
                 init_alpha_minval, init_alpha_range, cost_gamma, min_zero_gamma,
                 fixed_points=False, init_points="grid", n_layers=2,
                 device=torch.device("cpu"), th_dtype=torch.float32):
        super().__init__()
        self.n_cc = n_cc
        self.cc_stack = list()
        for i_n in range(n_cc):
            self.cc_stack.append(CConvex_Multi(manifold, n_layers, n_components,
                                            init_alpha_mode, init_alpha_linear_scale,
                                            init_alpha_minval, init_alpha_range, cost_gamma,
                                            min_zero_gamma, 
                                         fixed_points=fixed_points,
                                         init_points=init_points,
                                         device=device, th_dtype=th_dtype,
                                         alpha_dim=1,
                                         keepdim=True))
            self.add_module(f"cc_{i_n}", self.cc_stack[-1])
    
    def post_hook(self, optimizer, args, kwargs):
        for cc in self.cc_stack:
            cc.post_hook(optimizer, args, kwargs)
            
    def get_points_params(self):
        params = []
        for cc in self.cc_stack:
            params += cc.get_points_params()
        return params

    def forward(self, U_batch):
        p_list = [cc_f(U_batch) for cc_f in self.cc_stack]
        p = torch.cat(p_list, 1)

        return p

class CConvex_Multi(CConvexABC):

    def __init__(self, manifold, n_layers, n_components,
                 init_alpha_mode, init_alpha_linear_scale,
                 init_alpha_minval, init_alpha_range, cost_gamma,
                 min_zero_gamma, init_points="uniform", fixed_points=False,
                 alpha_dim=1,
                 device=torch.device("cpu"), th_dtype=torch.float32, keepdim=False):
        super().__init__(manifold, cost_gamma,
                         min_zero_gamma, device, th_dtype, keepdim)
        self.n_layers = n_layers
        self.n_components = n_components
        # alpha params
        self.init_alpha_minval = init_alpha_minval
        self.init_alpha_range = init_alpha_range
        self.alpha_dim = alpha_dim

        self.cconvx0 = CConvex(manifold, n_components,
                                                  init_alpha_mode,
                                                  init_alpha_linear_scale,
                                                  init_alpha_minval, init_alpha_range,
                                                  cost_gamma,
                                                  min_zero_gamma,
                                                  init_points=init_points,
                                                  fixed_points=fixed_points,
                                                  device=device, th_dtype=th_dtype,
                                                  alpha_dim=alpha_dim, keepdim=keepdim)
        if n_layers > 1:
            self.cconvx_list = nn.ModuleList([CConvex(manifold, n_components,
                                                    init_alpha_mode,
                                                    init_alpha_linear_scale,
                                                    init_alpha_minval, init_alpha_range,
                                                    cost_gamma,
                                                    min_zero_gamma,
                                                    init_points=init_points,
                                                    fixed_points=fixed_points,
                                                    device=device, th_dtype=th_dtype,
                                                    alpha_dim=alpha_dim, keepdim=keepdim)
                                            for i in range(n_layers-1)])

            self.w_list = nn.ParameterList(
                [nn.Parameter(init_uniform(minval=0., maxval=1., shape=alpha_dim,
                                        dtype=self.th_dtype).to(self.device))
                for i in range(n_layers-1)])

            self.conv_activation = nn.ReLU()
            self.w_activation = nn.Sigmoid()

    def forward(self, xs, gamma=None):
        single = xs.ndim == 1
        if single:
            xs = xs.unsqueeze(0)

        F = self.cconvx0(xs, gamma=gamma)
        for i, (cconvx, w) in enumerate(zip(self.cconvx_list, self.w_list)):
            w_pos = self.w_activation(w)
            phi = cconvx(xs, gamma=gamma)
            F = w_pos * self.conv_activation(F) + (1 - w_pos) * phi

        if single:
            F = F.squeeze(0)
        return F
    
    def get_points_params(self):
        p_params = self.cconvx0.get_points_params()
        for cc in self.cconvx_list:
            p_params += cc.get_points_params()
        return p_params

    def post_hook(self, optimizer, args, kwargs):
        self.cconvx0.post_hook(optimizer, args, kwargs)
        for cc in self.cconvx_list:
            cc.post_hook(optimizer, args, kwargs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def plot_trainloss(losses, losses_cc=None, losses_kde=None, results_dir=None):
    f, ax = plt.subplots()
    color = 'tab:blue'
    ax.set_ylabel('train loss', color=color)
    ax.plot(losses.keys(), losses.values(), label="train loss", color=color)
    ax.tick_params(axis='y', labelcolor=color)

    if losses_cc is not None:
        ax1 = ax.twinx()
        color = 'tab:green'
        ax1.set_ylabel('cc loss', color=color)  # we already handled the x-label with ax1
        ax1.plot(losses_cc.keys(), losses_cc.values(), label="cc loss", color=color)
        ax1.tick_params(axis='y', labelcolor=color)
    if losses_kde is not None:
        ax2 = ax.twinx()
        if losses_cc is not None:
            ax2.spines["right"].set_position(("axes", 1.2))
        color = 'tab:red'
        ax2.set_ylabel('kde loss', color=color)  # we already handled the x-label with
        ax2.plot(losses_kde.keys(), losses_kde.values(), label="kde loss", color=color)
        ax2.tick_params(axis='y', labelcolor=color)
    
    ax.set_xlabel("iteration")
    ax.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(results_dir, 'train_loss.png'))
    # save dicts in json
    with open(os.path.join(results_dir, 'train_loss.json'), 'w') as fp:
        json.dump(losses, fp)
    if losses_cc is not None:
        with open(os.path.join(results_dir, 'train_loss_cc.json'), 'w') as fp:
            json.dump(losses_cc, fp)
    if losses_kde is not None:
        with open(os.path.join(results_dir, 'train_loss_kde.json'), 'w') as fp:
            json.dump(losses_kde, fp)
    return f, ax


class ManifoldVQE(nn.Module):
    def __init__(self, manifold, eps=1e-3, device=torch.device("cpu"),
                 th_dtype=torch.float32, base_density='uniform'):
        super().__init__()
        self.device = device
        self.th_dtype = th_dtype
        self.manifold = manifold
        self.eps = eps

        if isinstance(base_density, densities.Density):
            self.base_distr = base_density
        elif base_density == "uniform":
            if isinstance(self.manifold, Product):
                self.base_distr = densities.get(self.manifold,
                                                'ProductUniformComponents')
            else:
                self.base_distr = densities.get_uniform(self.manifold)
        else:
            self.base_distr = densities.get(self.manifold, base_density)

    def check_cmonotone(self, batch_size=50, max_iters=1e4):
        import itertools
        U_sampled = self.base_distr.sample(batch_size).to(self.th_dtype).to(self.device)
        target_samples = self.forward(U_sampled).detach()

        # compute costs
        c_matrix = self.manifold.cost(U_sampled, target_samples)
        c0 = torch.diag(c_matrix).sum()
        for i_p in range(1, U_sampled.shape[0]):
            cp = torch.diag(torch.roll(c_matrix, i_p, 0)).sum()
            if c0 > cp:
                return False
            if max_iters < i_p:
                break
        max_iters -= i_p
        for i_p, perm in enumerate(itertools.permutations(range(U_sampled.shape[0]))):
            if i_p == 0: continue
            cp = c_matrix[perm, range(U_sampled.shape[0])].sum()
            if c0 > cp:
                return False
            if max_iters < i_p:
                break
        return True

    def evaluate_full_batch(self, Y_train, U_batch=None, U_conj=None, cc_loss=True,
                            c_batch=5000):
        """
        Gets a random sample of U, and evaluates loss function using X and Y.
        """
        batch_size = Y_train.shape[0]
        if U_batch is None:
            base_distribution = self.base_distr
            with torch.no_grad():
                U_batch = base_distribution.sample(batch_size).to(self.th_dtype).to(
                    self.device)
        
        # compute forward potential
        if c_batch > 0 and c_batch < U_batch.shape[0]: 
            phi = list()
            for i in range(0, U_batch.shape[0], c_batch):
                phi.append(self.compute_phi(U_batch[i:i+c_batch]))
            phi = torch.cat(phi)
        else:     
            phi = self.compute_phi(U_batch)  # convex potential
        # compute conjugate
        if c_batch > 0 and c_batch < batch_size: 
            UY = list()
            psi = list()
            for i in range(0, batch_size, c_batch):
                UY_batch = self.manifold.cost(U_batch, Y_train[i:i+c_batch]) 
                UY.append(UY_batch)
                psi.append(self.conjugate(phi=phi if U_conj is None else None, 
                             cost_matrix=UY_batch if U_conj is None else None,
                                        U_batch=U_conj))
            UY = torch.cat(UY, dim=1)
            psi = torch.cat(psi)
        else:     
            UY = self.manifold.cost(U_batch, Y_train)
            psi = self.conjugate(phi=phi if U_conj is None else None, 
                             cost_matrix=UY if U_conj is None else None,
                                        U_batch=U_conj)  # conjugate convex potential
        objective = psi.mean() + phi.mean()
        # conjugate loss
        if cc_loss:
            phi_cc = self.conjugate(phi=psi, cost_matrix=UY.T)
            cc_loss = (phi_cc - phi) ** 2
            cc_loss = torch.nn.functional.normalize(cc_loss, p=2, dim=0)
            return objective.squeeze(), cc_loss.mean()

        return objective.squeeze()

    def compute_phi(self, U_batch, vmap=False, gamma=None):
        """
            Compute the forward potential function (Phi)
        """
        batched = U_batch.dim() == 1 or vmap
        if batched:
            U_batch = U_batch.unsqueeze(0)
        assert U_batch.shape[1] == self.manifold.D

        phi_eval = self.c_convex(U_batch, gamma=gamma).unsqueeze(1)  # [bu,1]
        if vmap:
            phi_eval = phi_eval.squeeze(1)
        if batched:
            phi_eval = phi_eval.squeeze(0)
        return phi_eval

    def conjugate(self, Y_batch=None, U_batch=None, phi=None, cost_matrix=None, eps=None,
                  n_base=1000, vmap=False):
        """
            Compute the conjugate (Psi) of the forward potential function
            Y_batch: batch of points on the manifold, torch.tensor of shape (by, D)
            U_batch: batch of points on the manifold, torch.tensor of shape (bu, D)
            phi: convex forward potential function, torch.tensor of shape (bu, )
            cost_matrix: cost matrix, torch.tensor of shape (bu, by)
            n_base: number of samples to use for the base distribution
            vmap: whether to use vmap
        """

        if U_batch is None and (phi is None or cost_matrix is None):
            U_batch = self.base_distr.sample(n_base).to(self.th_dtype).to(self.device)

        if phi is None:
            phi = self.compute_phi(U_batch)  # [bu,1]

        if cost_matrix is None:
            if vmap:
                Y_batch = Y_batch.unsqueeze(0)
            if len(Y_batch.shape) == 1:
                Y_batch = Y_batch.reshape(-1, 1)
            UY = self.manifold.cost(U_batch, Y_batch)
        else:
            UY = cost_matrix

        ## min version (- concave)
        phi_cc = UY + phi
        if eps is None:
            eps = self.eps
        if eps > 0: # and not self.training:
            phi_cc = eps * torch.logsumexp(-phi_cc / eps, dim=0, keepdim=True)
        else:
            phi_cc = - phi_cc.min(dim=0, keepdim=True).values

        phi_cc = phi_cc.T

        if vmap:
            phi_cc = phi_cc.squeeze()
        return phi_cc

    def post_hook(self, optimizer, args, kwargs):
        pass
    
    def train(self, mode: bool = True):
        return super().train(mode)
    
    def gradient_manifold(self, x, grad, lr, **kwargs):
        dF = self.manifold.tangent_projection(x, grad)
        updated_x = self.manifold.exponential_map(x, lr * dF)
        return updated_x
        
    
    def train_loop(self, target_ditribution, batch_size, results_dir="",
                   lr=1e-3, num_iters=20000, intermediate_plot=False,
                   save_dir="", n_u=int(1e5), cc_weight=1e1,
                   c_batch=0,
                   best=False, max_tollerance=5, identity_iter=0,
                   early_stopping=False, l1_param_weight=0.):
        """
        Y_train: samples from target distribution
        """
        self.train()

        weight_loss = 1e-4

        losses = dict()
        losses_cc = dict() 
        losses_kde = dict()
        Y_train_th = torch.tensor(target_ditribution.sample(batch_size),
                                  dtype=self.th_dtype,
                                  device=self.device)

        
        best_state_dict = None
        best_loss = None
        tollerance = 0
        kde_best = None
        # get the parameters that are not point params
        p_params = self.get_points_params()

        if identity_iter > 0:
            tollerance_id = 0
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)
            self.optimizer.register_step_post_hook(self.post_hook)
            print("### Identity pretraining ###")
            for i in range(identity_iter):
                U_batch = self.base_distr.sample(n_u).to(self.th_dtype).to(
                    self.device)
                self.optimizer.zero_grad()
                loss_fn, cc_loss = self.evaluate_full_batch(U_batch, U_batch=U_batch, 
                                                            c_batch=c_batch)
                if cc_weight > 0:
                    loss_fn = loss_fn + cc_weight * cc_loss
                loss_fn.backward()
                self.optimizer.step()
                if loss_fn.item()< 1e-6:
                    tollerance_id += 1
                    if tollerance_id > 3:
                        break
                else:
                    tollerance_id = 0

                if i % 50 == 0:
                    print(f"\033[1m Iteration {i}: \033[0m {loss_fn.item():.4f}")
                    losses[i - identity_iter] = loss_fn.item()
                    losses_cc[i - identity_iter] = cc_loss.item()
                    
            print("### End Identity pretraining ###")

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.optimizer.register_step_post_hook(self.post_hook)

        for i in range(num_iters):
            self.optimizer.zero_grad()
            U_batch = self.base_distr.sample(n_u).to(self.th_dtype).to(
                self.device)
            loss_fn, cc_loss = self.evaluate_full_batch(Y_train_th, U_batch=U_batch,
                                                        c_batch=c_batch)
            if cc_weight > 0:
                loss_fn = loss_fn + cc_weight * cc_loss
            loss_fn.backward()
            # nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            with torch.no_grad():
                for param in p_params:
                        param.data = self.gradient_manifold(param.data, param.grad, 1e-5)
                        param.grad = None
            self.optimizer.step()
            
            
            if i % 50 == 0:
                print(f"\033[1m Iteration {i}: \033[0m {loss_fn.item():.2e}; scheduler lr: {get_lr(self.optimizer):.2e}")
                losses[i] = loss_fn.item()
                losses_cc[i] = cc_loss.item()
                
                

            if i % 500 == 0:
                self.eval()
                with torch.no_grad():
                    if c_batch > 0 and c_batch < batch_size:
                        n_plot = c_batch
                    else:
                        n_plot = batch_size
                    samples_est_random = self.sample(n_plot).to(self.th_dtype)
                    kde_loss = self.manifold.kde_loss(samples_est_random, Y_train_th)
                    losses_kde[i] = kde_loss.item()
                print(f"\033[94m KDE loss: \033[0m {kde_loss.item():.2e}")
                if kde_best is None or (kde_best) > (kde_loss.item()):
                    kde_best = kde_loss.item()
                    # if (kde_best + weight_loss * best_loss) >= (
                    #         kde_loss.item() + weight_loss * loss_fn.item()):
                    tollerance = 0
                else:
                    tollerance += 1
                if early_stopping and tollerance > max_tollerance:
                    break
                if intermediate_plot and i % 1500 == 0:
                    with torch.no_grad():
                        inv_samples = self.sample_inverse(Y_train_th,
                                                        n_base=int(
                                                            5e3)).cpu().detach()

                        self.manifold.plot_samples_3d([Y_train_th.detach().cpu(),
                                                    samples_est_random.detach().cpu(),
                                                    inv_samples.detach().cpu()],
                                                    save=os.path.join(results_dir,
                                                                        f'result_{i}'),
                                                    show=False,
                                                    subplots_titles=[ "GT distribution.",
                                                        f"forward", "inv"])
                self.train()
                f,ax = plot_trainloss(losses, losses_cc, losses_kde, results_dir)
                plt.close(f)


            if best_loss is None or best_loss > loss_fn.item():
                best_loss = loss_fn.item()
                if best:
                    best_state_dict = self.state_dict()
            
                                                         
        plot_trainloss(losses, losses_cc, losses_kde, results_dir)
        if not best:
            best_state_dict = self.state_dict()

        return best_state_dict


    def likelihood(self, U_sampled=None, n_base=1000, Y_sampled=None,
                   eps=5e-2, chunk_size=None, log=False):
        if U_sampled is None:
            U_sampled = self.manifold.grid(n_base).to(self.th_dtype)
        if Y_sampled is None:
            with torch.no_grad():
                Y_sampled = self.sample(base_sample=U_sampled)

        inv_pot_fn = lambda ys: self.conjugate(Y_batch=ys, 
                                                    U_batch=U_sampled.to(self.device),
                                                    eps=eps, vmap=True)
        def dF_riemmanian_inv( ys):
            dF = torch.func.grad(inv_pot_fn)(ys)
            dF = self.manifold.tangent_projection(ys, dF).squeeze(0)
            return dF
        
        def Qinv(ys):
            dF = dF_riemmanian_inv(ys)
            samples_ = self.manifold.exponential_map(ys, dF)
            return samples_


        def _H_inv(y):
            J = torch.func.jacfwd(Qinv)(y).squeeze(0)
            dF = partial(dF_riemmanian_inv)(y)
            return dF, J

        
        dFs, Js = [], []
        
        for y_batch_idx in tqdm(range(0, Y_sampled.shape[0], chunk_size)):
            # print(y_batch_idx)
            y_batch = Y_sampled[y_batch_idx:y_batch_idx+chunk_size]
            dF, J = torch.vmap(_H_inv)(
                y_batch.clone().detach().to(self.device).requires_grad_(True)
            )
            dFs.append(dF.detach())
            Js.append(J.detach())
        dF = torch.cat(dFs, dim=0)
        J = torch.cat(Js, dim=0)

        E = self.manifold.tangent_orthonormal_basis(Y_sampled.to(self.device), dF)
        
        JE = torch.bmm(J, E)
        JETJE = torch.einsum('nji,njk->nik', JE, JE)
        print(f"JE: {JE.shape}, JETJE: {JETJE.shape}")
        
    
        sign, lh = torch.slogdet(JETJE)
        lh = lh.cpu() * 0.5 + self.base_distr.log_prob(Y_sampled).cpu()
        if log:
            return lh, Y_sampled
        else:
            return torch.exp(lh), Y_sampled
        
    def likelihood_fw(self, U_sampled=None, n_base=1000,
                   eps=None, chunk_size=None, log=False):
        if U_sampled is None:
            U_sampled = self.manifold.grid(n_base).to(self.th_dtype)

        with torch.no_grad():
                Y_sampled = self.sample(base_sample=U_sampled)
                
        pot_fn = partial(lambda us: self.compute_phi(us, vmap=True, gamma=eps))
        def dF_riemmanian( us):
            dF = torch.func.grad(pot_fn)(us)
            dF = self.manifold.tangent_projection(us, dF).squeeze(0)
            return dF
        
        def Q(us):
            dF = dF_riemmanian(us)
            samples_ = self.manifold.exponential_map(us, dF)
            return samples_


        def _H(u):
            J = torch.func.jacfwd(Q)(u).squeeze(0)
            dF = partial(dF_riemmanian)(u)
            return dF, J
        
        dFs, Js = [], []
        
        for y_batch_idx in tqdm(range(0, U_sampled.shape[0], chunk_size)):
            u_batch = U_sampled[y_batch_idx:y_batch_idx+chunk_size]
            dF, J = torch.vmap(_H)(
                u_batch.clone().detach().to(self.device).requires_grad_(True)
            )
            dFs.append(dF.detach())
            Js.append(J.detach())
        dF = torch.cat(dFs, dim=0)
        J = torch.cat(Js, dim=0)

        E = self.manifold.tangent_orthonormal_basis(U_sampled.to(self.device), dF)
        
        JE = torch.bmm(J, E)
        JETJE = torch.einsum('nji,njk->nik', JE, JE)
        print(f"JE: {JE.shape}, JETJE: {JETJE.shape}")
        
    
        sign, lh = torch.slogdet(JETJE)
        lh = - lh.cpu() * 0.5 + self.base_distr.log_prob(U_sampled).cpu()
        if log:
            return lh, Y_sampled
        else:
            # lh = torch.det(JETJE) 
            return torch.exp(lh), Y_sampled


    def sample(self, N=100, base_sample=None):
        if base_sample is None:
            base_sample = self.base_distr.sample(N)
        U_sampled = base_sample.clone().detach().to(self.device).to(
            self.th_dtype).requires_grad_(True)

        pot_fn = partial(lambda us: self.compute_phi(us, vmap=True))
        dF = self.manifold.grad(pot_fn, U_sampled)  # gradient of the convex potential
        with torch.no_grad():
            samples_ = self.manifold.exponential_map(U_sampled, dF)

        return samples_

    def sample_inverse(self, target_sample, n_base=50000, eps=None):
        Y_sampled = torch.tensor(
            target_sample,
            device=self.device, dtype=self.th_dtype, requires_grad=True
        )

        # with torch.no_grad():
        U_base = self.base_distr.sample(n_base).to(self.th_dtype).to(self.device)

        pot_fn = lambda ys: self.conjugate(Y_batch=ys, U_batch=U_base, eps=eps,
                                           vmap=True)
        dF = self.manifold.grad(pot_fn, Y_sampled)  # gradient of the convex conjugate
        
        samples_ = self.manifold.exponential_map(Y_sampled, dF)

        return samples_


class ManifoldVQESingle(ManifoldVQE):
    def __init__(self, manifold, n_components, init_alpha_mode, init_alpha_linear_scale,
                 init_alpha_minval, init_alpha_range, cost_gamma,
                 min_zero_gamma, eps=1e-3, device=torch.device("cpu"),
                 init_points="uniform", fixed_points=False,
                 th_dtype=torch.float32, base_density='uniform'):
        super().__init__(manifold, eps, device, th_dtype, base_density)
        self.n_components = n_components

        self.c_convex = CConvex(manifold, n_components,
                                init_alpha_mode,
                                init_alpha_linear_scale,
                                init_alpha_minval, init_alpha_range, cost_gamma,
                                min_zero_gamma,
                                init_points, fixed_points, device, th_dtype)

    def post_hook(self, optimizer, args, kwargs):
        self.c_convex.post_hook(optimizer, args, kwargs)

    def plot_cconvex(self, show=False, save=None):
        U = self.manifold.grid(self.manifold.NUM_POINTS ** 2)
        n_points = U.shape[0]
        x = U[:, 0].numpy()
        x = x.reshape(math.ceil(math.sqrt(n_points)), -1)
        y = U[:, 1].numpy()
        y = y.reshape(math.ceil(math.sqrt(n_points)), -1)
        z = U[:, 2].numpy()
        z = z.reshape(math.ceil(math.sqrt(n_points)), -1)
        U = U.to(self.device)
        cconv_values = self.c_convex(U).reshape(n_points).detach().cpu().numpy()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["learned points", "Vector field"],
            specs=[[{'type': 'scene'} for i in range(2)]])
        fig.layout.coloraxis.colorscale = "ice"
        fig.update_layout(coloraxis=dict(colorscale="ice", colorbar=dict(len=0.5, y=0)))
        fig.update_layout(coloraxis2=dict(colorscale="hot", colorbar=dict(len=0.5,
                                                                          y=0.6)))
        fig.layout.title = "C-convex function"
        fig.add_traces([
            go.Surface(x=x, y=y, z=z,
                       # colorscale="Greys",
                       coloraxis="coloraxis",
                       surfacecolor=cconv_values,
                       customdata=cconv_values,
                       hovertemplate="value: %{customdata:.2f} <br> %{x:.2f};%{"
                                     "y:.2f} ;%{z:.2f}",
                       showscale=True),
            go.Scatter3d(x=self.c_convex.sample_points[:, 0].detach().cpu().numpy(),
                         y=self.c_convex.sample_points[:, 1].detach().cpu().numpy(),
                         z=self.c_convex.sample_points[:, 2].detach().cpu().numpy(),
                         mode='markers',
                         customdata=self.c_convex.alphas.squeeze(0).detach().cpu(

                         ).numpy(),
                         hovertemplate="alpha: %{marker.color:.2f} <br> %{x:.2f} ; %{"
                                       "y:.2f} ; %{z:.2f}",
                         marker=go.scatter3d.Marker(
                             color=self.c_convex.alphas.squeeze(
                                 0).detach().cpu().numpy(),
                             size=2.5,
                             coloraxis="coloraxis2",
                             # line={"width": 0.5},
                             showscale=True))
        ], rows=1, cols=1)

        ind_l = np.random.choice(cconv_values.flatten().shape[0], min(2000, U.shape[0]))
        dF = self.dF_riemannian(U[ind_l, :]).detach().cpu().numpy()
        fig.add_traces([
                           go.Surface(x=x, y=y, z=z, coloraxis="coloraxis",
                                      surfacecolor=cconv_values,
                                      showscale=False, showlegend=False, ),
                           go.Scatter3d(
                               x=x.flatten()[ind_l] + dF[:, 0],
                               y=y.flatten()[ind_l] + dF[:, 1],
                               z=z.flatten()[ind_l] + dF[:, 2],
                               showlegend=False, mode='markers',
                               marker=go.scatter3d.Marker(
                                   color=self.c_convex.alphas.detach().cpu().numpy(),
                                   size=1.5,
                                   coloraxis="coloraxis", symbol='diamond',
                                   showscale=False))
                       ] +
                       [go.Scatter3d(x=[x.flatten()[ind_l[i_l]],
                                        x.flatten()[ind_l[i_l]] + dF[i_l, 0]],
                                     y=[y.flatten()[ind_l[i_l]],
                                        y.flatten()[ind_l[i_l]] + dF[i_l, 1]],
                                     z=[z.flatten()[ind_l[i_l]],
                                        z.flatten()[ind_l[i_l]] + dF[i_l, 2]],
                                     showlegend=False, mode='lines',
                                     line=dict(
                                         color='rgb(255,0,0)',
                                         width=3, showscale=False
                                     )) for i_l in range(ind_l.shape[0])
                        ], rows=1, cols=2)
        if show: fig.show()
        if save is not None:
            fig.write_html(save + ".html")
        return fig

class ManifoldVQEMulti(ManifoldVQE):
    def __init__(self, manifold, n_layers, n_components,
                 init_alpha_mode, init_alpha_linear_scale,
                 init_alpha_minval, init_alpha_range, cost_gamma,
                 min_zero_gamma, init_points="uniform", fixed_points=False,
                 alpha_dim=1, eps=1e-3, device=torch.device("cpu"),
                 th_dtype=torch.float32, base_density='uniform'):
        super().__init__(manifold, eps, device, th_dtype, base_density)

        self.c_convex = CConvex_Multi(manifold, n_layers, n_components,
                                       init_alpha_mode, init_alpha_linear_scale,
                                       init_alpha_minval, init_alpha_range, cost_gamma,
                                       min_zero_gamma, init_points=init_points,
                                       fixed_points=fixed_points,
                                       alpha_dim=alpha_dim, device=device,
                                       th_dtype=th_dtype)

    def get_points_params(self):
        return self.c_convex.get_points_params()
    
    def post_hook(self, optimizer, args, kwargs):
        self.c_convex.post_hook(optimizer, args, kwargs)
    
    def plot_cconvex(self, show=False, save=None):
        x, y, z = self.manifold.surface()
        U = torch.stack([x.view(-1), y.view(-1), z.view(-1)], 1).to(self.device)
        cconv_values = self.c_convex(U).reshape(x.shape).detach().cpu().numpy()
        x = x.numpy()
        y = y.numpy()
        z = z.numpy()

        fig = make_subplots(
            rows=1, cols=self.c_convex.n_layers + 1,
            subplot_titles=[f"learned points {i_l}" for i_l in range(
                self.c_convex.n_layers)]
                           + ["Vector field"],
            specs=[[{'type': 'scene'} for i in range(self.c_convex.n_layers + 1)]])
        fig.layout.coloraxis.colorscale = "ice"
        fig.update_layout(coloraxis=dict(colorscale="ice", colorbar=dict(len=0.5, y=0)))
        fig.update_layout(coloraxis2=dict(colorscale="reds", colorbar=dict(len=0.5,
                                                                           y=0.6)))
        fig.layout.title = "C-convex function"
        for i_l in range(self.c_convex.n_layers):
            mu = self.c_convex.mus[i_l]
            mu = self.manifold.projx(mu)
            mu = mu.detach().cpu().numpy()
            fig.add_traces([
                go.Surface(x=x, y=y, z=z,
                           # colorscale="Greys",
                           coloraxis="coloraxis",
                           surfacecolor=cconv_values,
                           showscale=False),
                go.Scatter3d(x=mu[:, 0], y=mu[:, 1], z=mu[:, 2],
                             mode='markers',
                             marker=go.scatter3d.Marker(
                                 color=self.c_convex.alphas[i_l].detach().cpu().numpy(),
                                 size=2.5,
                                 coloraxis="coloraxis2",
                                 line={"width": 0.5},
                                 showscale=False))
            ], rows=1, cols=i_l + 1)

        ind_l = np.random.choice(cconv_values.flatten().shape[0], 1000)
        dF = self.dF_riemannian(U[ind_l, :]).detach().cpu().numpy()
        fig.add_traces([
                           go.Surface(x=x, y=y, z=z, coloraxis="coloraxis",
                                      surfacecolor=cconv_values,
                                      showscale=False, showlegend=False, ),
                           go.Scatter3d(
                               x=x.flatten()[ind_l] + dF[:, 0],
                               y=y.flatten()[ind_l] + dF[:, 1],
                               z=z.flatten()[ind_l] + dF[:, 2],
                               showlegend=False, mode='markers',
                               marker=go.scatter3d.Marker(size=1.5, symbol='diamond',
                                                          showscale=False))
                       ] +
                       [go.Scatter3d(x=[x.flatten()[ind_l[i_l]],
                                        x.flatten()[ind_l[i_l]] + dF[i_l, 0]],
                                     y=[y.flatten()[ind_l[i_l]],
                                        y.flatten()[ind_l[i_l]] + dF[i_l, 1]],
                                     z=[z.flatten()[ind_l[i_l]],
                                        z.flatten()[ind_l[i_l]] + dF[i_l, 2]],
                                     showlegend=False, mode='lines',
                                     line=dict(
                                         color='rgb(255,0,0)',
                                         width=3, showscale=False
                                     )) for i_l in range(ind_l.shape[0])
                        ], rows=1, cols=1 + self.c_convex.n_layers)
        if show: fig.show()
        if save is not None:
            fig.write_html(save + ".html")
        return fig


class AddScalar(torch.nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar

    def __call__(self, x):
        return x + self.scalar


class ManifoldVQR(ManifoldVQE):
    def __init__(self, manifold,n_u=50000,
                 eps=1e-3, device=torch.device("cpu"),
                 th_dtype=torch.float32, base_density='uniform'):
        super().__init__(manifold, eps, device, th_dtype, base_density)
        self.n_u = n_u

    def check_cmonotone(self, conds, batch_size=50, max_iters=1e4, base_samples=None, ):
        import itertools
        if base_samples is None:
            base_samples = self.base_distr.sample(batch_size)
        base_samples = base_samples.to(self.th_dtype).to(self.device)

        for cond in conds:
            target_samples = self.forward(base_samples, cond).detach()
            max_iters_x = max_iters
            # compute costs
            c_matrix = self.manifold.cost(base_samples, target_samples)
            c0 = torch.diag(c_matrix).sum()
            for i_p in range(1, base_samples.shape[0]):
                cp = torch.diag(torch.roll(c_matrix, i_p, 0)).sum()
                if c0 > cp:
                    return False
                if max_iters_x < i_p:
                    break
            max_iters_x -= i_p
            for i_p, perm in enumerate(
                    itertools.permutations(range(base_samples.shape[0]))):
                if i_p == 0: continue
                cp = c_matrix[perm, range(base_samples.shape[0])].sum()
                if c0 > cp:
                    return False
                if max_iters_x < i_p:
                    break
        return True

    def beta_g(self, x, cond):
        b_eval = self.beta(x)
        g_eval = self.gnet(cond)
        return torch.bmm(b_eval.unsqueeze(1),
                         g_eval.unsqueeze(2)).squeeze(1)

    def check_cmonotone_betag(self, conds, batch_size=50, max_iters=1e4,
                              base_samples=None, ):
        import itertools
        if base_samples is None:
            base_samples = self.base_distr.sample(batch_size)
        base_samples = base_samples.to(self.th_dtype).to(self.device)
        for cond in conds:
            beta_g = partial(self.beta_g, cond=cond)
            dF = self.dF_riemannian(base_samples, pot_fn=beta_g)
            target_samples = self.manifold.exponential_map(base_samples, dF)
            max_iters_x = max_iters
            # compute costs
            c_matrix = self.manifold.cost(base_samples, target_samples)
            c0 = torch.diag(c_matrix).sum()
            for i_p in range(1, base_samples.shape[0]):
                cp = torch.diag(torch.roll(c_matrix, i_p, 0)).sum()
                if c0 > cp:
                    return False
                if max_iters_x < i_p:
                    break
            max_iters_x -= i_p
            for i_p, perm in enumerate(
                    itertools.permutations(range(base_samples.shape[0]))):
                if i_p == 0: continue
                cp = c_matrix[perm, range(base_samples.shape[0])].sum()
                if c0 > cp:
                    return False
                if max_iters_x < i_p:
                    break
        return True

    def check_cmonotone_beta(self, batch_size=50, max_iters=1e4,
                             base_samples=None, ):
        import itertools
        if base_samples is None:
            base_samples = self.base_distr.sample(batch_size)
        base_samples = base_samples.to(self.th_dtype).to(self.device)

        # target_samples = self.forward(base_samples, cond).detach()
        dF = self.dF_riemannian(base_samples, pot_fn=self.beta)

        dF = dF.view(self.beta_dim, -1, self.manifold.D)
        dF_cc = dF.sum(0)
        target_samples = self.manifold.exponential_map(base_samples, dF_cc)
        max_iters_x = max_iters
        # compute costs
        c_matrix = self.manifold.cost(base_samples, target_samples)
        c0 = torch.diag(c_matrix).sum()
        for i_p in range(1, base_samples.shape[0]):
            cp = torch.diag(torch.roll(c_matrix, i_p, 0)).sum()
            if c0 > cp:
                return False
            if max_iters_x < i_p:
                break
        max_iters_x -= i_p
        for i_p, perm in enumerate(
                itertools.permutations(range(base_samples.shape[0]))):
            if i_p == 0: continue
            cp = c_matrix[perm, range(base_samples.shape[0])].sum()
            if c0 > cp:
                return False
            if max_iters_x < i_p:
                break

        for i_cc in range(self.beta_dim):
            dF_cc = dF[i_cc, :, :]
            target_samples = self.manifold.exponential_map(base_samples, dF_cc)

            max_iters_x = max_iters
            # compute costs
            c_matrix = self.manifold.cost(base_samples, target_samples)
            c0 = torch.diag(c_matrix).sum()
            for i_p in range(1, base_samples.shape[0]):
                cp = torch.diag(torch.roll(c_matrix, i_p, 0)).sum()
                if c0 > cp:
                    return False
                if max_iters_x < i_p:
                    break
            max_iters_x -= i_p
            for i_p, perm in enumerate(
                    itertools.permutations(range(base_samples.shape[0]))):
                if i_p == 0: continue
                cp = c_matrix[perm, range(base_samples.shape[0])].sum()
                if c0 > cp:
                    return False
                if max_iters_x < i_p:
                    break
        return True

    def check_cmonotone_phi(self, batch_size=50, max_iters=1e4,
                            base_samples=None, ):
        import itertools
        if base_samples is None:
            base_samples = self.base_distr.sample(batch_size)
        base_samples = base_samples.to(self.th_dtype).to(self.device)

        # target_samples = self.forward(base_samples, cond).detach()
        dF = self.dF_riemannian(base_samples, pot_fn=self.c_convex)
        target_samples = self.manifold.exponential_map(base_samples, dF)

        max_iters_x = max_iters
        # compute costs
        c_matrix = self.manifold.cost(base_samples, target_samples)
        c0 = torch.diag(c_matrix).sum()
        for i_p in range(1, base_samples.shape[0]):
            cp = torch.diag(torch.roll(c_matrix, i_p, 0)).sum()
            if c0 > cp:
                return False
            if max_iters_x < i_p:
                break
        max_iters_x -= i_p
        for i_p, perm in enumerate(
                itertools.permutations(range(base_samples.shape[0]))):
            if i_p == 0: continue
            cp = c_matrix[perm, range(base_samples.shape[0])].sum()
            if c0 > cp:
                return False
            if max_iters_x < i_p:
                break
        return True

    def evaluate_full_batch_old(self, Y_train, X_train=None, U_batch=None):
        """
        X_train is the conditioning
        """
        x_batch_size = X_train.shape[0]
        y_batch_size = Y_train.shape[1]
        batch_size = x_batch_size * y_batch_size
        if U_batch is None:
            U_batch = self.base_distr.sample(y_batch_size).to(self.th_dtype).to(
                self.device)
        u_batch_size = U_batch.shape[0]
        # equiprobable samples
        nu = torch.ones(batch_size, 1, device=self.device) / batch_size
        mu = torch.ones(u_batch_size, 1, device=self.device) / u_batch_size

        # Sample U
        UY = self.manifold.cost(U_batch, Y_train.reshape(batch_size, -1) ) # [u_B, x_B * y_B]

        b_eval = self.beta(U_batch)
        phi_eval = self.c_convex(U_batch)
        # Evaluate loss function - Part I
        net_X = self.gnet(X_train).repeat(y_batch_size, 1)
        lse_arg = - UY - b_eval @ net_X.T - phi_eval.reshape(-1, 1) # [u_B, x_B * y_B]
        psi_eval = self.eps * torch.logsumexp(lse_arg / self.eps, dim=0)
        objective = torch.tensordot(psi_eval, nu, 1)
        # Evaluate loss function - Part II
        objective = objective + (phi_eval.T @ mu).squeeze()
        # Evaluate loss function - Part III
        x_bar = mu @ nu.T @ net_X
        objective = objective + torch.trace(b_eval.T @ x_bar)

        return objective.squeeze()
    

    def comput_cost_matrix(self, U_batch, Y_train):
        """
        :param U_batch: batch of points on the manifold, torch.tensor of shape [
        u_B, D]
        :param Y_train: sampled points on the manifold, torch.tensor of shape [
        x_B, y_B, D]
        :return: the cost matrix, torch.tensor of shape [x_B, y_B, u_B]
        """
        x_batch_size = Y_train.shape[0]
        batch_size = Y_train.shape[1]
        cost_matrix = self.manifold.cost(Y_train.reshape(x_batch_size * batch_size, *Y_train.shape[2:]),
                                         U_batch).reshape(x_batch_size, batch_size,
                                                          U_batch.shape[
                                                              0])  # [x_B, y_B, u_B]
        return cost_matrix
    

    def evaluate_full_batch(self, Y_train, X_train=None, U_batch=None, 
                            U_conj=None, cc_loss=True, c_batch=5000):
        """
        Given the pairs (Y,X), evaluate the VQR  functional for each sample. We
        minimize the sum of c-convex functions
        :param Y_train: sampled points on the manifold, torch.tensor of shape [
        x_B, y_B, D]
        :param X_train: batch of conditionings, torch.tensor of shape [x_B, dX]
        :param U_batch: batch of samples from the base distribution, torch.tensor
        of shape [u_B, D]
        :return: the loss computed fora each X_train over the samples Y_train,
        torch.tensor of shape [x_B]
        """
        x_batch_size = Y_train.shape[0]
        batch_size = Y_train.shape[1]
        if U_batch is None:
            U_batch = self.base_distr.sample(batch_size).to(self.th_dtype) # [u_B, D]
        u_batch_size = U_batch.shape[0]
        if c_batch > 0 and c_batch < U_batch.shape[0]: 
            phi = list()
            X_train = X_train.to(self.device)
            for i in range(0, U_batch.shape[0], c_batch):
                phi.append(self.compute_phi(U_batch[i:i+c_batch].to(self.device), X_batch=X_train))
            phi = torch.cat(phi, dim=1)
            X_train = X_train.to("cpu")
        else:     
            phi = self.compute_phi(U_batch.to(self.device), X_batch=X_train.to(self.device))  # [x_B, u_B]
        objective_phi = torch.mean(phi)  # [1]
        
        # compute conjugate
        if c_batch > 0 and c_batch < batch_size: 
            cost_matrix = list()
            psi = list()
            U_batch = U_batch.to(self.device)
            for i in range(0, batch_size, c_batch):
                UY_batch = self.comput_cost_matrix(U_batch.to(self.device), Y_train[:,i:i+c_batch,:].to(self.device)) 
                cost_matrix.append(UY_batch)
                psi.append(self.conjugate(phi=phi if U_conj is None else None, 
                             cost_matrix=UY_batch if U_conj is None else None,
                                        U_batch=U_conj))
            U_batch = U_batch.to("cpu")
            cost_matrix = torch.cat(cost_matrix, dim=1)
            psi = torch.cat(psi, dim=1)
        else:     
            cost_matrix = self.comput_cost_matrix(U_batch.to(self.device), Y_train.to(self.device))  # [x_B, y_B, u_B]
            psi = self.conjugate(phi=phi if U_conj is None else None, 
                             cost_matrix=cost_matrix if U_conj is None else None,
                             U_batch=U_conj)  # [x_B, y_B]
            
        
        # Evaluate loss function - Part I
        objective_psi = torch.mean(psi)  # [1]
        # Evaluate loss function - Part II
        objective = objective_phi + objective_psi

        # conjugate loss
        if cc_loss:
            phi_cc = self.conjugate(phi=psi,
                                    cost_matrix=cost_matrix.permute(0, 2, 1))  # [x_B, u_B]
            cc_loss = (phi_cc - phi) ** 2
            return objective.squeeze(), cc_loss.mean()
            objective += torch.mean(cc_loss / batch_size, dim=0).mean() * cc_weight

        return objective.squeeze()

    def conjugate(self, Y_batch=None, U_batch=None, X_batch=None,
                  phi=None, cost_matrix=None, n_base=100000, eps=None, vmap=False):
        """
            Compute the conjugate (Psi) of the forward potential function
            :param Y_train: sampled points on the manifold, torch.tensor of shape [
                            x_B, y_B, D]
            U_batch: batch of points on the manifold, torch.tensor of shape [u_B, D]
            phi: convex forward potential function, torch.tensor of shape [x_B, u_B]
            cost_matrix: cost matrix, torch.tensor of shape [u_B, y_B]
            n_base: number of samples to use for the base distribution
            vmap: whether to use vmap
            return: the conjugate of the forward potential function, torch.tensor of
                    [x_B, y_B]
        """

        if U_batch is None and (phi is None or cost_matrix is None):
            U_batch = self.manifold.grid(n_base).to(self.th_dtype).to(self.device)

        if phi is None:
            phi = self.compute_phi(U_batch, X_batch=X_batch)  # [x_B, u_B]

        if cost_matrix is None:
            if vmap:
                Y_batch = Y_batch.unsqueeze(0).unsqueeze(0)
            if len(Y_batch.shape) == 1:
                Y_batch = Y_batch.reshape(-1, 1)
            cost_matrix = self.comput_cost_matrix(U_batch, Y_batch)  # [x_B, y_B, u_B]

        ## min version (- concave)
        phi_cc = cost_matrix + phi.unsqueeze(1)  # [x_B,  y_B, u_B,]
        if eps is None:
            eps = self.eps
        if eps > 0:
            phi_cc = eps * torch.logsumexp(- phi_cc / eps, dim=-1)
        else:
            phi_cc = - phi_cc.min(dim=-1).values  # [x_B, y_B]

        if vmap:
            phi_cc = phi_cc.squeeze()
        return phi_cc

    def train_loop(self, target_ditribution, batch_size=None, train_samples=1000,
                   results_dir="", lr=1e-3, num_iters=20000, intermediate_plot=False,
                   save_dir="", n_u=50000, step_interval=0,
                   cc_weight=1e1, c_batch=0,
                   l1_param_weight=0., use_tqdm=False,
                   early_stopping=False, kde_interval=200,
                   identity_iter=10, num_workers=1,
                   best=False, max_tollerance=6):
        """
        Y_train: samples from target distribution
        """
        self.train()


        train_samples = int(train_samples)

        losses = dict()
        losses_cc = dict() 
        losses_kde = dict()
        if batch_size is None: batch_size = train_samples
        else: batch_size = int(batch_size)
        dataset = DensityDataset(target_ditribution, n_samples=train_samples,
                                 mode="train", th_dtype=self.th_dtype)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers)
                
        if identity_iter > 0:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
            self.optimizer.register_step_post_hook(self.post_hook)
            tollerance_id = 0
            print("### Identity pretraining ###")
            for i in range(identity_iter):
                loss =[ 0,0]
                pbar = tqdm(dataloader) if use_tqdm else dataloader
                i_epoch = 0
                self.optimizer.zero_grad()
                for Y, X in pbar:
                    U_batch = self.base_distr.sample(n_u).to(self.th_dtype)
                    loss_fn, cc_loss = self.evaluate_full_batch(U_batch.repeat([X.shape[0]] + [
                        1] * U_batch.ndim),
                                                       X_train=X,
                                                       U_batch=U_batch,
                                                       c_batch=c_batch)
                    loss[0] += loss_fn.item() / len(pbar)
                    loss[1] += cc_loss.item() / len(pbar)
                    if cc_weight > 0:
                        loss_fn = loss_fn + cc_weight * cc_loss
                    loss_fn /= len(pbar)
                    loss_fn.backward()
                    if step_interval > 0 and i_epoch % step_interval == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    i_epoch += 1
                    if use_tqdm:  pbar.set_description(f"Loss: {loss[0] / i_epoch:.2e}")                
                self.optimizer.step()
                
                loss[0] /= len(dataloader)
                loss[1] /= len(dataloader)
                if abs(loss[0]) < 1e-5:
                    tollerance_id += 1
                    if tollerance_id > 3:
                        break
                else:
                    tollerance_id = 0
                if i % 2 == 0:
                    print(f"\033[1m Iteration {i}: \033[0m  {loss[0]:.4f}")
                    losses[i-identity_iter] = loss[0]
                    losses_cc[i-identity_iter] = loss[1]
            print("### End Identity pretraining ###")

        best_state_dict = None
        best_loss = None
        weight_loss = 1e-4
        kde_best = None
        last_kde = 0
        tollerance = 0
        n_eval = int(1e3)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.optimizer.register_step_post_hook(self.post_hook)

        cond_list = [torch.tensor([i_x]).to(self.th_dtype)
                     for i_x in np.linspace(target_ditribution.min_cond,
                                            target_ditribution.max_cond, 3)]

        for i in range(num_iters):
            loss = [0,0]
            i_epoch = 0
            pbar = tqdm(dataloader) if use_tqdm else dataloader
            self.optimizer.zero_grad()
            for Y, X in pbar:
                U_batch = self.base_distr.sample(n_u).to(self.th_dtype)
                loss_fn, cc_loss = self.evaluate_full_batch(Y,
                                                   X_train=X,
                                                   U_batch=U_batch,c_batch=c_batch) 
                loss[0] += loss_fn.item() / len(pbar)
                loss[1] += cc_loss.item() / len(pbar)
                if cc_weight > 0:
                    loss_fn = loss_fn + cc_weight * cc_loss
                loss_fn /= len(pbar)
                loss_fn.backward()
                if step_interval > 0 and i_epoch % step_interval == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                i_epoch += 1
                if use_tqdm: pbar.set_description(f"Loss: {loss[0] / i_epoch:.2e}")
                # break
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss[0] /= len(dataloader)
            loss[1] /= len(dataloader)
            if i % 2 == 0:
                print(f"\033[1m Iteration {i}: \033[0m  {loss[0]:.4f}")
                losses[i] = loss[0]
                losses_cc[i] = loss[1]

            if best_loss is None or best_loss > loss[0]:
                best_loss = loss_fn.item()

            if i % kde_interval == 0:
                self.eval()
                print(f"\033[94m KDE loss: \033[0m ")
                with torch.no_grad():
                    kde_losses = []
                    surf_samples = self.manifold.grid(1000).clone().detach()
                    for cond in cond_list:
                        sample_gt, cond_val = target_ditribution.sample(n_eval,
                                                                        cond)
                        samples_est_random = self.sample(cond_val, n_eval)
                        kde_loss = self.manifold.kde_loss(samples_est_random,
                                                          sample_gt.to(
                                                              self.th_dtype).to(
                                                              self.device),
                                                          surf_samples=surf_samples)
                        kde_losses.append(kde_loss.item())
                        print(f"      cond: {cond_val}; kde_loss: {kde_loss.item():.4f}")

                    losses_kde[i] = (sum(kde_losses) / len(kde_losses))
                    last_kde = losses_kde[i + identity_iter]
                    print(f" \033[94m Average KDE loss: {last_kde:.4f} \033[0m ")
                
                f,ax = plot_trainloss(losses, losses_cc, losses_kde, results_dir)
                plt.close()

                if kde_best is None or kde_best >= last_kde:
                    kde_best = last_kde
                    tollerance = 0
                else:
                    tollerance += 1
                if early_stopping and tollerance > max_tollerance:
                    break

                if intermediate_plot and i % (4 * kde_interval) == 0:
                    with torch.no_grad():
                        
                        samples_gt, cond_val = target_ditribution.sample(n_eval,
                                                                         cond_list[1])
                        base_samples = self.base_distr.sample(n_eval).to(
                            self.th_dtype).to(self.device)
                        samples_est = self.sample(cond_val,
                                                  base_sample=base_samples,
                                                  vmap=True, chunk_size=1000
                                                  ).cpu().detach()
                        plot_samples = [samples_gt.detach().cpu(), samples_est]
                        self.manifold.plot_samples_3d(plot_samples,
                                                      save=os.path.join(results_dir,
                                                                        f'result_{i}'),
                                                      show=False,
                                                      subplots_titles=[
                                                          "GT distribution.",
                                                          f"Forward"])
                       
                self.train()

            if best and ((weight_loss * best_loss + kde_best) <= (weight_loss *
                                                                  loss[0] + last_kde)):
                best_state_dict = self.state_dict()
                torch.save({
                    'target_distr': target_ditribution, 'base_distr': self.base_distr,
                    'model_state_dict': best_state_dict},
                    os.path.join(results_dir, 'best_state_dict.pth'))

        plot_trainloss(losses, losses_cc, losses_kde, results_dir)
        return self.state_dict()


    def phi_eval(self, us, xs):
        phi_eval = self.c_convex(us)
        b_eval = self.beta(us)
        net_X = self.gnet(xs)
        if xs.ndim == 1:
            net_X = torch.tile(net_X, (us.shape[0], 1))
        out = torch.bmm(b_eval.unsqueeze(1), net_X.unsqueeze(2)).squeeze(1) + \
              phi_eval.reshape(-1, 1)

        return out

    def psi_eval(self, Y_train, X_train, n_base=1000, base_distribution=None, ):
        if base_distribution is None:
            base_distribution = self.base_distr
        # Sample U
        U_batch = base_distribution.sample(n_base).to(self.th_dtype).to(
            self.device)

        UY = - self.manifold.cost(U_batch, Y_train)
        b_eval = self.beta(U_batch)
        phi_eval = self.c_convex(U_batch)
        # Evaluate loss function - Part I
        net_X = self.gnet(X_train)
        if net_X.ndim == 1:
            net_X = torch.tile(net_X, (U_batch.shape[0], 1))

        lse_arg = UY - b_eval @ net_X.T - phi_eval.reshape(-1, 1)
        psi_eval = self.eps * torch.logsumexp(lse_arg / self.eps, dim=0)
        return psi_eval

    def dF_riemannian(self, us, conds=None, fast=False, forward=True, n_base=1000,
                      pot_fn=None):
        if fast:
            # functorch
            pass
        else:
            if us.shape[0] == 1:
                if pot_fn is None:
                    if forward:
                        pot_fn = partial(self.phi_eval, xs=conds)
                    else:
                        pot_fn = partial(self.psi_eval, X_train=conds, n_base=n_base,
                                         base_distribution=None)
                with torch.enable_grad():
                    us = us.clone().detach().requires_grad_(True)
                    dF = torch.autograd.functional.jacobian(pot_fn, us)
                    dF = self.manifold.tangent_projection(us, dF.view(-1, us.shape[-1]))
            else:
                dF = torch.concat(
                    [
                        self.dF_riemannian(us[[i], :], conds, fast=fast,
                                           forward=forward,
                                           n_base=n_base, pot_fn=pot_fn)
                        for i in range(us.shape[0])
                    ], dim=0
                )
        return dF

    def forward(self, U, conds, t=1):
        """
        OT = exp map graf
        """
        dF = self.dF_riemannian(U, conds)
        z = self.manifold.exponential_map(U, t * dF)

        return z

    def inverse(self, Y, conds, t=1, n_base=1000):
        """
        OT = exp map graf
        """
        dF = self.dF_riemannian(Y, conds, forward=False, n_base=n_base)
        z = self.manifold.exponential_map(Y, t * dF)

        return z

    def sample(self, conds, N=100, base_sample=None, vmap=True, chunk_size=None):
        if base_sample is None:
            base_sample = self.base_distr.sample(N)
        U_sampled = base_sample.clone().detach().to(self.th_dtype).to(
            self.device).requires_grad_(True)

        if not torch.is_tensor(conds):
            # check if float or int and batch accordingly
            if isinstance(conds, float) or isinstance(conds, int):
                conds = np.array([conds])
            conds = torch.tensor(conds)
        if conds.ndim == 0:
            conds = conds.unsqueeze(0)
        conds = conds.to(self.device)

        with torch.no_grad():
            net_X = self.precompute_conditioning(conds)  # [bx, dx]

        pot_fn = partial(lambda us: self.compute_phi(us, net_X=net_X, vmap=True))
        dF = self.manifold.grad(pot_fn, U_sampled, vmap=vmap, chunk_size=chunk_size)  #
        # gradient of the  convex  potential
        with torch.no_grad():
            samples_ = self.manifold.exponential_map(U_sampled, dF)

        return samples_

    def sample_inverse(self, target_sample, conds, n_base=1000, n_conj=10000, vmap=True,
                       chunk_size=None, eps=None):
        Y_sampled = target_sample.clone().detach().to(self.th_dtype).to(
            self.device).requires_grad_(True)
        if not torch.is_tensor(conds):
            conds = torch.tensor(conds)
        

        with torch.no_grad():
            U_conj = self.manifold.grid(n_conj).to(self.th_dtype)
            phi = self.compute_phi(U_conj.to(self.device), X_batch=conds.to(self.device))

        pot_fn = lambda ys: self.conjugate(Y_batch=ys, U_batch=U_conj.to(self.device),
                                           X_batch=conds.to(self.device),
                                           phi=phi, eps=eps, vmap=True)
        dF = self.manifold.grad(pot_fn, Y_sampled, vmap=vmap, chunk_size=chunk_size)  #
        # gradient of the convex  conjugate

        with torch.no_grad():
            samples_ = self.manifold.exponential_map(Y_sampled, dF)

        return samples_

    def likelihood(self, conds, U_sampled=None,Y_sampled=None, n_base=50000, 
                   eps=5e-2, log=False, chunk_size=None):
        if conds.ndim == 0:
            conds = conds.unsqueeze(0)
        if U_sampled is None:
            U_sampled = self.manifold.grid(n_base).to(self.th_dtype)
        if Y_sampled is None:
            with torch.no_grad():
                Y_sampled = self.sample(conds, base_sample=U_sampled)
        with torch.no_grad():
            phi = self.compute_phi(U_sampled, X_batch=conds.to(self.device))

        inv_pot_fn = lambda ys: self.conjugate(Y_batch=ys, 
                                                    U_batch=U_sampled.to(self.device), 
                                                    X_batch=conds.to(self.device),
                                                    phi=phi, eps=eps, vmap=True)
        def dF_riemmanian_inv( ys):
            dF = torch.func.grad(inv_pot_fn)(ys)
            dF = self.manifold.tangent_projection(ys, dF).squeeze(0)
            return dF
        
        def Qinv(ys):
            dF = dF_riemmanian_inv(ys)
            samples_ = self.manifold.exponential_map(ys, dF).squeeze(0)
            return samples_

        def _H_inv(y):
            J = torch.func.jacfwd(Qinv)(y).squeeze(0)
            dF = partial(dF_riemmanian_inv)(y)
            return dF, J

        dFs, Js = [], []
        Es = []
        for y_batch_idx in tqdm(range(0, Y_sampled.shape[0], chunk_size)):
            # print(y_batch_idx)
            y_batch = Y_sampled[y_batch_idx:y_batch_idx+chunk_size]
            dF, J = torch.vmap(_H_inv)(
                y_batch.clone().detach().to(self.device).requires_grad_(True)
            )
            # dFs.append(dF.detach())
            E = self.manifold.tangent_orthonormal_basis(y_batch.to(self.device), dF.detach())
            Es.append(E)
            Js.append(J.detach())
        J = torch.cat(Js, dim=0)
        E = torch.cat(Es, dim=0)
        
        JE = torch.bmm(J, E)
        JETJE = torch.einsum('nji,njk->nik', JE, JE)
        print(f"JE: {JE.shape}, JETJE: {JETJE.shape}")
        
        sign, lh = torch.slogdet(JETJE)
        lh = lh.cpu() * 0.5 + self.base_distr.log_prob(U_sampled).cpu()
        if log:
            return lh, Y_sampled
        else:
            return torch.exp(lh), Y_sampled
    
    def likelihood_fw(self, conds, U_sampled=None, n_base=10000,
                   eps=None, chunk_size=None, log=False):
        if conds.ndim == 0:
            conds = conds.unsqueeze(0)
        if U_sampled is None:
            U_sampled = self.manifold.grid(n_base).to(self.th_dtype)

        with torch.no_grad():
            Y_sampled = self.sample(conds=conds, base_sample=U_sampled)
            net_X = self.precompute_conditioning(conds.to(self.device)) 
                
        pot_fn = partial(self.compute_phi, net_X=net_X, vmap=True, gamma=eps)
        def dF_riemmanian( us):
            dF = torch.func.grad(pot_fn)(us)
            dF = self.manifold.tangent_projection(us, dF).squeeze(0)
            return dF
        
        def Q(us):
            dF = dF_riemmanian(us)
            samples_ = self.manifold.exponential_map(us, dF)
            return samples_


        def _H(u):
            J = torch.func.jacfwd(Q)(u).squeeze(0)
            dF = partial(dF_riemmanian)(u)
            return dF, J

        
        dFs, Js = [], []
        Es = []
        for y_batch_idx in tqdm(range(0, U_sampled.shape[0], chunk_size)):
            # print(y_batch_idx)
            u_batch = U_sampled[y_batch_idx:y_batch_idx+chunk_size]
            dF, J = torch.vmap(_H)(
                u_batch.clone().detach().to(self.device).requires_grad_(True)
            )
            dFs.append(dF.detach())
            Js.append(J.detach())
        dF = torch.cat(dFs, dim=0)
        J = torch.cat(Js, dim=0)

        E = self.manifold.tangent_orthonormal_basis(U_sampled.to(self.device), dF.to(self.device))
        
        JE = torch.bmm(J, E)
        JETJE = torch.einsum('nji,njk->nik', JE, JE)
        print(f"JE: {JE.shape}, JETJE: {JETJE.shape}")
        
    
        sign, lh = torch.slogdet(JETJE)
        lh = - lh.cpu() * 0.5 # + self.base_distr.log_prob(U_sampled).cpu()
        if log:
            return lh, Y_sampled
        else:
            return torch.exp(lh), Y_sampled

class ManifoldVQRSingle(ManifoldVQR):
    def __init__(self, manifold, n_components, init_alpha_mode, init_alpha_linear_scale,
                 init_alpha_minval, init_alpha_range, cost_gamma, min_zero_gamma,
                 n_u=50000, n_class_cond=None, n_layers=1,
                 fixed_points=False, init_points=None,
                 beta_dim=1, cond_size=1, cond_hidden=(2, 10),
                 eps=1e-3, device=torch.device("cpu"),
                 activation=None,
                 th_dtype=torch.float32, base_density='uniform', stack=False):
        super().__init__(manifold, n_u, eps, device, th_dtype,
                         base_density)
        self.n_components = n_components
        self.beta_dim = beta_dim
        self.convx_act = nn.ReLU()
        

        self.define_modules(manifold, n_components, init_alpha_mode, init_alpha_linear_scale,
                 init_alpha_minval, init_alpha_range, cost_gamma, min_zero_gamma,
                 n_u, n_class_cond, n_layers,
                 fixed_points, init_points,
                 beta_dim, cond_size, cond_hidden,
                 eps, device, activation,
                 th_dtype, base_density, stack)

    def define_modules(self, manifold, n_components, init_alpha_mode, init_alpha_linear_scale,
                 init_alpha_minval, init_alpha_range, cost_gamma, min_zero_gamma,
                 n_u=50000, n_class_cond=None, n_layers=1,
                 fixed_points=False, init_points=None,
                 beta_dim=1, cond_size=1, cond_hidden=(2, 10),
                 eps=1e-3, device=torch.device("cpu"),
                 activation=None,
                 th_dtype=torch.float32, base_density='uniform', stack=False):
        cond_hidden = [cond_size] + list(cond_hidden)
        self.n_convx = len(cond_hidden) - 1
        self.beta_list = nn.ModuleList()
        self.x_net_list = nn.ModuleList()

        for i in range(self.n_convx):

            if stack:
                self.beta_list.append(CConvexStack(self.manifold, cond_hidden[i + 1],
                                                   n_components,
                                                   init_alpha_mode,
                                                   init_alpha_linear_scale,
                                                   init_alpha_minval, init_alpha_range,
                                                   cost_gamma,
                                                   min_zero_gamma,
                                                   init_points=init_points,
                                                   fixed_points=fixed_points,
                                                   device=device,
                                                   th_dtype=th_dtype, ))
            else:
                self.beta_list.append(CConvex(manifold, n_components,
                                              init_alpha_mode,
                                              init_alpha_linear_scale,
                                              init_alpha_minval, init_alpha_range,
                                              cost_gamma,
                                              min_zero_gamma,
                                              init_points=init_points,
                                              fixed_points=fixed_points,
                                              device=device,
                                              th_dtype=th_dtype,
                                              alpha_dim=cond_hidden[i + 1],
                                              keepdim=True))
            if n_class_cond is not None and i == 0:
                self.x_net_list.append(nn.Sequential(
                    nn.Embedding(n_class_cond, cond_hidden[i + 1]), nn.ReLU(), ).to(
                    device).to(th_dtype))
            else:
                self.x_net_list.append(
                    MLP(in_dim=cond_hidden[i], hidden_dims=[cond_hidden[i + 1]],
                        last_nl="relu").to(device).to(th_dtype))
        self.activation = activation
        if activation is not None:
            self.act = nn.SELU()


        self.n_components = n_components
        self.c_convex = CConvex(manifold, n_components,
                 init_alpha_mode, init_alpha_linear_scale,
                 init_alpha_minval, init_alpha_range, cost_gamma,
                 min_zero_gamma,
                 init_points, fixed_points=fixed_points,
                 device=device,
                 th_dtype=th_dtype, alpha_dim=1, keepdim=True)


    def post_hook(self, optimizer, args, kwargs):
        for i in range(len(self.beta_list)):
            self.beta_list[i].post_hook(optimizer, args, kwargs)
        self.c_convex.post_hook(optimizer, args, kwargs)

    def plot_cconvex(self, show=False, save=None):
        x, y, z = self.manifold.surface()
        U = torch.stack([x.view(-1), y.view(-1), z.view(-1)], 1).to(self.device)
        cconv_values = self.c_convex(U).reshape(x.shape).detach().cpu().numpy()
        x = x.numpy()
        y = y.numpy()
        z = z.numpy()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["learned points", "Vector field"],
            specs=[[{'type': 'scene'} for i in range(2)]])
        fig.layout.coloraxis.colorscale = "ice"
        fig.update_layout(coloraxis=dict(colorscale="ice", colorbar=dict(len=0.5, y=0)))
        fig.update_layout(coloraxis2=dict(colorscale="reds", colorbar=dict(len=0.5,
                                                                           y=0.6)))
        fig.layout.title = "C-convex function"
        fig.add_traces([
            go.Surface(x=x, y=y, z=z,
                       coloraxis="coloraxis",
                       surfacecolor=cconv_values,
                       showscale=False),
            go.Scatter3d(x=self.c_convex.sample_points[:, 0].detach().cpu().numpy(),
                         y=self.c_convex.sample_points[:, 1].detach().cpu().numpy(),
                         z=self.c_convex.sample_points[:, 2].detach().cpu().numpy(),
                         mode='markers',
                         marker=go.scatter3d.Marker(
                             color=self.c_convex.alphas.detach().cpu().numpy(),
                             size=2.5,
                             coloraxis="coloraxis2",
                             line={"width": 0.5},
                             showscale=False))
        ], rows=1, cols=1)

        ind_l = np.random.choice(cconv_values.flatten().shape[0], 1000)
        dF = self.dF_riemannian(U[ind_l, :], ).detach().cpu().numpy()
        fig.add_traces([
                           go.Surface(x=x, y=y, z=z, coloraxis="coloraxis",
                                      surfacecolor=cconv_values,
                                      showscale=False, showlegend=False, ),
                           go.Scatter3d(
                               x=x.flatten()[ind_l] + dF[:, 0],
                               y=y.flatten()[ind_l] + dF[:, 1],
                               z=z.flatten()[ind_l] + dF[:, 2],
                               showlegend=False, mode='markers',
                               marker=go.scatter3d.Marker(
                                   color=self.c_convex.alphas.detach().cpu().numpy(),
                                   size=1.5,
                                   coloraxis="coloraxis", symbol='diamond',
                                   showscale=False))
                       ] +
                       [go.Scatter3d(x=[x.flatten()[ind_l[i_l]],
                                        x.flatten()[ind_l[i_l]] + dF[i_l, 0]],
                                     y=[y.flatten()[ind_l[i_l]],
                                        y.flatten()[ind_l[i_l]] + dF[i_l, 1]],
                                     z=[z.flatten()[ind_l[i_l]],
                                        z.flatten()[ind_l[i_l]] + dF[i_l, 2]],
                                     showlegend=False, mode='lines',
                                     line=dict(
                                         color='rgb(255,0,0)',
                                         width=3, showscale=False
                                     )) for i_l in range(ind_l.shape[0])
                        ], rows=1, cols=2)
        if show: fig.show()
        if save is not None:
            fig.write_html(save + ".html")
        return fig

    def precompute_conditioning(self, X_batch):
        out_x = [self.x_net_list[0](X_batch)]
        for i in range(1, len(self.x_net_list)):
            out_x.append(self.x_net_list[i](out_x[-1]))
        return out_x

    def compute_phi(self, U_batch, X_batch=None, net_X=None, 
                    vmap=False, gamma=None):
        """
        U_batch: batch of points on the manifold, torch.tensor of shape (bu, D)
        X_batch: batch of points on the manifold, torch.tensor of shape (bx, dX)
        net_X: precomputed conditioning, torch.tensor of shape (bx, dx)
        vmap: whether to use vmap
        """
        batched = U_batch.dim() == 1 or vmap
        if batched:
            U_batch = U_batch.unsqueeze(0)
        assert U_batch.shape[1] == self.manifold.D

        if net_X is None:
            if X_batch.ndim == 0:
                X_batch = X_batch.unsqueeze(0)
            net_X = self.precompute_conditioning(X_batch)  # [bx, dx]

        x_batch = net_X[0].shape[0]
        
        phi_eval = self.c_convex(U_batch, gamma=gamma).T # [bu,1]
        for i_l in range(len(self.beta_list)):
            b_eval = self.beta_list[i_l](U_batch)  # [bu, dx]
            inner_prod = net_X[i_l] @ b_eval.T  # [bx, bu]
            if self.activation is not None:
                inner_prod = self.act(inner_prod)
            phi_eval = phi_eval + inner_prod   # [bx, bu]

        if vmap:
            phi_eval = phi_eval.squeeze(1)
        if batched:
            phi_eval = phi_eval.squeeze(0)
        return phi_eval


class ManifoldVQRMulti(ManifoldVQRSingle):

    def define_modules(self, manifold, n_components, init_alpha_mode, init_alpha_linear_scale,
                 init_alpha_minval, init_alpha_range, cost_gamma, min_zero_gamma,
                 n_u=50000, n_class_cond=None, n_layers=1,
                 fixed_points=False, init_points=None,
                 beta_dim=1, cond_size=1, cond_hidden=(2, 10),
                 eps=1e-3, device=torch.device("cpu"),
                 activation=None,
                 th_dtype=torch.float32, base_density='uniform', stack=False):
        cond_hidden = [cond_size] + list(cond_hidden)
        self.n_convx = len(cond_hidden) - 1
        self.beta_list = nn.ModuleList()
        self.x_net_list = nn.ModuleList()

        for i in range(self.n_convx):

            if stack:
                self.beta_list.append(CConvexStackMulti(manifold, cond_hidden[i + 1],
                                                   n_components,
                                                   init_alpha_mode,
                                                   init_alpha_linear_scale,
                                                   init_alpha_minval, init_alpha_range,
                                                   cost_gamma,
                                                   min_zero_gamma,
                                                   init_points=init_points,
                                                   fixed_points=fixed_points,
                                                   n_layers=n_layers,
                                                   device=device,
                                                   th_dtype=th_dtype, ))
            else:
                self.beta_list.append(CConvex_Multi(manifold, n_layers, n_components,
                 init_alpha_mode, init_alpha_linear_scale,
                 init_alpha_minval, init_alpha_range, cost_gamma,
                 min_zero_gamma, init_points, fixed_points,
                 alpha_dim=cond_hidden[i + 1],
                 device=device, th_dtype=th_dtype, keepdim=True))
            if n_class_cond is not None and i == 0:
                self.x_net_list.append(nn.Sequential(
                    nn.Embedding(n_class_cond, cond_hidden[i + 1]), nn.ReLU(), ).to(
                    device).to(th_dtype))
            else:
                self.x_net_list.append(
                    MLP(in_dim=cond_hidden[i], hidden_dims=[cond_hidden[i + 1]],
                        last_nl="relu").to(device).to(th_dtype))
        self.activation = activation
        if activation is not None:
            self.act = nn.SELU()


        self.n_components = n_components
        self.c_convex = CConvex_Multi(manifold, n_layers, n_components,
                 init_alpha_mode, init_alpha_linear_scale,
                 init_alpha_minval, init_alpha_range, cost_gamma,
                 min_zero_gamma, init_points, fixed_points,
                 alpha_dim=1,
                 device=device, th_dtype=th_dtype, keepdim=True)
        