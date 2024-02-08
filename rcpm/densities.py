import random
import torch
from torch.distributions.normal import Normal
import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle

from rcpm.manifolds import Manifold, Sphere, Torus
import rcpm.utils as utils

@dataclass
class Density(ABC):
    manifold: Manifold
    is_uniform = False

    @abstractmethod
    def log_prob(self, x):
        pass

    @abstractmethod
    def sample(self, n_samples, **kwargs):
        pass

    def __hash__(self): return 0  # For jitting


## SPHERE
class SphereUniform(Density):
    is_uniform = True

    def log_prob(self, xs):
        assert xs.ndim == 2
        n_batch, D = xs.shape
        assert D == self.manifold.D

        if self.manifold.D == 2:
            SA = 2. * np.pi
        elif self.manifold.D == 3:
            SA = 4. * np.pi
        else:
            raise NotImplementedError()

        return torch.full([n_batch], torch.log(1. / torch.as_tensor(SA)))

    def sample(self, n_samples, **kwargs):
        with torch.no_grad():
            xs = torch.normal(0, 1.0, size=[int(n_samples), self.manifold.D])
            xs = self.manifold.projx(xs)
        return xs

    def contour_phi(self, n=100, phi=0.0):
        thetaphi = torch.stack([torch.linspace(0, 2 * np.pi, n),
                                torch.tensor([phi]).repeat(n)], 1)
        thetaphi = utils.spherical_to_euclidean(thetaphi)
        return thetaphi

    def contour_theta(self, n=100, theta=0.0):
        thetaphi = torch.stack([
            torch.tensor([theta]).repeat(n),
            torch.linspace(0, 2 * np.pi, n), ], 1)
        thetaphi = utils.spherical_to_euclidean(thetaphi)
        return thetaphi

    def contours(self, center, n_contour=5, n_samples=100):
        if self.manifold.D == 2:
            # S1
            theta_center = utils.S1euclideantospherical(center)
            phis = np.linspace(0, np.pi, n_contour)
            contours = list()
            for phi in phis:
                theta_cont = [theta_center - phi, theta_center + phi]
                cont = np.stack([utils.S1sphericaltoeuclidean(theta_cont[0]),
                                 utils.S1sphericaltoeuclidean(theta_cont[1])], 0)
                contours.append(cont)
            taus = phis / np.pi
        elif self.manifold.D == 3:
            # S2
            R = utils.rotation3D_matrix(center, np.array([0, 0, 1.0]))
            contours = list()
            phis = np.linspace(0, np.pi, n_contour)
            for phi in phis:
                cont = self.contour_phi(n_samples, phi)
                contours.append(cont @ R)
            taus = phis / np.pi
        else:
            raise NotImplementedError("Only S1 and S2 are supported")
        return contours, taus


@dataclass
class Spherical3DUniform(Density):
    theta_max: float = 2 * np.pi
    phi_max: float = 2 * np.pi
    theta_min: float = 0.0
    phi_min: float = 0.0
    is_uniform = True

    def __post_init__(self):
        assert self.manifold.D == 3

    def log_prob(self, xs):
        raise NotImplementedError()

    def sample(self, n_samples):
        theta = torch.rand(size=[n_samples, 1]) * (
                self.theta_max - self.theta_min) + self.theta_min
        phi = torch.rand(size=[n_samples, 1]) * (
                self.phi_max - self.phi_min) + self.phi_min
        xs = torch.tensor(utils.spherical_to_euclidean(torch.cat([theta, phi], 1)))
        return xs

    def contour_phi(self, n=100, phi=0.0):
        thetaphi = torch.stack([torch.linspace(0, 2 * np.pi, n),
                                torch.tensor([phi]).repeat(n)], 1)
        thetaphi = utils.spherical_to_euclidean(thetaphi)
        return thetaphi

    def contour_theta(self, n=100, theta=0.0):
        thetaphi = torch.stack([
            torch.tensor([theta]).repeat(n),
            torch.linspace(0, 2 * np.pi, n), ], 1)
        thetaphi = utils.spherical_to_euclidean(thetaphi)
        return thetaphi

    def contours(self, center=None, n_contour=5, n_samples=100):
        if center is None:
            center = utils.spherical_to_euclidean(
                torch.tensor([np.mean([self.theta_min, self.theta_max]),
                              np.mean([self.phi_min, self.phi_max])]))
        R = utils.rotation_matrix(center, np.array([0, 0, 1.0]))  # z -> center
        contours = list()
        phis = np.linspace(0, np.pi, n_contour)
        for phi in phis:
            cont = self.contour_phi(n_samples, phi)
            contours.append(cont @ R)
        return contours, phis


@dataclass()
class WrappedNormal(Density):
    loc: np.ndarray
    scale: np.ndarray

    def __init__(self, manifold, loc=None, scale=None):
        super().__init__(manifold=manifold)
        if loc is None:
            self.loc = self.manifold.projx(-torch.ones(self.manifold.D))
        else:
            if not torch.is_tensor(loc):
                loc = torch.tensor(loc, dtype=float)
            self.loc = self.manifold.projx(loc)
        self.scale = scale
        if self.scale is None:
            self.scale = torch.full((self.manifold.D - 1,), 0.3)
        elif not torch.is_tensor(self.scale):
            self.scale = torch.tensor(scale, dtype=float)

        self.distributions = [Normal(0.0, 1.0) for d in range(self.manifold.D - 1)]

    def log_prob(self, z):
        raise NotImplementedError("We need to define the log prob on the shpere")
        

    def sample(self, n_samples, **kwargs):
        v = self.scale * torch.stack([d.sample([n_samples]) for d
                                      in self.distributions], dim=1)
        v = self.manifold.unsqueeze_tangent(v)
        x = self.manifold.zero_like(self.loc)
        u = self.manifold.transp(x, self.loc, v)
        z = self.manifold.exponential_map(self.loc, u)
        return z

    def __hash__(self):
        return 0  # For jitting

    def frechet_mean(self):
        return self.loc

    def contour_phi(self, n=100, phi=0.0):
        thetaphi = torch.stack([torch.linspace(0, 2 * np.pi, n),
                                torch.tensor([phi]).repeat(n)], 1)
        thetaphi = utils.spherical_to_euclidean(thetaphi)
        return thetaphi

    def contour_theta(self, n=100, theta=0.0):
        thetaphi = torch.stack([
            torch.tensor([theta]).repeat(n),
            torch.linspace(0, 2 * np.pi, n), ], 1)
        thetaphi = utils.spherical_to_euclidean(thetaphi)
        return thetaphi

    def contours(self, center=None, n_contour=5, n_samples=100):
        if center is None:
            center = self.loc
        R = utils.rotation_matrix(center, np.array([0, 0, 1.0]))  # z -> center
        contours = list()
        taus = np.linspace(0, np.pi, n_contour)
        for tau in taus:
            cont = self.contour_phi(n_samples, tau)
            contours.append(cont @ R)
        taus /= np.pi
        return contours, taus


class ConditionalWrappedNormal(WrappedNormal):
    loc: np.ndarray
    continous_cond = True

    def __init__(self, manifold, loc=None, min_cond=0.1, max_cond=1.0,
                 dtype=torch.float32, **kwargs):
        super().__init__(manifold=manifold)
        self.cond_size = 1
        if loc is None:
            self.loc = self.manifold.projx(-torch.ones(self.manifold.D))
        else:
            if not torch.is_tensor(loc):
                loc = torch.tensor(loc, dtype=float)
            self.loc = self.manifold.projx(loc)
        self.distributions = [torch.distributions.normal.Normal(0.0, 1.0)
                              for d in range(self.manifold.D - 1)]

        self.dtype = dtype

        self.min_cond = min_cond
        self.max_cond = max_cond

    def log_prob(self, z, scale):
        raise NotImplementedError("We need to define the log prob on the shpere")

    def _sample(self, n_samples, cond=0):
        if not torch.is_tensor(cond):
            cond = torch.as_tensor(cond, dtype=self.dtype)
        cond = torch.clamp(cond, min=self.min_cond, max=self.max_cond)
        if cond.ndim == 0:
            cond = cond.unsqueeze(0)
        v = cond.tile(1, self.manifold.D - 1) * torch.stack([d.sample([
            n_samples]) for d in self.distributions], dim=1)
        v = self.manifold.unsqueeze_tangent(v)
        x = self.manifold.zero_like(self.loc)
        u = self.manifold.transp(x, self.loc, v)
        z = self.manifold.exponential_map(self.loc, u)

        return z, cond.to(self.dtype)

    def sample(self, n_samples, cond=None, train=False, **kwargs):
        if cond is None:
            cond = random.uniform(self.density.min_cond, self.density.max_cond)

        return self._sample(n_samples, cond)

    def frechet_mean(self, cond=None):
        return self.loc


class ConditionalWrappedNormaldF(WrappedNormal):
    loc: np.ndarray
    continous_cond = True

    def __init__(self, manifold, loc=None, scale=None, df=None, min_cond=1e-16, \
                 max_cond=1.0, dtype=torch.float64):
        super().__init__(manifold=manifold)
        self.cond_size = 1
        if loc is None:
            self.loc = self.manifold.projx(-torch.ones(self.manifold.D))
        else:
            if not torch.is_tensor(loc):
                loc = torch.tensor(loc, dtype=dtype)
            self.loc = self.manifold.projx(loc)
        self.scale = scale
        if self.scale is None:
            self.scale = torch.full((self.manifold.D - 1,), 0.3)
        elif not torch.is_tensor(self.scale):
            self.scale = torch.tensor(scale, dtype=dtype)
        self.df = df
        if self.df is None:
            self.df = torch.rand((self.manifold.D,), dtype=dtype)
        self.df = self.manifold.tangent_projection(self.loc, self.df)

        self.distributions = [torch.distributions.normal.Normal(0.0, 1.0)
                              for d in range(self.manifold.D - 1)]

        self.min_cond = min_cond
        self.max_cond = max_cond

    def log_prob(self, z, loc):
        raise NotImplementedError("We need to define the log prob on the shpere")

    def sample(self, n_samples, cond=None):
        if cond is None:
            # scale = torch.full((self.manifold.D - 1,), 0.3)
            cond = torch.rand(n_samples, 1) * (self.max_cond - self.min_cond) + \
                   self.min_cond
        elif not torch.is_tensor(cond):
            cond = np.tile(cond, (n_samples, 1))
            cond = torch.clamp(torch.tensor(cond, dtype=float),
                               min=self.min_cond, max=self.max_cond)

        loc = self.manifold.exponential_map(self.loc, cond * self.df)
        # Sample in 2d with the defined variance
        v = self.scale * torch.stack([d.sample([n_samples]) for d
                                      in self.distributions], dim=1)
        # Make it a tangent -- append 0 to the first axis
        v = self.manifold.unsqueeze_tangent(v)
        # v = torch.cat((v, torch.zeros_like(v[..., :1]),), dim=-1)
        x = self.manifold.zero_like(loc)
        # what does transp do?
        u = self.manifold.transp(x, loc, v)
        z = self.manifold.exponential_map(loc, u)
        return z, cond

    def frechet_mean(self, cond):
        cond = torch.clamp(torch.tensor(cond, dtype=float),
                           min=self.min_cond, max=self.max_cond)
        loc = self.manifold.exponential_map(self.loc, cond * self.df)
        return loc
    


from experiments.datasets.shapes import StarDataProvider, HeartDataProvider


@dataclass()
class DataProvider(WrappedNormal):
    loc: None
    scale: None

    def __init__(self, manifold, loc=None, scale=None):
        super().__init__(manifold=manifold, loc=loc, scale=scale)
        self.distributions = None

    def log_prob(self, z):
        raise NotImplementedError("We need to define the log prob on the shpere")


    def sample(self, n_samples, train=False, **kwargs):
        v = self.scale * self.distributions[0].sample(n_samples)[1]
        v = self.manifold.unsqueeze_tangent(v)
        x = self.manifold.zero_like(self.loc)
        u = self.manifold.transp(x, self.loc, v)
        z = self.manifold.exponential_map(self.loc, u)
        return z

    def __hash__(self):
        return 0  # For jitting

    def frechet_mean(self):
        return self.loc


@dataclass()
class Star(DataProvider):
    loc: None
    scale: None

    def __init__(self, manifold, loc=None, scale=None):
        super().__init__(manifold=manifold, loc=loc, scale=scale)
        self.distributions = [StarDataProvider(), ]


@dataclass()
class Heart(DataProvider):
    loc: None
    scale: None

    def __init__(self, manifold, loc=None, scale=None):
        super().__init__(manifold=manifold, loc=loc, scale=scale)
        self.distributions = [HeartDataProvider(), ]


class ConditionalScaleDataProvider(ConditionalWrappedNormal):
    continous_cond = True
    def __init__(self, manifold, loc=None, min_cond=0.3, max_cond=2.0, **kwargs):
        super().__init__(manifold=manifold, loc=loc, min_cond=min_cond,
                         max_cond=max_cond)
        self.distributions = None
        self.th = 0.05
        
    def log_prob(self, z, cond, n_grid=30000):
        samples_grid = self.manifold.grid(n_grid)
        # compute kde to understand inside points
        samples_gt = self.sample(n_grid, cond)[0]
        test_kde = self.manifold.kde(samples_gt, samples_grid, bandwidth=0.01, norm=False)
        th = (test_kde.max() - test_kde.min()) * self.th + test_kde.min()
        test_kde_b = torch.where(test_kde < th, torch.zeros_like(test_kde), 
                            torch.full_like(test_kde, 1.0))
        area_factor = n_grid / (test_kde_b.sum() * self.manifold.area)
        
        loglh_gt = self.manifold.kde(samples_gt, z, bandwidth=0.01, norm=False)
        loglh_gt = torch.where(loglh_gt < th, torch.zeros_like(loglh_gt), 
                               torch.full_like(loglh_gt, area_factor))
        loglh_gt = torch.log(loglh_gt)
        return loglh_gt
    
    def sample(self, n_samples, cond=None, train=False, **kwargs):
        
        if cond is None:
            cond = torch.rand(1) * (self.max_cond - self.min_cond)
        elif not torch.is_tensor(cond):
            cond = torch.tensor(cond, dtype=torch.float32)

        # Sample in 2d with the defined variance
        v = cond.tile(1, self.manifold.D - 1) * self.distributions[0].sample(n_samples)[
            1]
        v = self.manifold.unsqueeze_tangent(v).to(self.loc.device)
        x = self.manifold.zero_like(self.loc)
        u = self.manifold.transp(x, self.loc, v)
        z = self.manifold.exponential_map(self.loc, u)
        return z, cond


class ConditionalStarScale(ConditionalScaleDataProvider):
    def __init__(self, manifold, loc=None, min_cond=0.3, max_cond=2.0, **kwargs):
        super().__init__(manifold=manifold, loc=loc, min_cond=min_cond,
                         max_cond=max_cond)
        self.distributions = [StarDataProvider(), ]
        self.th = 0.4


class ConditionalHeartScale(ConditionalScaleDataProvider):
    
    def __init__(self, manifold, loc=None, min_cond=0.3, max_cond=2.0, **kwargs):
        super().__init__(manifold=manifold, loc=loc, min_cond=min_cond,
                         max_cond=max_cond)
        self.distributions = [HeartDataProvider(), ]
        self.th = 0.1



@dataclass
class RezendeSphereFourMode(Density):
    # Define the target_mu as a PyTorch tensor
    target_mu = utils.spherical_to_euclidean(torch.tensor([
        [1.5, 0.7 + torch.pi / 2],
        [1., -1. + torch.pi / 2],
        [5., 0.6 + torch.pi / 2],
        [4., -0.7 + torch.pi / 2]
    ]))
    
    def __post_init__(self):
        self.modes = []
        scale = torch.full((self.manifold.D-1,), 0.31)
        self.dists = [
            WrappedNormal(manifold=self.manifold, loc=loc, scale=scale)
            for loc in self.target_mu
        ]

    def log_prob(self, x):
        assert x.ndim == 2
        logp = torch.logsumexp(10. * torch.mm(x, self.target_mu.T.to(x.dtype)), dim=1)

        return logp

    def sample(self, n_samples, mode=None):
        n = int(np.ceil(n_samples / len(self.dists)))
        samples = torch.cat([
            d.sample(n) for d in self.dists
        ], dim=0)
        samples = samples[torch.randperm(n_samples)]
        return samples
    
    def frechet_mean(self):
        return self.target_mu.mean(dim=0)
    
class RezendeSphereFourModeTrasl(Density):
    # Define the target_mu as a PyTorch tensor
    target_mu_sph = torch.tensor([
            [1.5, 0.7 + torch.pi / 2],
            [1., -1. + torch.pi / 2],
            [5., 0.6 + torch.pi / 2],
            [4., -0.7 + torch.pi / 2]
        ])
    
    
    target_mu = utils.spherical_to_euclidean(target_mu_sph)
    
    loc: np.ndarray
    continous_cond = True

    def __init__(self, manifold, loc=None, scale=0.3, min_cond=0.1, max_cond=1,
                 dtype=torch.float32, **kwargs):
        super().__init__(manifold=manifold)
        self.cond_size = 1

        #
        self.scale = scale
        self.dtype = dtype
        self.min_cond = min_cond
        self.max_cond = max_cond
        
        self.distributions = [torch.distributions.normal.Normal(0.0, 1.0) for d in range(self.manifold.D - 1)]


    def log_prob(self, x, cond):
        # TODO: This is unnormalized
        if not torch.is_tensor(cond):
            cond = torch.as_tensor(cond, dtype=self.dtype)
        cond = torch.clamp(cond, min=self.min_cond, max=self.max_cond)
        assert x.ndim == 2

        ## dist
        targets = cond * self.target_mu_sph + (1-cond) * self.target_mu_sph.mean(dim=0)
        targets = utils.spherical_to_euclidean(targets)
        
        x_dist = self.manifold.dist(x, targets.to(x.dtype)) * np.sqrt(2)
        var = torch.as_tensor(self.scale ** 2)
        p = torch.exp(-x_dist ** 2 / (2* var )) / torch.sqrt(2 * np.pi * var)
        p = p.sum(dim=1) / 4
        p = torch.log(p)
        return p

    def sample(self, n_samples, cond=0, train=False, **kwargs):
        if not torch.is_tensor(cond):
            cond = torch.as_tensor(cond, dtype=self.dtype)
        cond = torch.clamp(cond, min=self.min_cond, max=self.max_cond)

        
        v = torch.stack([d.sample([n_samples]) for d
                                      in self.distributions], dim=1)
        # normalize
        v = self.scale * v / np.sqrt(2)
        v = self.manifold.unsqueeze_tangent(v)
        perm = torch.randint(self.target_mu_sph.size(0), (n_samples,))
        targets = cond * self.target_mu_sph + (1-cond) * self.target_mu_sph.mean(dim=0)
        targets = utils.spherical_to_euclidean(targets)
        loc = targets[perm]
        x = self.manifold.zero_like(loc)
        yv = torch.einsum("ij,ij->i", loc, v).unsqueeze(1)
        xy = torch.einsum("ij,ij->i", x, loc).unsqueeze(1)
        u =  v - yv / (1 + xy) * (x + loc)
        samples = self.manifold.exponential_map(loc, u)
                
        return samples, cond

    def frechet_mean(self, cond=0):
        targets = cond * self.target_mu_sph + (1-cond) * self.target_mu_sph.mean(dim=0)
        mean_mu = targets.mean(dim=0, keepdim=True) 
        mean_mu = utils.spherical_to_euclidean(mean_mu)
        return mean_mu.squeeze(0)

class RezendeSphereFourModeCond(Density):
    # Define the target_mu as a PyTorch tensor
    target_mu_sph = torch.tensor([
            [1.5, 0.7 + torch.pi / 2],
            [1., -1. + torch.pi / 2],
            [5., 0.6 + torch.pi / 2],
            [4., -0.7 + torch.pi / 2]
        ])
    
    
    target_mu = utils.spherical_to_euclidean(target_mu_sph)
    
    loc: np.ndarray
    continous_cond = True

    def __init__(self, manifold, loc=None, min_cond=0.1, max_cond=0.5,
                 dtype=torch.float32, **kwargs):
        super().__init__(manifold=manifold)
        self.cond_size = 1

        #
        self.normal_phi = torch.distributions.normal.Normal(0.0, 1)
        self.normal_psi = torch.distributions.normal.Normal(0.0, 1)
        self.dtype = dtype
        self.min_cond = min_cond
        self.max_cond = max_cond
        
        self.distributions = [torch.distributions.normal.Normal(0.0, 1.0) for d in range(self.manifold.D - 1)]


    def log_prob(self, x, cond):
        # This is unnormalized
        if not torch.is_tensor(cond):
            cond = torch.as_tensor(cond, dtype=self.dtype)
        cond = torch.clamp(cond, min=self.min_cond, max=self.max_cond)
        assert x.ndim == 2

        ## dist
        x_dist = self.manifold.dist(x, self.target_mu.to(x.dtype)) * np.sqrt(2)
        var = (cond ** 2)
        p = torch.exp(-x_dist ** 2 / (2* var )) / torch.sqrt(2 * np.pi * var)
        p = p.sum(dim=1) / 4
        p = torch.log(p)
        return p

    def sample(self, n_samples, cond=0, train=False, **kwargs):
        if not torch.is_tensor(cond):
            cond = torch.as_tensor(cond, dtype=self.dtype)
        cond = torch.clamp(cond, min=self.min_cond, max=self.max_cond)

        
        v = torch.stack([d.sample([n_samples]) for d
                                      in self.distributions], dim=1)
        # normalize
        v = cond.tile(1, v.shape[1]) * v / np.sqrt(2)
        # Make it a tangent -- append 0 to the first axis
        v = self.manifold.unsqueeze_tangent(v)
        perm = torch.randint(self.target_mu_sph.size(0), (n_samples,))
        loc = self.target_mu[perm]
        x = self.manifold.zero_like(loc)
        yv = torch.einsum("ij,ij->i", loc, v).unsqueeze(1)
        xy = torch.einsum("ij,ij->i", x, loc).unsqueeze(1)
        u =  v - yv / (1 + xy) * (x + loc)
        samples = self.manifold.exponential_map(loc, u)
                
        return samples, cond

    def frechet_mean(self, cond=0):
        return self.target_mu.mean(dim=0)
    
class RezendeTorusFourModeTrasl(Density):
    # Define the target_mu as a PyTorch tensor
    target_mu_sph = torch.tensor([
            [1.5, 0.7+ np.pi/2 ],
            [1., -1. +np.pi/2],
            [5., 0.6 +np.pi/2],
            [4., -0.7 +np.pi/2 ]
        ]).flip(1)
    
    
    target_mu = utils.TORUSsphericaltoeuclidean(target_mu_sph)
    
    loc: np.ndarray
    continous_cond = True

    def __init__(self, manifold, loc=None, scale=0.2, min_cond=0.1, max_cond=1,
                 dtype=torch.float32, **kwargs):
        super().__init__(manifold=manifold)
        self.cond_size = 1

        #
        self.normal_phi = torch.distributions.normal.Normal(0.0, 1)
        self.normal_psi = torch.distributions.normal.Normal(0.0, 1)
        self.dtype = dtype
        self.scale = scale
        self.min_cond = min_cond
        self.max_cond = max_cond
    

    def log_prob(self, x, cond=0):
        # This is unnormalized
        if not torch.is_tensor(cond):
            cond = torch.as_tensor(cond, dtype=self.dtype)
        cond = torch.clamp(cond, min=self.min_cond, max=self.max_cond)
        assert x.ndim == 2
        
        x_sph = torch.as_tensor(utils.TORUSeuclideantospherical(x))
        p = torch.zeros(x.shape[0])
        var = (cond ** 2)
        for i in range(4):
            p += torch.exp(-(x_sph[:, 0]-self.target_mu_sph[i,0]) ** 2 / (2* var )) / torch.sqrt(2 * np.pi * var)
            p += torch.exp(-(x_sph[:, 1]-self.target_mu_sph[i,1]) ** 2 / (2* var )) / torch.sqrt(2 * np.pi * var)

        p = p / 8
        p = torch.log(p)

        ## dist
        targets = cond * self.target_mu_sph + (1-cond) * self.target_mu_sph.mean(dim=0)
        targets = utils.TORUSsphericaltoeuclidean(targets)

        var = torch.as_tensor(self.scale ** 2)
        x_dist = self.manifold.manifolds[0].dist(x[:,:2], targets.to(x.dtype)[:,:2])
        p = torch.exp(-x_dist ** 2 / (2* var )) / torch.sqrt(2 * np.pi * var)
        y_dist = self.manifold.manifolds[1].dist(x[:,2:], targets.to(x.dtype)[:,2:])
        p = p * torch.exp(-y_dist ** 2 / (2* var )) / torch.sqrt(2 * np.pi * var)
        p = p.sum(dim=1) / 4
        p = torch.log(p)
        return p


    def sample(self, n_samples, cond=0, train=False, **kwargs):
        if not torch.is_tensor(cond):
            cond = torch.as_tensor(cond, dtype=self.dtype)
        cond = torch.clamp(cond, min=self.min_cond, max=self.max_cond)

        pp = torch.zeros(n_samples, 2)
        pp[:, 0] = self.scale * torch.normal(0,1.0,(n_samples,))
        pp[:, 1] = self.scale * torch.normal(0,1.0,(n_samples,))
        perm = torch.randint(0, 4, (n_samples,))
        targets = cond * self.target_mu_sph + (1-cond) * self.target_mu_sph.mean(dim=0)
        loc = targets[perm]
        pp = pp + loc
                
        pp = utils.TORUSsphericaltoeuclidean(pp)
        return pp, cond

    def frechet_mean(self, cond=0):
        targets = cond * self.target_mu_sph + (1-cond) * self.target_mu_sph.mean(dim=0)
        mean_mu = targets.mean(dim=0, keepdim=True) 
        mean_mu = utils.TORUSsphericaltoeuclidean(mean_mu)
        return mean_mu.squeeze(0)

class RezendeTorusFourModeCond(Density):
    # Define the target_mu as a PyTorch tensor
    target_mu_sph = torch.tensor([
            [1.5, 0.7+ np.pi/2 ],
            [1., -1. +np.pi/2],
            [5., 0.6 +np.pi/2],
            [4., -0.7 +np.pi/2 ]
        ]).flip(1)
    
    
    target_mu = utils.TORUSsphericaltoeuclidean(target_mu_sph)
    
    loc: np.ndarray
    continous_cond = True

    def __init__(self, manifold, loc=None, min_cond=0.1, max_cond=0.5,
                 dtype=torch.float32, **kwargs):
        super().__init__(manifold=manifold)
        self.cond_size = 1

        #
        self.normal_phi = torch.distributions.normal.Normal(0.0, 1)
        self.normal_psi = torch.distributions.normal.Normal(0.0, 1)
        self.dtype = dtype
        self.min_cond = min_cond
        self.max_cond = max_cond
    

    def log_prob(self, x, cond=0):
        # TODO: This is unnormalized
        if not torch.is_tensor(cond):
            cond = torch.as_tensor(cond, dtype=self.dtype)
        cond = torch.clamp(cond, min=self.min_cond, max=self.max_cond)
        assert x.ndim == 2
        
        x_sph = torch.as_tensor(utils.TORUSeuclideantospherical(x))
        p = torch.zeros(x.shape[0])
        var = (cond ** 2)
        for i in range(4):
            p += torch.exp(-(x_sph[:, 0]-self.target_mu_sph[i,0]) ** 2 / (2* var )) / torch.sqrt(2 * np.pi * var)
            p += torch.exp(-(x_sph[:, 1]-self.target_mu_sph[i,1]) ** 2 / (2* var )) / torch.sqrt(2 * np.pi * var)

        p = p / 8
        p = torch.log(p)

        ## dist
        var = (cond ** 2)
        x_dist = self.manifold.manifolds[0].dist(x[:,:2], self.target_mu.to(x.dtype)[:,:2])
        p = torch.exp(-x_dist ** 2 / (2* var )) / torch.sqrt(2 * np.pi * var)
        y_dist = self.manifold.manifolds[1].dist(x[:,2:], self.target_mu.to(x.dtype)[:,2:])
        p = p * torch.exp(-y_dist ** 2 / (2* var )) / torch.sqrt(2 * np.pi * var)
        p = p.sum(dim=1) / 4
        p = torch.log(p)
        return p


    def sample(self, n_samples, cond=0, train=False, **kwargs):
        if not torch.is_tensor(cond):
            cond = torch.as_tensor(cond, dtype=self.dtype)
        cond = torch.clamp(cond, min=self.min_cond, max=self.max_cond)

        pp = torch.zeros(n_samples, 2)
        pp[:, 0] = cond * torch.normal(0,1.0,(n_samples,))
        pp[:, 1] = cond * torch.normal(0,1.0,(n_samples,))
        perm = torch.randint(0, 4, (n_samples,))
        loc = self.target_mu_sph[perm]
        pp = pp + loc
                
        pp = utils.TORUSsphericaltoeuclidean(pp)
        return pp, cond

    def frechet_mean(self, cond=0):
        return self.target_mu.mean(dim=0)


@dataclass
class ProductUniformComponents(Density):
    def __init__(self, manifold):
        super().__init__(manifold)
        self.base_dists = []
        for man in self.manifold.manifolds:
            self.base_dists.append(get_uniform(man))
        self.is_uniform = True

    def log_prob(self, xs):
        # Note this is not necessarily uniform
        assert xs.ndim == 2
        n_batch = xs.shape[0]
        log_probas = torch.zeros(n_batch)
        d = 0
        for i, base_dist in enumerate(self.base_dists):
            D = self.manifold.manifolds[i].D
            log_probas += base_dist.log_prob(xs[:, d:d + D])
            d = d + D
        return log_probas

    def sample(self, n_samples, **kwargs):
        # Note this is not necessarily uniform
        xs = []
        for base_dist in self.base_dists:
            samples_man = base_dist.sample(n_samples)
            xs.append(samples_man)
        xs = torch.cat(xs, dim=-1)
        return xs

    def __hash__(self): return 0  # For jitting

    def contours(self, center, n_contour=5, n_samples=100):
        taus = np.linspace(0, 1., n_contour, endpoint=True)
        contours = self.manifold.geodesic_contours(center, taus, n=n_samples)

        return contours, taus


## TORUS ##
class TorusUniform(ProductUniformComponents):
    def __init__(self, manifold):
        super().__init__(manifold)
        assert isinstance(manifold, Torus)
        self.is_uniform = True

    def sample(self, n_samples, **kwargs):
        xs = super().sample(n_samples)
        if self.manifold.D == 2:
            xs = torch.stack((utils.S1euclideantospherical(xs[:, :2]),
                              utils.S1euclideantospherical(xs[:, 2:])), 1)
        return xs

    def contour_phi(self, n=100, phi=0.0):
        theta = torch.linspace(0, 2 * np.pi, n)
        phi = torch.tensor([phi]).repeat(n)
        if self.manifold.D == 2:
            contour = torch.cat([theta, phi], 1)
        else:
            contour = torch.cat([utils.S1sphericaltoeuclidean(theta),
                                 utils.S1sphericaltoeuclidean(phi)], 1)
        return contour

    def contour_theta(self, n=100, theta=0.0):
        theta = torch.tensor([theta]).repeat(n)
        phi = torch.linspace(0, 2 * np.pi, n)
        if self.manifold.D == 2:
            contour = torch.cat([theta, phi], 1)
        else:
            contour = torch.cat([utils.S1sphericaltoeuclidean(theta),
                                 utils.S1sphericaltoeuclidean(phi)], 1)
        return contour


class WrappedNormalTorus(Density):
    loc: np.ndarray
    scale: np.ndarray

    def __init__(self, manifold, loc=[1, 1], scale=None):
        super().__init__(manifold=manifold)
        assert isinstance(manifold, Torus)
        self.loc = loc
        if not torch.is_tensor(self.loc):
            self.loc = torch.tensor(self.loc, dtype=float)
        if self.loc.shape[0] == 2:
            self.loc = utils.TORUSsphericaltoeuclidean(self.loc)
        self.scale = scale
        if self.scale is None:
            self.scale = torch.full((2,), 0.3)
        elif not torch.is_tensor(self.scale):
            self.scale = torch.tensor(scale, dtype=float)

        self.normal_theta = WrappedNormal(self.manifold.manifolds[0], self.loc[:2],
                                          self.scale[0])
        self.normal_phi = WrappedNormal(self.manifold.manifolds[1], self.loc[2:],
                                        self.scale[1])

    def log_prob(self, z):
        raise NotImplementedError()


    def sample(self, n_samples, **kwargs):
        v_theta = self.normal_theta.sample(n_samples)
        v_phi = self.normal_phi.sample(n_samples)
        z = torch.cat([v_theta, v_phi], dim=1)  # [n_samples, 4]

        return z

    def __hash__(self):
        return 0  # For jitting

    def frechet_mean(self):
        return self.loc

    def contours(self, center=None, n_contour=5, n_samples=100):
        # split the contour on the S1 spheres
        contour_theta, taus_theta = self.normal_theta.contours(center, n_contour,
                                                               n_samples)
        contour_phi, taus_phi = self.normal_phi.contours(center, n_contour, n_samples)
        contours = list()
        for i in range(n_contour):
            contours.append(torch.cat([contour_theta[i], contour_phi[i]], dim=1))
        return contours, taus_theta

class DataProviderTorus(WrappedNormalTorus):
    loc: None
    scale: None

    def __init__(self, manifold, loc=np.array([1, 1]), scale=None):
        super().__init__(manifold=manifold, loc=loc, scale=scale)
        if loc.shape[0] == 2:
            self.spherical_loc = torch.as_tensor(loc)
        else:
            self.spherical_loc = utils.TORUSeuclideantospherical(loc)
        self.spherical_loc = manifold.projx(self.spherical_loc)
        self.distributions = None

    def log_prob(self, z):
        raise NotImplementedError()


    def sample(self, n_samples, **kwargs):
        # Sample in 2d with the defined variance
        v = self.scale * self.distributions[0].sample(n_samples)[1] \
            * np.array([-1, 1]) + self.spherical_loc
        z = utils.TORUSsphericaltoeuclidean(v)

        return z

    def __hash__(self):
        return 0  # For jitting

    def frechet_mean(self):
        return self.loc


Torus_factor = 1.7


class StarTorus(DataProviderTorus):
    loc: None
    scale: None

    def __init__(self, manifold, loc=np.array([1, 1]), scale=None):
        super().__init__(manifold=manifold, loc=loc, scale=scale)
        self.distributions = [StarDataProvider(), ]


class HeartTorus(DataProviderTorus):
    loc: None
    scale: None

    def __init__(self, manifold, loc=np.array([1, 1]), scale=None):
        super().__init__(manifold=manifold, loc=loc, scale=scale)
        self.distributions = [HeartDataProvider(), ]


class ConditionalWrappedNormalTorus(Density):
    loc: np.ndarray

    def __init__(self, manifold, loc=None, min_cond=0.1, max_cond=1.0, **kwargs):
        super().__init__(manifold=manifold)
        assert isinstance(manifold, Torus)
        self.cond_size = 1
        self.loc = loc
        if not torch.is_tensor(self.loc):
            self.loc = torch.tensor(self.loc, dtype=float)
        if self.loc.shape[0] == 2:
            self.loc = utils.TORUSsphericaltoeuclidean(self.loc)

        self.normal_theta = ConditionalWrappedNormal(self.manifold.manifolds[0],
                                                     self.loc[:2],
                                                     min_cond, max_cond)
        self.normal_phi = ConditionalWrappedNormal(self.manifold.manifolds[1],
                                                   self.loc[2:],
                                                   min_cond, max_cond)
        self.min_cond = min_cond
        self.max_cond = max_cond

    def log_prob(self, z, scale):
        raise NotImplementedError()


    def sample(self, n_samples, cond=None, **kwargs):
        # assert cond is None or cond.shape[0] == self.cond_size
        if cond is None:
            # cond = torch.full((self.manifold.D - 1,), 0.3)
            cond = torch.rand(n_samples, 2) * (self.max_cond - self.min_cond) + \
                   self.min_cond
        elif not torch.is_tensor(cond):
            cond = np.tile(cond, (n_samples, 1))
            cond = torch.clamp(torch.tensor(cond, dtype=float),
                               min=self.min_cond, max=self.max_cond)
        v_theta, _ = self.normal_theta.sample(n_samples, cond)
        v_phi, _ = self.normal_phi.sample(n_samples, cond)
        z = torch.cat([v_theta, v_phi], dim=1)
        return z, cond

    def frechet_mean(self, cond=None):
        return self.loc


class ConditionalStarScaleTorus(StarTorus):
    def __init__(self, manifold, loc=None, min_cond=0.3, max_cond=2.0, **kwargs):
        super().__init__(manifold=manifold, loc=loc)
        self.cond_size = 1
        self.continous_cond = True

        self.min_cond = min_cond
        self.max_cond = max_cond
        
    def log_prob(self, z, cond, n_grid=30000):
        samples_grid = self.manifold.grid(n_grid)
        # compute kde to understand inside points
        samples_gt = self.sample(n_grid, cond)[0]
        test_kde = self.manifold.kde(samples_gt, samples_grid, bandwidth=0.01, norm=False)
        th = (test_kde.max() - test_kde.min()) * 0.5 + test_kde.min()
        test_kde_b = torch.where(test_kde < th, torch.zeros_like(test_kde), 
                            torch.full_like(test_kde, 1.0))
        area_factor = n_grid / (test_kde_b.sum() * self.manifold.area)
        
        loglh_gt = self.manifold.kde(samples_gt, z, bandwidth=0.01, norm=False)
        loglh_gt = torch.where(loglh_gt < th, torch.zeros_like(loglh_gt), 
                               torch.full_like(loglh_gt, area_factor))
        loglh_gt = torch.log(loglh_gt)

        return loglh_gt

    def sample(self, n_samples, cond=None, **kwargs):
        # assert cond is None or cond.shape[0] == self.cond_size
        if cond is None:
            cond = torch.rand(n_samples, 1) * (self.max_cond - self.min_cond) + \
                   self.min_cond
        elif not torch.is_tensor(cond):
            cond = torch.clamp(torch.tensor(cond, dtype=torch.float32),
                               min=self.min_cond, max=self.max_cond)

        # Sample in 2d with the defined variance
        cond_vec = cond.tile(n_samples, 2) * np.array([Torus_factor, 1])
        v = cond_vec * self.distributions[0].sample(n_samples)[
            1]
        v +=  self.spherical_loc.to(v.device)
        z = utils.TORUSsphericaltoeuclidean(v)
        return z, cond

    def frechet_mean(self, cond=None):
        return self.loc


class ConditionalHeartScaleTorus(HeartTorus):
    def __init__(self, manifold, loc=None, min_cond=0.3, max_cond=2.0, **kwargs):
        super().__init__(manifold=manifold, loc=loc)
        self.cond_size = 1
        self.continous_cond = True

        self.min_cond = min_cond
        self.max_cond = max_cond
        
    def log_prob(self, z, cond, n_grid=30000):
        samples_grid = self.manifold.grid(n_grid)
        # compute kde to understand inside points
        samples_gt = self.sample(n_grid, cond)[0]
        test_kde = self.manifold.kde(samples_gt, samples_grid, bandwidth=0.01, norm=False)
        th = (test_kde.max() - test_kde.min()) * 0.3 + test_kde.min()
        test_kde_b = torch.where(test_kde < th, torch.zeros_like(test_kde), 
                            torch.full_like(test_kde, 1.0))
        area_factor = n_grid / (test_kde_b.sum() * self.manifold.area)
        
        loglh_gt = self.manifold.kde(samples_gt, z, bandwidth=0.01, norm=False)
        loglh_gt = torch.where(loglh_gt < th, torch.zeros_like(loglh_gt), 
                               torch.full_like(loglh_gt, area_factor))
        loglh_gt = torch.log(loglh_gt)
        return loglh_gt

    def sample(self, n_samples, cond=None, train=False, **kwargs):
        # assert cond is None or cond.shape[0] == self.cond_size
        if cond is None:
            cond = torch.rand(n_samples, 1) * (self.max_cond - self.min_cond) + \
                   self.min_cond
        elif not torch.is_tensor(cond):
            cond = torch.clamp(torch.tensor(cond, dtype=torch.float32),
                               min=self.min_cond, max=self.max_cond)

        # Sample in 2d with the defined variance
        cond_vec = cond.tile(n_samples, 2) * np.array([-Torus_factor, 1])
        v = cond_vec * self.distributions[0].sample(n_samples)[1] + \
            self.spherical_loc
        z = utils.TORUSsphericaltoeuclidean(v)
        return z, cond

    def frechet_mean(self, cond=None):
        return self.loc
    

class RealDensityCond(Density):
    continous_cond = False

    def __init__(self, manifold, seed=42, **kwargs):
        self.manifold = manifold
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self.len_cond = None
        self.cond_keys = None

    def conds(self):
        return self.cond_keys

    
    def split_data(self, train=0.6, valid=0.2, test=0.2):
        assert train + valid + test == 1.0
        self.splits = {'train': dict(), 'val':  dict(), 'test':  dict()}
        # randomly choose the samples
        for k, v in self.dict_values.items():
            self._rng.shuffle(v)
            n = v.shape[0]
            n_train = int(n * train)
            n_valid = int(n * valid)
            n_test = n - n_train - n_valid
            self.splits["train"][k] = v[:n_train]
            self.splits["val"][k] = v[n_train:n_train + n_valid]
            self.splits["test"][k] = v[n_train + n_valid:]

    def log_prob(self, x):
        pass

    def _sample(self, n_samples, cond=0, mode=None):
        raise NotImplementedError()

    def sample(self, n_samples, cond=0, mode=None, train=False):
        n_samples = int(n_samples)
        return self._sample(n_samples, cond, mode)

    def collate_fn(self, batch):
        return  torch.utils.data.default_collate(batch)

    def cond_text(self, cond):
        return self.cond_keys[cond]


class ContinentalDrift(RealDensityCond):
    ordered_keys = ['600', '560', '540', '500', '470', '450', '430', '400',
                    '370', '340', '300', '280', '260', '240', '220', '200', '170',
                    '150', '120', '105', '90', '65', '50', '35', '20', 'pleist', ]

    def __init__(self, manifold, seed=42, **kwargs):
        assert isinstance(manifold, Sphere) and manifold.D == 3
        super().__init__(manifold=manifold, seed=seed, **kwargs)

        with open(f'data/cont_drift_noref.pkl', 'rb') as f:
            self.loaded_dict = pickle.load(f)
        self.len_cond = len(self.loaded_dict)
        self.cond_size = 1
        self.min_cond = 0
        self.max_cond = len(self.loaded_dict) - 1
        self.cond_keys = {i: k for i, k in enumerate(self.ordered_keys)}

        self.dict_values = {k: np.stack(np.nonzero(self.loaded_dict[k]), -1)
                           for k, v in self.loaded_dict.items()}  # latitute, longitude
        
        self.split_data()
    


    def _sample(self, n_samples, cond=0, mode=None):
        cond = int(np.clip(cond, self.min_cond, self.max_cond))
        # cond = 0
        if mode is not None:
            vals = self.splits[mode][self.cond_keys[cond]]
        else:
            vals = self.dict_values[self.cond_keys[cond]]
        h, w = self.loaded_dict[self.cond_keys[cond]].shape

        if n_samples > len(vals):
            lat_lon = vals.astype(float)  # latitute, longitude
        else:
            indx = self._rng.choice(len(vals), n_samples, replace=False)
            lat_lon = vals[indx].astype(float)  # latitute, longitude
        lat_lon = np.flip(lat_lon)
        # remap to range
        # latitude: [-pi/2, pi/2]
        lat_lon[:, 0] = (lat_lon[:, 0] * (np.pi / h) - np.pi / 2)
        # longitude: [-pi, pi]
        lat_lon[:, 1] = (lat_lon[:, 1] * (2 * np.pi / w) - np.pi) + np.pi
        xyz = utils.spherical_to_euclidean(lat_lon)

        return torch.tensor(xyz), cond  # .to(torch.int64)

    def frechet_mean(self, cond=None):
        cond = int(np.clip(cond, self.min_cond, self.max_cond))
        lat_lon = self.dict_values[self.cond_keys[cond]].mean(axis=0)
        h, w = self.loaded_dict[self.cond_keys[cond]].shape

        lat_lon[0] = (lat_lon[0] * (np.pi / h) - np.pi / 2)
        # longitude: [-pi, pi]
        lat_lon[1] = (lat_lon[1] * (2 * np.pi / w) - np.pi) + np.pi
        xyz = utils.spherical_to_euclidean(lat_lon).squeeze(0)

        return torch.tensor(xyz)


class Codon(RealDensityCond):

    def __init__(self, manifold, seed=42, **kwargs):
        self.cond_size = 1
        assert isinstance(manifold, Torus)
        super().__init__(manifold=manifold, seed=seed, **kwargs)

        df = pd.read_csv('data/data-precs.csv')
        # remove nan values
        df = df.dropna(subset=['phi', 'psi'])
        # generate dict of aa
        self.dict_values = df.groupby('name')[['phi', 'psi']].apply(
            lambda group: group.values.tolist()).to_dict()
        self.dict_values = {k: np.asarray(v) for k, v in self.dict_values.items()}
        self.cond_keys = list(self.dict_values.keys())
        self.len_cond = len(self.cond_keys)
        self.min_cond = 0
        self.max_cond = self.len_cond - 1
        self.split_data()


    def _sample(self, n_samples, cond=0, mode=None):
        cond = int(np.clip(cond, self.min_cond, self.max_cond))
        cond_key = self.cond_keys[cond]
        if mode is not None:
            vals = self.splits[mode][self.cond_keys[cond]]
        else:
            vals = self.dict_values[self.cond_keys[cond]]
        vals = np.deg2rad(vals)

        if n_samples > len(vals):
            pp = vals.astype(float)  # phi, psi
        else:
            indx = self._rng.choice(vals.shape[0], n_samples, replace=False)
            pp = vals[indx].astype(float)  # phi, psi

        # convert to inner representation
        pp = utils.TORUSsphericaltoeuclidean(pp)
        return torch.tensor(pp), cond

    def frechet_mean(self, cond=None):
        if cond is None:
            cond = 0
        cond = int(np.clip(cond, self.min_cond, self.max_cond))
        cond_key = self.cond_keys[cond]
        vals = np.deg2rad(self.dict_values[cond_key])
        return utils.TORUSsphericaltoeuclidean(np.mean(vals, axis=0))

from torch.utils.data import Dataset


class DensityDataset(Dataset):
    def __init__(self, density: Density, n_samples=None, n_conds=50,
                 th_dtype=torch.float32, device=torch.device("cpu"), mode="train"):
        self.density = density
        self.n_conds = n_conds
        self.n_samples = n_samples
        self.th_dtype = th_dtype
        self.device = device
        self.mode = mode

    def __getitem__(self, item):
        if self.density.continous_cond:
            item = random.uniform(self.density.min_cond, self.density.max_cond)
        samples, cond = self.density.sample(n_samples=self.n_samples,
                                            train=self.mode == "train",
                                             mode=self.mode, cond=item)
        return (torch.as_tensor(samples).to(self.th_dtype).to(self.device),
                torch.as_tensor(cond).to(self.device))

    def __len__(self):
        if self.density.continous_cond:
            return self.n_conds
        return self.density.len_cond


def get(manifold, name):
    if name == 'SphereBaseWrappedNormal':
        assert isinstance(manifold, Sphere)
        loc = manifold.zero()
        scale = torch.full((manifold.D - 1,), 0.3)
        return WrappedNormal(manifold=manifold, loc=loc, scale=scale)
    elif name == 'LouSphereSingleMode':
        assert isinstance(manifold, Sphere)
        loc = manifold.projx(-torch.ones(manifold.D))
        scale = torch.full((manifold.D - 1,), 0.3)
        return WrappedNormal(manifold=manifold, loc=loc, scale=scale)
    elif 'Earth' in name:
        try:
            name, year = name.split('_')
            return getattr(sys.modules[name], name)(manifold=manifold, year=year)
        except:
            print(f"Error loading data class {name}")
            raise
    else:
        try:
            return getattr(sys.modules[__name__], name)(manifold=manifold)
        except:
            print(f"Error loading data class {name}")
            raise


def get_uniform(manifold):
    if isinstance(manifold, Sphere):
        return SphereUniform(manifold=manifold)
    elif isinstance(manifold, Torus):
        return TorusUniform(manifold=manifold)

    else:
        assert False


def get_normal(manifold, loc=None, scale=None, cond=False, **kwargs):
    if cond:
        if isinstance(manifold, Sphere):
            return ConditionalWrappedNormal(manifold=manifold, loc=loc, **kwargs)
        if isinstance(manifold, Torus):
            return ConditionalWrappedNormalTorus(manifold=manifold, loc=loc, **kwargs)

    assert False


def get_density(name, manifold, **kwargs):
    if name == "uniform":
        return get_uniform(manifold)
    elif name == "normal":
        return get_normal(manifold, kwargs['loc'], kwargs['scale'])
    elif name == "rendez4":
        if isinstance(manifold,Sphere):
            return RezendeSphereFourMode(manifold, **kwargs)
    elif name == "star":
        if isinstance(manifold, Sphere):
            return Star(manifold, kwargs['loc'], kwargs['scale'])
        elif isinstance(manifold, Torus):
            return StarTorus(manifold, kwargs['loc'], kwargs['scale'])
        else:
            raise NotImplementedError(f"Unknown manifold {manifold}")
    elif name == "heart":
        if isinstance(manifold, Sphere):
            return Heart(manifold, kwargs['loc'], kwargs['scale'])
        elif isinstance(manifold, Torus):
            return HeartTorus(manifold, kwargs['loc'], kwargs['scale'])
        else:
            raise NotImplementedError(f"Unknown manifold {manifold}")
    elif name == "rezende":
        if isinstance(manifold, Sphere):
            return RezendeSphereFourMode(manifold)
    else:
        raise NotImplementedError(f"Unknown density {name}")


def get_CondDensity(name, manifold, **kwargs):
    if name == "normal":
        return get_normal(manifold, cond=True, **kwargs)
    elif name == "star":
        if isinstance(manifold, Sphere):
            return ConditionalStarScale(manifold, **kwargs)
        elif isinstance(manifold, Torus):
            return ConditionalStarScaleTorus(manifold, **kwargs)

        else:
            raise NotImplementedError(f"Unknown manifold {manifold}")
    elif name == "heart":
        if isinstance(manifold, Sphere):
            return ConditionalHeartScale(manifold, **kwargs)
        elif isinstance(manifold, Torus):
            return ConditionalHeartScaleTorus(manifold, **kwargs)


        else:
            raise NotImplementedError(f"Unknown manifold {manifold}")
    elif name == "contdrift":
        if isinstance(manifold, Sphere):
            return ContinentalDrift(manifold, **kwargs)
        else:
            raise NotImplementedError(f"Unknown manifold {manifold} for contdrift")
    elif name == "codon":
        if isinstance(manifold, Torus):
            return Codon(manifold, **kwargs)
        else:
            raise NotImplementedError(f"Unknown manifold {manifold} for codon")
    elif name == "rezende":
        if isinstance(manifold, Sphere):
            return RezendeSphereFourModeCond(manifold,**kwargs)
        if isinstance(manifold, Torus):
            return RezendeTorusFourModeCond(manifold,**kwargs)
    elif name == "rezende_tr":
        if isinstance(manifold, Sphere):
            return RezendeSphereFourModeTrasl(manifold,**kwargs)
        if isinstance(manifold, Torus):
            return RezendeTorusFourModeTrasl(manifold,**kwargs)
    else:
        raise NotImplementedError(f"Unknown density {name}")
