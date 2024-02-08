import math
import torch
import numpy as np
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.express.colors import sample_colorscale
from abc import ABC, abstractmethod
from scipy.stats import gaussian_kde

import rcpm.utils as utils

def acos_jitter(x, jitter=1e-4):
    epsilon = 1e-7
    x_clamp = torch.clamp(x, -1 + epsilon, 1 - epsilon)
    x_acos = torch.acos(x_clamp)

    return x_acos


latex_font_family = 'Times New Roman, Times, serif'

def rescale_matrix(matrix, a, b):
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    if min_val == max_val:
        # Handle the case where all values in the matrix are the same
        return np.full_like(matrix, (a + b) / 2)

    scaled_matrix = (matrix - min_val) * (b - a) / (max_val - min_val) + a
    return scaled_matrix


def rescale_tensor(tensor, a, b):
    min_val = tensor.min()
    max_val = tensor.max()

    if min_val == max_val:
        # Handle the case where all values in the tensor are the same
        return tensor.new_full(tensor.size(), (a + b) / 2)

    scaled_tensor = (tensor - min_val) * (b - a) / (max_val - min_val) + a
    return scaled_tensor


white_colormap = [[0, 'rgb(220, 220, 220)'], [1, 'rgb(225, 225, 225)']]


@dataclass
class Manifold(ABC):
    D: int  # Dimension of the ambient Euclidean space
    max_dist = 1.
    min_dist = 0.0
    kde_factor = 0.05
    name = "Manifold"
    int_method = "interpolation"
    area=1.0

    def __init__(self, D, kde_factor=0.1, max_dist=1., min_dist=0.0):
        self.D = D
        self.kde_factor = kde_factor
        self.max_dist = max_dist
        self.min_dist = min_dist

    @abstractmethod
    def exponential_map(self, x, v):
        pass

    @abstractmethod
    def log(self, x, y):
        '''
        x: (N, D) center of the log
        y: (N, D) point on the manifold to log
        '''
        pass

    @abstractmethod
    def tangent_projection(self, x, v):
        pass

    @abstractmethod
    def projx(self, x):
        pass

    @abstractmethod
    def dist(self, x, y):
        pass

    @abstractmethod
    def cost(self, x, y):
        pass

    # They use it just for the normalizing flows
    def tangent_orthonormal_basis(self, x, dF):
        pass

    def grad(self, pot_fn, x, vmap=True, chunk_size=None):
        if vmap:
            x = x.clone().detach().requires_grad_(True)
            dF_func = torch.func.grad(pot_fn)
            dF = torch.vmap(dF_func, randomness='error', chunk_size=chunk_size)(x)
            dF = self.tangent_projection(x, dF)
        else:
            dF = list()
            for i in range(x.shape[0]):
                with torch.enable_grad():
                    out = torch.autograd.functional.jacobian(pot_fn,
                                                             x[i, :]).squeeze(0)
                    out = self.tangent_projection(x[i, :], out)
                dF.append(out)
            dF = torch.concat(dF, dim=0)

        return dF
        
    def hessian(self, pot_fn, x, vmap=True, chunk_size=None):
        if vmap:
            x = x.clone().detach().requires_grad_(True)
            dF_func = torch.func.hessian(pot_fn)
            dF = torch.vmap(dF_func, randomness='error', chunk_size=chunk_size)(x)
            dF = self.tangent_projection(x, dF)
        else:
            dF = list()
            for i in range(x.shape[0]):
                with torch.enable_grad():
                    out = torch.autograd.functional.jacobian(pot_fn,
                                                                x[i, :]).squeeze(0)
                    out = self.tangent_projection(x[i, :], out)
                dF.append(out)
            dF = torch.concat(dF, dim=0)

        return dF


    @abstractmethod
    def grid(self, n=1000):
        '''
        n: number of points on the grid
        return: (n, D) tensor
        '''
        pass

    def kde(self, x, grid_points=None, bandwidth=None, kernel=None, x_vals=None,norm=True):
        if grid_points is None:
            # grid_points = torch.tensor(self.grid())
            grid_points = self.grid().clone().detach()
        grid_points = grid_points.to(x.device).to(x.dtype)
        # compute the distance matrix
        dist_matrix = self.dist(x, grid_points)

        # compute bandwidth using Scott's rule of thumb if not provided
        if bandwidth is None:
            bandwidth = self.kde_factor
        if bandwidth == 'scott':
            n = dist_matrix.size(0)
            d = dist_matrix.size(1)
            bandwidth = torch.std(dist_matrix) * (n ** (-1 / (d + 4)))

        # compute the kernel density estimate
        if kernel is None or kernel == 'gaussian':
            kernel = lambda x: torch.exp(-x)
        kde = kernel(dist_matrix / bandwidth)
        if x_vals is not None:
            kde = torch.sum(kde * x_vals.unsqueeze(1), dim=0) / torch.sum(kde, dim=0)
        else:
            kde = torch.mean(kde, dim=0) / bandwidth
        # Rescale the array to positive values that sum to 1
        if norm:
            kde = (kde - torch.min(kde)) / (torch.max(kde) - torch.min(kde))
            kde /= torch.sum(kde)

        return kde

    def distrbution_kde(self, model_samples, surf_samples=None, kde_factor=None):
        '''
        model_samples: (N, D)
        surf_samples: (N, D)

        '''
        heatmap = self.kde(model_samples, surf_samples, kde_factor)
        return heatmap

    def kde_loss(self, target_samples, gt_samples, surf_samples=None,
                 kde_factor=None):
        '''
        target_samples: (N, D)
        gt_samples: (N, D)
        surf_samples: (N, D)
        '''
        if surf_samples is None:
            surf_samples = self.grid().clone().detach()
        kde_target = self.kde(target_samples, surf_samples, kde_factor)
        kde_gt = self.kde(gt_samples, surf_samples, kde_factor)
        return torch.mean(torch.abs(kde_target - kde_gt))

    def log_geodesic(self, x, y):
        xy = (x * y).sum(dim=-1, keepdim=True)
        xy = torch.clamp(xy, min=-1 + 1e-6, max=1 - 1e-6)
        val = torch.acos(xy)
        return (y - xy * x) / torch.sin(val)

    def geodesic_path(self, x, y, n=1000, max_path=1.0):
        logx_y = self.log(x, y)

        t = torch.linspace(0, max_path, n)

        return self.exponential_map(x, t[:, None] * logx_y)

    @abstractmethod
    def geodesic_contours(self, x, d, n=100, error=1e-3, experimental=True):
        '''
        x: center of the contour
        d: list of distance of the contours from the center as values between 0 and 1
        n: number of points on the contour
        error: error tolerance for the contour
        '''
        pass

    def prob_dist_calibration(manifold, center, d, experimental=True, degree=3):
        as_tensor = isinstance(center, torch.Tensor)
        if not as_tensor:
            center = torch.Tensor(center)
        if d is None:
            if experimental:
                dist_ratios = manifold.interpolate_contourProb(np.linspace(1e-6, 1, d,
                                                                           endpoint=False),
                                                               degree=degree)
            else:
                dist_ratios = np.linspace(1e-6, 1, d, endpoint=False)
        else:
            # rescale d to be in [0, pi]
            if experimental:
                dist_ratios = manifold.interpolate_contourProb(np.array(d),
                                                               degree=degree)
            else:
                dist_ratios = np.array(d)
            # phis = np.array([np.pi * num for num in d])
        dist_ratios = dist_ratios * manifold.max_dist
        return center, dist_ratios

    def coverage_test_geodesic(manifold, samples, center,
                               quantile_probs=10, empirical=True,
                               degree=3):
        ''' Given a set of points on the manifold, compute their coverages
        on the set of quantile levels using the geoodesic distances.
        samples: list of samples from the target distribution
        center: center of the contours
        quantile_levels: quantile levels as list of ints or
                        number of geodesic levels to compute as int
        return coverage of each quantile level
        '''
        center = manifold.projx(center)
        if isinstance(quantile_probs, int):
            quantile_probs = np.linspace(0.1, 1, quantile_probs,
                                         endpoint=False)
        if empirical:
            dist_ratios = manifold.interpolate_contourProb(quantile_probs,
                                                           degree=degree)
        else:
            dist_ratios = quantile_probs
        # compute the geodesic distances from the center
        geodesic_dists = manifold.dist(center, samples) / manifold.max_dist  # (1, N, )
        # compute the coverages for each quantile level
        coverages = {q: (geodesic_dists <= d).type(torch.float).mean()
                     for q, d in zip(quantile_probs, dist_ratios)}

        return coverages

    def interpolate_contourProb(self, x, degree=3):
        data = np.load(f"data/{self.name}.npz")
        prob = data["x"]
        d_ratio = data["y"]

        if self.int_method == "linear":
            coefficients = np.polyfit(prob, d_ratio, deg=degree)
            f = np.poly1d(coefficients)
        if self.int_method == "interpolation":
            return np.interp(x, prob, d_ratio, left=0, right=1)
        elif self.int_method == "barycentric":
            from scipy.interpolate import BarycentricInterpolator
            f = BarycentricInterpolator(prob, d_ratio)
        elif self.int_method == "lagrange":
            from scipy.interpolate import lagrange
            f = lagrange(prob, d_ratio)
        elif self.int_method == "spline":
            from scipy.interpolate import UnivariateSpline
            f = UnivariateSpline(prob, d_ratio, k=degree)
        elif self.int_method == "pchip":
            from scipy.interpolate import PchipInterpolator
            prob, unique_indices = np.unique(prob, return_index=True)
            d_ratio = d_ratio[unique_indices]
            f = PchipInterpolator(prob, d_ratio)
        elif self.int_method == "akima":
            from scipy.interpolate import Akima1DInterpolator
            prob, unique_indices = np.unique(prob, return_index=True)
            d_ratio = d_ratio[unique_indices]
            f = Akima1DInterpolator(prob, d_ratio)
        elif self.int_method == "cubic":
            from scipy.interpolate import CubicSpline
            prob, unique_indices = np.unique(prob, return_index=True)
            d_ratio = d_ratio[unique_indices]
            f = CubicSpline(prob, d_ratio)
        elif self.int_method == "cubichermitian":
            from scipy.interpolate import CubicHermiteSpline
            prob, unique_indices = np.unique(prob, return_index=True)
            d_ratio = d_ratio[unique_indices]
            dx = np.gradient(prob)  # Compute the spacing between x points
            dy = np.gradient(d_ratio)  # Compute the spacing between y points
            derivatives = dy / dx  # Compute the derivative as dy/dx
            f = CubicHermiteSpline(prob, d_ratio, derivatives)

        return f(x)


## 3d plot functions

def surface3d(self):
    pass


def to_3d(self, samples):
    pass


def plot_samples_3d(self, samples, save=None, colors=None, title="",
                    show=False, subplots_titles="", use_heatmap=True,
                    heatmap_vals=None,
                    show_samples=True, marker_size=2, kde_factor=None,
                    kde_colormap="Oranges", samples_colors='rgb(128, 128, 128)',
                    showgrid=False, camera_eye=None, save_pdf=False, **kwargs):

    coords3d, coords = self.surface3d()
    x = coords3d[:, :, 0].numpy()
    y = coords3d[:, :, 1].numpy()
    z = coords3d[:, :, 2].numpy()

    fig = make_subplots(
        rows=1, cols=len(samples),
        subplot_titles=subplots_titles,
        specs=[[{'type': 'scene'} for i in range(len(samples))]],
        horizontal_spacing=0.02,)
    scene_dict = dict(showbackground=showgrid, showticklabels=showgrid,
                      showgrid=showgrid, showaxeslabels=showgrid,
                      showline=showgrid, zeroline=showgrid)
    if camera_eye is None:
        camera_eye = (1.5, 1.5, 1.5)
    else:
        camera_eye = camera_eye # self.to_3d(camera_eye)

    fig.update_scenes(dict(
        camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2])),
        xaxis=scene_dict, yaxis=scene_dict, zaxis=scene_dict
    ))
    if not use_heatmap:
        kde_colormap = white_colormap
    fig.update_layout(coloraxis=dict(colorscale=kde_colormap))

    fig.layout.title = f"{self.name} " + title
    for i_c in range(len(samples)):
        fig['layout']['scene' + str(i_c + 1)].aspectmode = "data"

    for i_v in range(len(samples)):
        if use_heatmap:
            try:
                coords_shape = coords.shape
                heatmap = self.kde(samples[i_v],
                                coords.reshape(-1, coords_shape[-1]),
                                bandwidth=kde_factor,
                                x_vals=heatmap_vals[i_v] if heatmap_vals is not None else None).reshape(
                    coords_shape[0], coords_shape[1])
            except Exception as e:
                print(e)
                heatmap = x
        else:
            heatmap = x 
            fig.layout.coloraxis.colorscale = "Greys"
        xyz_samples = self.to_3d(samples[i_v])
        traces = [go.Surface(x=x, y=y, z=z,
                             # colorscale="Greys",
                             coloraxis="coloraxis", surfacecolor=heatmap,
                             showscale=False, showlegend=False)]
        if show_samples:
            traces += [go.Scatter3d(x=xyz_samples[:, 0], y=xyz_samples[:, 1],
                                    z=xyz_samples[:, 2], mode='markers',
                                    showlegend=False,
                                    marker=go.scatter3d.Marker(color=samples_colors,
                                                               size=marker_size,
                                                               colorscale="Peach",
                                                               showscale=False))
                       ]
        fig.add_traces(traces, rows=1, cols=i_v + 1)
    if show: fig.show()
    if save is not None:
        fig.write_html(save + ".html")
        if save_pdf:
            fig.write_image(save + ".pdf", width=2560, height=1600)
    return fig


def plot_contours_3d(self, contours, phis=None, target_samples=None, save=None,
                     show=False, title="", show_heatmap=True, show_samples=True,
                     contour_colormap="Rainbow", kde_colormap="Greys",
                     samples_colors='rgb(128, 128, 128)', showgrid=False,
                     camera_eye=None, save_pdf=False):
    coords3d, coords = self.surface3d()
    x = coords3d[:, :, 0].numpy()
    y = coords3d[:, :, 1].numpy()
    z = coords3d[:, :, 2].numpy()

    fig = make_subplots(
        rows=1, cols=len(contours),
        subplot_titles=list(contours.keys()),
        specs=[[{'type': 'scene'} for i in range(len(contours))]])
    scene_dict = dict(showbackground=showgrid, showticklabels=showgrid,
                      showgrid=showgrid, showaxeslabels=showgrid,
                      showline=showgrid, zeroline=showgrid)
    if camera_eye is None:
        camera_eye = (1.5, 1.5, 1.5)
    elif camera_eye.shape[0] != 3:
        camera_eye = self.to_3d(camera_eye)
    fig.update_scenes(dict(
        camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2])),
        xaxis=scene_dict, yaxis=scene_dict, zaxis=scene_dict
    ))
    fig.update_layout(coloraxis2=dict(colorscale=contour_colormap,
                                      colorbar=dict(len=0.5, y=0.6), cmin=0,
                                      cmax=1))
    for i_c in range(len(contours)):
        fig['layout']['scene' + str(i_c + 1)].aspectmode = "data"
    fig.layout.title = f"Contours {self.name} " + title

    use_heatmap = target_samples is not None and show_heatmap
    if not use_heatmap:
        kde_colormap = white_colormap
    fig.update_layout(
        coloraxis=dict(colorscale=kde_colormap, colorbar=dict(len=0.5, y=0)))
    for i_c, k_c in enumerate(contours):
        if phis is None:
            phis = np.arange(len(contours[k_c]))

        if use_heatmap:
            try:
                coords_shape = coords.shape
                heatmap = self.kde(target_samples[i_c],
                                   coords.reshape(-1, coords_shape[-1])).reshape(
                    coords_shape[0], coords_shape[1])
            except Exception as e:
                print(e)
                heatmap = x
        else:
            heatmap = x

        xyz_contours = [self.to_3d(contours[k_c][i_cc])
                        for i_cc in range(len(contours[k_c]))]
        traces = [go.Surface(x=x, y=y, z=z, coloraxis="coloraxis",
                             surfacecolor=heatmap,
                             showscale=False, showlegend=False)]
        if show_samples and target_samples is not None:
            xyz_samples = self.to_3d(target_samples[i_c])
            traces += [go.Scatter3d(x=xyz_samples[:, 0], y=xyz_samples[:, 1],
                                    z=xyz_samples[:, 2], mode='markers',
                                    showlegend=False,
                                    marker=go.scatter3d.Marker(
                                        color=samples_colors, size=1.5,
                                        showscale=False))]

        traces += [go.Scatter3d(x=xyz_contours[i_cc][:, 0],
                                y=xyz_contours[i_cc][:, 1],
                                z=xyz_contours[i_cc][:, 2],
                                mode='markers', showlegend=k_c == 0,
                                name=f"{phis[i_cc]:.1f}",
                                marker=go.scatter3d.Marker(
                                    color=np.repeat(phis[i_cc],
                                                    contours[k_c][i_cc].shape[0]),
                                    size=2.5, coloraxis="coloraxis2",
                                    showscale=False))
                   for i_cc in range(len(contours[k_c]))]
        fig.add_traces(traces, rows=1, cols=1 + i_c)

    fig.update_layout(showlegend=True,
                      font=dict(family=latex_font_family),
                      title_font_family=latex_font_family,
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                                  itemsizing='constant', itemwidth=30,
                                  font=dict(
                                      family=latex_font_family, size=12, color="black"),
                                  bgcolor="White", bordercolor="Black",
                                  borderwidth=1, ))
    if show: fig.show()
    if save is not None:
        fig.write_html(save + ".html")
        if save_pdf:
            fig.write_image(save + ".pdf", width=2560, height=1600)
    return fig


####################

eps = 1e-5 
divsin = lambda x: x / torch.sin(x)
sindiv = lambda x: torch.sin(x) / (x + eps)
divsinh = lambda x: x / torch.sinh(x)
sinhdiv = lambda x: torch.sinh(x) / (x + eps)


def lorentz_cross(x, y):
    z = torch.cross(x, y)
    z[..., 0] = -z[..., 0]
    return z

@dataclass
class Sphere(Manifold):
    jitter: float = 1e-4
    max_dist = np.pi
    name = 'S2'
    area = 4 * np.pi

    NUM_POINTS: int = 100

    def __init__(self, D=3, jitter=1e-3, kde_factor='scott', max_dist=np.pi,
                 min_dist=0.0):
        super().__init__(D, kde_factor, max_dist, min_dist)
        self.jitter = jitter
        self.theta = torch.linspace(0, 2 * np.pi, self.NUM_POINTS)
        self.phi = torch.linspace(0, np.pi, self.NUM_POINTS)
        self.tp = torch.tensor(np.asarray(
            np.meshgrid(self.theta, self.phi, indexing='ij')))
        # tp = tp.transpose([1, 2, 0]).reshape(-1, 2)
        self.tp = self.tp.moveaxis([0, 1, 2], [2, 0, 1]).reshape(-1, 2)

    def plot_samples_3d(self, *args, **kwargs):
        return plot_samples_3d(self, *args, **kwargs)

    def plot_contours_3d(self, *args, **kwargs):
        return plot_contours_3d(self, *args, **kwargs)

    def exponential_map(self, x, v):
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        # v_norm = torch.clamp(v_norm, min=1e-16)
        v_norm = v_norm + 1e-16
        exp_m = x * torch.cos(v_norm) + v * torch.sin(v_norm) / (v_norm)
        exp_m_norm = torch.norm(exp_m, dim=-1, keepdim=True)
        # exp_m_norm = torch.clamp(exp_m_norm, min=1e-16)
        exp_m_norm = exp_m_norm + 1e-16
        exp_m = exp_m / exp_m_norm
        return exp_m

    def log_geodesic(self, x, y):
        xy = (x * y).sum(dim=-1, keepdim=True)
        xy = torch.clamp(xy, min=-1 + 1e-6, max=1 - 1e-6)
        val = torch.acos(xy)
        return (y - xy * x) / torch.sin(val)

    def log(self, x, y):
        if x.ndim == 1:
            x = x[None, :]
        if y.ndim == 1:
            y = y[None, :]
        inner = x @ y.T
        tang = (y - inner.T * x.repeat(y.shape[0], 1))
        tang_norm = torch.norm(tang, dim=-1, keepdim=True)
        # tang_norm = torch.clamp(tang_norm, min=1e-16)
        mask = (tang_norm > 0).squeeze()
        tang[mask, :] = tang[mask, :] / tang_norm[mask, :]
        # inner = torch.clamp(inner.T, min=-1 + 1e-6, max=1 - 1e-6)
        inner = inner / (1 + self.jitter)
        tang = (torch.acos(inner)) * tang
        return tang

    def tangent_projection(self, x, u):
        '''
            x: center of the tangent space, (N, D) or (D,)
            u: vectors to be projected, (N, D) or (D,)
        '''
        if x.ndim == 1:
            x = x[None, :]
        if u.ndim == 1:
            u = u[None, :]
        proj_u = u - x * torch.sum(x * u, dim=1, keepdim=True)
        return proj_u

    def tangent_orthonormal_basis(self, x, dF):
        '''
            x: center of the tangent space, (N, D) 
            dF: vectors to be projected, (N, D)
            return: (N, D-1)
        '''
        assert x.ndim == 2

        if x.shape[1] == 2:
            E = x[:, torch.tensor([1, 0])] * torch.tensor([-1., 1.], dtype=x.dtype, device=x.device)
            E = E.reshape(*E.shape, 1)
        elif x.shape[1] == 3:
            norm_v = dF / torch.norm(dF, dim=-1, keepdim=True)
            E = torch.stack((norm_v, 
               torch.cross(x, norm_v, dim=-1)), dim=-1)
        else:
            raise NotImplementedError()

        return E

    def dist(self, x, y):
        inner = x @ y.T
        is_tensor = torch.is_tensor(inner)
        if not is_tensor:
            inner = torch.tensor(inner)
        d = acos_jitter(inner, self.jitter)

        if not is_tensor:
            d = d.numpy()
        return d

    def cost(self, x, y):
        return self.dist(x, y) ** 2 / 2.

    def projx(self, x):
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x = x / (x_norm + 1e-8)
        return x

    def transp(self, x, y, u):
        yu = torch.sum(y * u, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        return u - yu / (1 + xy) * (x + y)

    def logdetexp(self, x, u):
        norm_u = torch.norm(u, dim=-1)
        val = torch.log(torch.abs(torch.sin(norm_u)))
        return (u.shape[-1] - 2) * val

    def zero(self):
        y = torch.zeros(self.D)
        y[..., 0] = -1.
        return y

    def zero_like(self, x):
        y = torch.zeros_like(x)
        y[..., 0] = -1.
        return y

    def squeeze_tangent(self, x):
        return x[..., 1:]

    def unsqueeze_tangent(self, x):
        return torch.cat((torch.zeros_like(x[..., :1]), x), dim=-1)

    def geodesic_contours(self, center, d=None, n=100, error=1e-3, experimental=False,
                          degree=3, max_iter=100, verbose=False):
        center, dist_ratios = self.prob_dist_calibration(center, d,
                                                         experimental=experimental,
                                                         degree=degree)
        if self.D == 2:
            # S1
            theta_center = utils.S1euclideantospherical(center)

            contours = list()
            for phi in dist_ratios:
                theta_cont = [theta_center - phi, theta_center + phi]
                cont = np.stack([utils.S1sphericaltoeuclidean(theta_cont[0]),
                                 utils.S1sphericaltoeuclidean(theta_cont[1])], 0)
                contours.append(cont)
        elif self.D == 3:
            # S2
            R = utils.rotation3D_matrix(center, np.array([0, 0, 1.0]))
            contours = list()
            for phi in dist_ratios:
                cont = torch.stack([torch.linspace(0, 2 * np.pi, n),
                                    torch.tensor([phi]).repeat(n)], 1)
                cont = utils.spherical_to_euclidean(cont)
                contours.append(cont @ R)
        else:
            raise NotImplementedError("Only S1 and S2 are supported")
        return contours

    def plot_samples(self, model_samples, kde_factor=0.1, save='t.png', show=False):
        estimated_density = gaussian_kde(
            utils.euclidean_to_spherical(model_samples).T, kde_factor)
        heatmap = estimated_density(self.tp.T).reshape(
            2 * self.NUM_POINTS, self.NUM_POINTS)
        self.plot_mollweide(heatmap, save=save, show=show)

    def distrbution_kde(self, model_samples, surf_samples=None, kde_factor=None):
        '''
        model_samples: (N, D)
        surf_samples: (N, D)

        '''
        heatmap = self.kde(model_samples, surf_samples, kde_factor)
        return heatmap

    def grid(self, n=None):
        if self.D == 3:
            n = self.NUM_POINTS if n is None else int(n)
            return utils.fibonacci_sphere(n)
        if n is None:
            n = self.NUM_POINTS
            u = self.theta
            v = self.phi
        else:
            u = torch.linspace(0, 2 * np.pi, math.ceil(math.sqrt(n)))
            v = torch.linspace(0, np.pi, math.ceil(math.sqrt(n)))
        x = torch.flatten(torch.outer(torch.cos(u), torch.sin(v)))
        y = torch.flatten(torch.outer(torch.sin(u), torch.sin(v)))
        z = torch.flatten(torch.outer(torch.ones_like(u), torch.cos(v)))
        xyz = torch.stack([x, y, z], dim=1)
        if len(xyz) > n:
            xyz = xyz[np.random.randint(0, len(xyz), n)]
        return xyz

    def surface3d(self, n=None):
        if n is None:
            u = self.theta
            v = self.phi
        else:
            u = torch.linspace(0, 2 * np.pi, n)
            v = torch.linspace(0, np.pi, n)
        x = torch.outer(torch.cos(u), torch.sin(v))
        y = torch.outer(torch.sin(u), torch.sin(v))
        z = torch.outer(torch.ones_like(u), torch.cos(v))

        xyz = torch.stack([x, y, z], dim=2)

        return xyz, xyz

    @staticmethod
    def _show_markers(points, **kwargs):
        longitudes, latitudes = np.split(utils.euclidean_to_spherical(points), 2,
                                         -1)
        longitudes = np.degrees(longitudes.squeeze(1))
        latitudes = -np.degrees((latitudes.squeeze(1) - np.pi / 2))

        trace = go.Scattergeo(lat=latitudes, lon=longitudes,
                              mode='markers', **kwargs)
        return trace

    @staticmethod
    def _show_lines(points, points0, **kwargs):
        longitudes, latitudes = utils.euclidean_to_spherical(points)
        longitudes0, latitudes0 = utils.euclidean_to_spherical(points0)

        longitudes = np.degrees(longitudes)
        latitudes = np.degrees(latitudes - np.pi / 2)
        longitudes0 = np.degrees(longitudes0)
        latitudes0 = np.degrees(latitudes0 - np.pi / 2)

        lons = np.empty(3 * len(longitudes0))
        lons[::3] = longitudes0.numpy()
        lons[1::3] = longitudes.numpy()
        lons[2::3] = None
        lats = np.empty(3 * len(latitudes))
        lats[::3] = latitudes0.numpy()
        lats[1::3] = latitudes.numpy()
        lats[2::3] = None

        trace = [go.Scattergeo(lat=latitudes.numpy(), lon=longitudes.numpy(),
                               mode='markers', **kwargs),
                 go.Scattergeo(lat=lats, lon=lons,
                               mode='lines', line=dict(width=1, color='red'),
                               opacity=0.5)]
        return trace

    @staticmethod
    def plot_mollewide(samples, samples0=None, samples_colors='rgb(128, 128, 128)',
                       title="", subplots_titles="", canonical_rotation=torch.eye(3),
                       scatter_size=2, opacity=1.0, show_colorscale=False, cmin=None,
                       save_pdf=False,
                       show=True, save=None, width=1980, height=1200):
        ''' Plot the distribution on SO3 as in "Mollewide projection:
        a spherical projection suitable for visualizing distributions of
        unit vectors
        :param rotations: [B, 4] rotations as quaterions
        '''

        fig = make_subplots(
            rows=1, cols=len(samples),
            subplot_titles=subplots_titles,
            specs=[[{'type': 'scattergeo'} for i in range(len(samples))]],
             horizontal_spacing = 0.02)
        fig.update_layout(coloraxis=dict(colorscale="Peach"))
        if cmin is not None:
            fig.update_layout(coloraxis=dict(cmin=cmin))

        for i_v in range(len(samples)):
            traces = []
            xyz_samples = torch.as_tensor(samples[i_v])
            if isinstance(samples_colors, list):
                colors = samples_colors[i_v]
                
                customdata=colors
                hovertemplate='Lat: %{lat:.2f}°; Lon: %{lon:.2f}°<br>Val: %{customdata:.2f}<extra></extra>'
            else:
                colors = samples_colors
                customdata=None
                hovertemplate='Lat: %{lat:.2f}°; Lon: %{lon:.2f}°<extra></extra>'

            # xyz_samples = canonical_rotation @ xyz_samples
            if samples0 is None:
                traces.append(Sphere._show_markers(xyz_samples,
                                                   marker=go.scattergeo.Marker
                                                   (color=colors,
                                                    size=scatter_size,
                                                    coloraxis="coloraxis",
                                                    # colorscale="Peach",
                                                    showscale=show_colorscale),
                                                   customdata=customdata,
                                                   hovertemplate=hovertemplate))
            else:
                traces += Sphere._show_lines(xyz_samples,
                                             marker=go.scattergeo.Marker(
                                                 color=colors,
                                                 size=scatter_size,
                                                 coloraxis="coloraxis",
                                                 showscale=show_colorscale))
            fig.add_traces(traces, rows=1, cols=i_v + 1)

        if isinstance(samples_colors, list): 
            bgcolor = sample_colorscale("Peach", [0])[0]
        else:
            bgcolor = 'rgba(0, 0, 0, 0)'
        # normalize col_widths to sum to 1
        fig.update_geos(projection_type="mollweide", visible=False, showframe=True,
                        lataxis_showgrid=True, lataxis_gridwidth=0.5,
                        lonaxis_showgrid=True, lonaxis_gridwidth=0.5,
                        framecolor=bgcolor,)
        fig.update_layout(width=width, height=height, title=title,
                          margin={"autoexpand":True})

        if show: fig.show()
        if save is not None:
            fig.write_html(save + ".html")
            if save_pdf:
                fig.write_image(save + ".pdf")
        return fig

    def plot_kde_mollewide(self, samples, title="", subplots_titles="",
                           n_points=int(2e4), kde_factor=None, scatter_size=int(2e5),
                          show_samples=False, samples_colors='rgb(128, 128, 128)',
                          show=True, save=None, width=1980, height=1200):
        fig = make_subplots(
            rows=1, cols=len(samples),
            subplot_titles=subplots_titles,
            specs=[[{'type': 'scattergeo'} for i in range(len(samples))]])

        kde_points = self.grid(n_points)
        if kde_factor is None:
            kde_factor = self.kde_factor

        for i_v in range(len(samples)):
            traces = []
            xyz_samples = torch.as_tensor(samples[i_v])
            heatmap = self.kde(xyz_samples, kde_points,
                               bandwidth=kde_factor)
            traces.append(self._show_markers(kde_points,
                                             marker=go.scattergeo.Marker
                                             (color=heatmap, opacity=0.9,
                                              size=scatter_size / n_points,
                                              colorscale="Peach",
                                              showscale=False)))
            if show_samples:
                traces.append(self._show_markers(xyz_samples,
                                             marker=go.scattergeo.Marker
                                             (opacity=0.7,
                                              size=1.2,color=samples_colors,
                                              showscale=False)))

            fig.add_traces(traces, rows=1, cols=i_v + 1)

        # normalize col_widths to sum to 1
        fig.update_geos(projection_type="mollweide", visible=False, showframe=True,
                        lataxis_showgrid=True, lataxis_gridwidth=0.5,
                        lonaxis_showgrid=True, lonaxis_gridwidth=0.5, )
        fig.update_layout(width=width, height=height, title=title,
                          margin={"autoexpand":True})

        if show: fig.show()
        if save is not None:
            fig.write_html(save + ".html")
        return fig

    @staticmethod
    def plot_contour_mollewide(contours, phis=None, target_samples=None,
                               samples_colors='rgb(128, 128, 128)',
                               contour_colormap="Rainbow",
                               title="", subplots_titles="",
                               center=None,
                               scatter_size=2,
                               show=True, save=None, width=1980, height=1200):
        ''' Plot the distribution on SO3 as in "Mollewide projection:
        a spherical projection suitable for visualizing distributions of
        unit vectors
        :param rotations: [B, 4] rotations as quaterions
        '''

        fig = make_subplots(
            rows=1, cols=len(contours),
            subplot_titles=subplots_titles,
            specs=[[{'type': 'scattergeo'} for i in range(len(contours))]])
        fig.update_layout(coloraxis2=dict(colorscale=contour_colormap,
                                          colorbar=dict(len=0.5, y=0.6), cmin=0,
                                          cmax=1))

        for i_c, k_c in enumerate(contours):
            if phis is None:
                phis = np.arange(len(contours[k_c]))
            traces = []
            if target_samples is not None:
                traces += [Sphere._show_markers(torch.as_tensor(target_samples[i_c]),
                                                marker=go.scattergeo.Marker
                                                (color=samples_colors,
                                                 size=scatter_size,
                                                 showscale=False)), ]
            traces += [Sphere._show_markers(torch.as_tensor(contours[k_c][i_cc]),
                                            opacity=0.7,
                                            marker=go.scattergeo.Marker
                                            (color=np.repeat(phis[i_cc],
                                                             contours[k_c][i_cc].shape[
                                                                 0]),
                                             size=scatter_size,
                                             coloraxis="coloraxis2",
                                             showscale=False))
                       for i_cc in range(len(contours[k_c]))]
            if center is not None:
                traces += [Sphere._show_markers(torch.as_tensor(center[k_c]),
                                                marker=go.scattergeo.Marker
                                                (color='red', size=15, symbol='x',
                                                 showscale=False)), ]
            fig.add_traces(traces, rows=1, cols=i_c + 1)

        # normalize col_widths to sum to 1
        fig.update_geos(projection_type="mollweide", visible=False, showframe=True,
                        lataxis_showgrid=True, lataxis_gridwidth=0.5,
                        lonaxis_showgrid=True, lonaxis_gridwidth=0.5, )
        fig.update_layout(width=width, height=height,
                          margin={"r": 0, "t": 0, "l": 0, "b": 0},
                          title=f"Contours " + title)
        if show: fig.show()
        if save is not None:
            fig.write_html(save + ".html")
        return fig


    def to_3d(self, samples):
        return samples

    def coverage_test_geodesic(manifold, samples, center,
                               quantile_probs=10, empirical=False,
                               degree=3):
        return super().coverage_test_geodesic(samples, center,
                                              quantile_probs, empirical, degree)


class Euclidean(Manifold):
    def exponential_map(self, x, v):
        return x + v

    def tangent_projection(self, x, u):
        return u

    def cost(self, x, y):
        return 0.5 * self.dist(x, y) ** 2

    def dist(self, x, y):
        return - torch.matmul(x, y)

    def tangent_orthonormal_basis(self, x, dF):
        tang_vecs = [torch.eye(x.shape[1]) for i in range(x.shape[0])]
        return torch.stack(tang_vecs, 0)


def get(manifold):
    if manifold == 'S1':
        return Sphere(D=2)
    elif manifold == 'S2':
        return Sphere(D=3)
    elif manifold == 'R':
        return Euclidean(D=1)
    else:
        assert False


@dataclass
class Product(Manifold):
    manifolds_str: str = 'S1,S1'
    kde_factor = 1.0

    def __post_init__(self):
        self.manifolds = []
        for man in self.manifolds_str.split(','):
            self.manifolds.append(get(man))
        self.D = sum([man.D for man in self.manifolds])

        d = 0
        max_dist = 0
        for man in self.manifolds:
            max_dist += man.max_dist ** 2
            d = d + man.D
        self.max_dist = np.sqrt(max_dist)

    def exponential_map(self, x, v):
        exp_prod = []
        d = 0
        for man in self.manifolds:
            exp_man = man.exponential_map(x[:, d:d + man.D], v[:, d:d + man.D])
            exp_prod.append(exp_man)
            d = d + man.D
        exp_prod = torch.cat(exp_prod, dim=1)
        return exp_prod

    def log(self, x, v):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if v.ndim == 1:
            v = v.unsqueeze(0)

        log_prod = []
        d = 0
        for man in self.manifolds:
            log_map = man.log(x[:, d:d + man.D], v[:, d:d + man.D])
            log_prod.append(log_map)
            d = d + man.D
        log_prod = torch.cat(log_prod, dim=1)
        return log_prod

    def tangent_projection(self, x, u):
        proj_prod = []
        d = 0
        for man in self.manifolds:
            proj_man = man.tangent_projection(x[:, d:d + man.D], u[:, d:d + man.D])
            proj_prod.append(proj_man)
            d = d + man.D
        proj_prod = torch.cat(proj_prod, dim=1)
        return proj_prod

    def cost(self, x, y):
        cost_prod = torch.zeros([x.shape[0], y.shape[0]], device=x.device,
                                dtype=x.dtype)
        d = 0
        for man in self.manifolds:
            cost_man = man.cost(x[:, d:d + man.D], y[:, d:d + man.D])
            cost_prod = cost_man + cost_prod
            d = d + man.D
        return cost_prod

    def dist(self, x, y):
        dist_prod = 0  # torch.zeros([x.shape[0], y.shape[0]])
        d = 0
        for man in self.manifolds:
            dist_prod += man.dist(x[:, d:d + man.D], y[:, d:d + man.D]) ** 2
            d = d + man.D
        return torch.sqrt(dist_prod)

    def tangent_orthonormal_basis(self, x, dF):
        d = 0
        blocks = []
        for man in self.manifolds:
            onb_man = man.tangent_orthonormal_basis(x[:, d:d + man.D],
                                                    dF[:, d:d + man.D])
            blocks.append(onb_man)
            d = d + man.D
        
        onb = torch.vmap(torch.block_diag)(*(blocks))
        return onb

    def projx(self, x):
        x_proj = []
        d = 0
        for man in self.manifolds:
            x_proj_man = man.projx(x[:, d:d + man.D])
            d = d + man.D
            x_proj.append(x_proj_man)
        x_proj = torch.cat(x_proj, dim=1)
        return x_proj

    def grid(self, n=1000):
        grids = []
        grid = None
        d = 0
        n_root = int(n ** (1 / len(self.manifolds)))
        for man in self.manifolds:
            grids.append(man.grid(n_root))
            d = d + man.D
            if grid is not None:
                grid = torch.cartesian_prod(grid, grids[-1])
            else:
                grid = grids[-1]
        return grid

    def plot_samples(self, model_samples, save='t.png'):
        pass

    def plot_density(self, log_prob_fn, save='t.png'):
        pass


@dataclass
class Torus(Product):
    '''
    Torus manifold. points are parametrized as 4d vectors (x,y,z,w) with
    x,y,z,w in [-1,1]
    '''
    manifolds_str: str = 'S1,S1'
    name = 'Torus'
    area = 4 * np.pi ** 2

    NUM_POINTS = 160

    theta = torch.linspace(0, 2 * np.pi, NUM_POINTS)
    phi = torch.linspace(0, 2 * np.pi, NUM_POINTS)
    tp = torch.Tensor(np.asarray(np.meshgrid(theta, phi, indexing='ij')))
    tp = tp.transpose(1, 2).transpose(0, 1).reshape(-1, 2)
    max_dist = 4.3797

    def __init__(self, kde_factor=1.0):
        super().__init__(manifolds_str='S1,S1', D=4)
        self.kde_factor = kde_factor

    def plot_samples_3d(self, *args, **kwargs):
        return plot_samples_3d(self, *args, **kwargs)

    def plot_contours_3d(self, *args, **kwargs):
        return plot_contours_3d(self, *args, **kwargs)

    def exponential_map(self, x, v):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if v.ndim == 1:
            v = v.unsqueeze(0)
        spherical = x.shape[1] == 2
        if spherical:
            x = torch.concat((utils.S1sphericaltoeuclidean(x[:, 0]),
                              utils.S1sphericaltoeuclidean(x[:, 1])), 1)  # b x 4
            v = torch.concat((utils.S1sphericaltoeuclidean(v[:, 0]),
                              utils.S1sphericaltoeuclidean(v[:, 1])), 1)
        exp_map = (self.manifolds[0].exponential_map(x[:, :2], v[:, :2]),
                   self.manifolds[1].exponential_map(x[:, 2:], v[:, 2:]))
        if spherical:
            exp = torch.stack((utils.S1euclideantospherical(exp_map[0]),
                               utils.S1euclideantospherical(exp_map[1])), 1)
        else:
            exp = torch.cat(exp_map, dim=1)
        return exp

    def tangent_projection(self, x, u):
        if x.ndim == 1:
            x = x[None, :]
        if u.ndim == 1:
            u = u[None, :]
        spherical = x.shape[1] == 2
        if spherical:
            x = torch.concat((utils.S1sphericaltoeuclidean(x[:, 0]),
                              utils.S1sphericaltoeuclidean(x[:, 1])), 1)
            u = torch.concat((utils.S1sphericaltoeuclidean(u[:, 0]),
                              utils.S1sphericaltoeuclidean(u[:, 1])), 1)
        proj = super().tangent_projection(x, u)
        if spherical:
            proj = torch.stack((utils.S1euclideantospherical(proj[:, :2]),
                                utils.S1euclideantospherical(proj[:, 2:])), 1)
        return proj

    def cost(self, x, y):
        spherical = x.shape[1] == 2
        if spherical:
            x = torch.concat((utils.S1sphericaltoeuclidean(x[:, 0]),
                              utils.S1sphericaltoeuclidean(x[:, 1])), 1)
            y = torch.concat((utils.S1sphericaltoeuclidean(y[:, 0]),
                              utils.S1sphericaltoeuclidean(y[:, 1])), 1)
        # call super cost with x and y
        cost = super().cost(x, y)
        return cost

    def dist(self, x, y):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        spherical = x.shape[1] == 2
        if spherical:
            x = torch.concat((utils.S1sphericaltoeuclidean(x[:, 0]),
                              utils.S1sphericaltoeuclidean(x[:, 1])), 1)
            y = torch.concat((utils.S1sphericaltoeuclidean(y[:, 0]),
                              utils.S1sphericaltoeuclidean(y[:, 1])), 1)
        # call super dist with x and y
        dist = super().dist(x, y)
        return dist

    def projx(self, x):
        batched = x.ndim == 2
        if not batched:
            x = x.unsqueeze(0)
        spherical = x.shape[1] == 2
        if spherical:
            x = torch.concat((utils.S1sphericaltoeuclidean(x[:, 0]),
                              utils.S1sphericaltoeuclidean(x[:, 1])), 1)
        # call super projx with x
        projx = super().projx(x)
        if spherical:
            projx = torch.stack((utils.S1euclideantospherical(projx[:, :2]),
                                 utils.S1euclideantospherical(projx[:, 2:])), 1)
        if not batched:
            projx = projx.squeeze(0)
        return projx

    def geodesic_path(self, x, y, n=1000, max_path=1.0):
        # fix the opposite point problem
        if (x[:2] == torch.abs(y[:2])).any():
            x[:2] += 1e-6
        if (x[2:] == torch.abs(y[2:])).any():
            x[2:] += 1e-6
        return super().geodesic_path(x, y, n, max_path)

    def geodesic_contours(self, center, d, n=100, error=1e-3):
        '''
        center: starting point on the manifold [4]
        d: distance ratios, can be a list or a number
        n: number of points to sample
        '''
        as_tensor = isinstance(center, torch.Tensor)
        if not as_tensor:
            center = torch.Tensor(center)
        u = torch.linspace(0, 2 * np.pi, n)
        v = torch.linspace(0, 2 * np.pi, n)
        u = u.repeat(n, 1).flatten()
        v = v.repeat(n, 1).transpose(0, 1).flatten()

        eucl = utils.TORUSsphericaltoeuclidean(torch.stack((u, v), 1))
        eucl = torch.cat((center[np.newaxis, :], eucl), 0)
        D = self.dist(center[np.newaxis, :], eucl).flatten()
        # rescale D values between 0 and 1
        D = (D - D.min()) / (D.max() - D.min())
        # D = (D - self.min_dist) / (self.max_dist - self.min_dist)

        contours = list()
        for d_ in d:
            # find the entries of D that are within error of d_
            idx = torch.where(torch.abs(D - d_) < error)[0]
            contours.append(eucl[idx, :])
        return contours

    def grid(self, n=100):
        n_samples = math.ceil(n ** (1 / 2))
        u = torch.linspace(0, 2 * np.pi, n_samples)
        v = torch.linspace(0, 2 * np.pi, n_samples)
        u = u.repeat(v.shape[0], 1)
        v = v.repeat(u.shape[0], 1).transpose(0, 1)
        points = utils.TORUSsphericaltoeuclidean(
            torch.stack((u.flatten(), v.flatten()), dim=1))
        # select randomly n points
        if points.shape[0] > n:
            idx = torch.linspace(0, points.shape[0] - 1, n, dtype=int)
            points = points[idx, :]
        return points

    def surface3d(self, n=None, R=1, r=0.4):
        if n is None:
            u = self.theta
            v = self.phi
        else:
            u = torch.linspace(0, 2 * np.pi, n)
            v = torch.linspace(0, 2 * np.pi, n)
        u = u.repeat(v.shape[0], 1)
        v = v.repeat(u.shape[0], 1).transpose(0, 1)
        points = utils.TORUSsphericaltoeuclidean(
            torch.stack((u.flatten(), v.flatten()), dim=1)).reshape(u.shape[0],
                                                                    u.shape[1], -1)

        x, y, z = utils.productS1toTorus(u.flatten(), v.flatten(), R, r)
        x = x.reshape(u.shape)
        y = y.reshape(u.shape)
        z = z.reshape(u.shape)
        return torch.stack([x, y, z], dim=2), points

    def distrbution_kde(self, model_samples, spherical_samples=None,
                        kde_factor='scott'):
        spherical = model_samples.shape[1] == 2
        if spherical:
            thetas = model_samples
        else:
            thetas = np.stack((utils.S1euclideantospherical(model_samples[:, :2]),
                               utils.S1euclideantospherical(model_samples[:, 2:])),
                              axis=1)
        estimated_density = gaussian_kde(thetas.T, kde_factor)
        if spherical_samples is None:
            spherical_samples = self.tp
        else:
            if spherical_samples.shape[1] == 4:
                spherical_samples = np.stack(
                    (utils.S1euclideantospherical(spherical_samples[:, :2]),
                     utils.S1euclideantospherical(
                         spherical_samples[:, 2:])),
                    axis=1)
        heatmap = estimated_density(spherical_samples.T)

        return heatmap

    def to_3d(self, samples):
        batched = samples.ndim == 2
        if not batched:
            samples = samples[np.newaxis, :]
        if samples.shape[1] == 2:
            xyz_samples = utils.productS1toTorus(samples[:, 0],
                                                 samples[:, 1])
        else:
            xyz_samples = utils.S1eucltoTorus(samples)
        xyz_samples = np.stack(xyz_samples, axis=1)
        if not batched:
            xyz_samples = xyz_samples[0]
        return xyz_samples

    def plot_flat(self, samples, save=None, title="",
                  show=False, subplots_titles="", cmin=None,
                  show_hist=False, marker_size=2, show_colorscale=False,
                  kde_colormap="Peach", samples_colors='rgb(128, 128, 128)',
                  showgrid=False, save_pdf=False, opacity=1.0,
                  width=None, height=None):
        fig = make_subplots(
            rows=1, cols=len(samples),
            subplot_titles=subplots_titles, shared_yaxes=True,
            specs=[[{'type': 'xy'} for i in range(len(samples))]])
        fig.update_layout(autosize=False, width=450 * len(samples), height=450)
        scene_dict = dict(showbackground=showgrid, showticklabels=showgrid,
                          showgrid=showgrid, showaxeslabels=showgrid,
                          showline=showgrid, zeroline=showgrid)

        fig.update_scenes(dict(xaxis=scene_dict, yaxis=scene_dict))
        fig.update_layout(coloraxis=dict(colorscale=kde_colormap))
        if cmin is not None:
            fig.update_layout(coloraxis=dict(cmin=cmin))

        epsilon = 1e-1
        fig.update_xaxes(range=[-np.pi - epsilon, np.pi + epsilon],
                         showgrid=False, showline=False, zeroline=False)
        fig.update_yaxes(range=[-np.pi - epsilon, np.pi + epsilon],
                         showgrid=False, showline=False, zeroline=False)

        fig.layout.title = f"{self.name} " + title
        fig.layout.scattermode = "overlay"
        fig['layout']['scene'].aspectratio = {"x": 1, "y": 1}
        fig["layout"]['scene'].aspectmode = "data"

        for i_v in range(len(samples)):
            # check if samples_colors is a list or a vector or a string
            
            if isinstance(samples_colors, list):
                colors = samples_colors[i_v]
                customdata=colors
                hovertemplate='Phi: %{x:.2f}°; Psi: %{y:.2f}°<br>Val: %{customdata:.2e}<extra></extra>'
            else:
                colors = samples_colors
                customdata=None
                hovertemplate='Phi: %{x:.2f}°; Psi: %{y:.2f}°<br><extra></extra>'
            

            samples_pp = utils.TORUSeuclideantospherical(samples[i_v])
            # remap the values to be in the range [-pi, pi] using the modulo
            samples_pp = samples_pp % (2 * np.pi)
            samples_pp = samples_pp - 2 * np.pi * (samples_pp > np.pi)

            traces = [go.Scatter(x=samples_pp[:, 0], y=samples_pp[:, 1],
                                 showlegend=False, mode='markers',
                                 xaxis='x', yaxis='y', opacity=0.7,
                                 marker=go.scatter.Marker(color=colors,
                                                          size=marker_size,
                                                          coloraxis="coloraxis",
                                                          showscale=show_colorscale),
                                 customdata=customdata,
                                 hovertemplate=hovertemplate)]
            if show_hist:
                traces += [go.Histogram2dContour(x=samples_pp[:, 0], y=samples_pp[:, 1],
                                                 coloraxis="coloraxis",
                                                 xaxis='x', yaxis='y',
                                                 contours_coloring='heatmap', )
                           ]
            fig.add_traces(traces, rows=1, cols=i_v + 1)
        if isinstance(samples_colors, list): 
            bgcolor = sample_colorscale("Peach", [0])[0]
        else:
            bgcolor = 'rgba(0, 0, 0, 0)'
        fig.update_layout(plot_bgcolor=bgcolor,)
        fig.update_xaxes( 
               tickwidth=2, griddash='dash', tickcolor="black", 
               ticks="inside", 
               tickvals=[-np.pi,0,np.pi], 
                    ticktext=[r"180", "0", r"180"], 
               tickfont=dict( size=14),
               )
        fig.update_yaxes( 
                    tickwidth=2, griddash='dash', tickcolor="black", 
                    ticks="inside", 
                    tickvals=[-np.pi,0,np.pi],
                     ticktext=[r"180", "0", r"180"], 
                    tickfont=dict( size=14),
                    )
        fig.add_hline(y=0, line_width=0.3, line_dash="dash", line_color="black")
        fig.add_vline(x=0, line_width=0.3, line_dash="dash", line_color="black")
        if width is not None:
            fig.update_layout(width=width)
        if height is not None:
            fig.update_layout(height=height)
        if show: fig.show()
        if save is not None:
            fig.write_html(save + ".html")
            if save_pdf:
                fig.write_image(save + ".pdf")
        return fig

    def plot_kde_flat(self, samples, save=None, title="", n_points=int(4e4),
                      show=False, subplots_titles="",
                      scatter_size=int(5), kde_factor=0.3, show_samples=False,
                      kde_colormap="Peach", samples_colors='rgb(128, 128, 128)',
                      save_pdf=False, width=1200, height=700):
        fig = make_subplots(
            rows=1, cols=len(samples),
            subplot_titles=subplots_titles, shared_yaxes=True,
            specs=[[{'type': 'xy'} for i in range(len(samples))]])
        fig.update_layout(autosize=False, width=450 * len(samples), height=450)
        scene_dict = dict(showgrid=False, showline=False, zeroline=False)
        fig.update_scenes(dict(xaxis=scene_dict, yaxis=scene_dict))
        fig.update_layout(coloraxis=dict(colorscale=kde_colormap),
                          plot_bgcolor='rgba(0, 0, 0, 0)',
                          paper_bgcolor='rgba(0, 0, 0, 0)',
                          )
        epsilon = 1e-1
        fig.update_xaxes(range=[-np.pi - epsilon, np.pi + epsilon],
                         showgrid=False, showline=False, zeroline=False)
        fig.update_yaxes(range=[-np.pi - epsilon, np.pi + epsilon],
                         showgrid=False, showline=False, zeroline=False)

        fig.layout.title = f"{self.name} " + title
        fig.layout.scattermode = "overlay"
        fig['layout']['scene'].aspectratio = {"x": 1, "y": 1}

        kde_points = self.grid(n_points)
        if kde_factor is None:
            kde_factor = self.kde_factor
        kde_pp = utils.TORUSeuclideantospherical(kde_points)
        # remap the values to be in the range [-pi, pi] using the modulo
        kde_pp = kde_pp % (2 * np.pi)
        kde_pp = kde_pp - 2 * np.pi * (kde_pp > np.pi)

        for i_v in range(len(samples)):
            traces = []

            heatmap = self.kde(samples[i_v], kde_points,
                               bandwidth=kde_factor)
            # check if it sums to 1
            traces.append(go.Scatter(x=kde_pp[:, 0], y=kde_pp[:, 1],
                                     showlegend=False, mode='markers',
                                     xaxis='x', yaxis='y', opacity=0.9,
                                     marker=go.scatter.Marker(color=heatmap,
                                                              symbol='square',
                                                              size=scatter_size,
                                                              coloraxis="coloraxis",
                                                              showscale=True)))
            if show_samples:
                samples_pp = utils.TORUSeuclideantospherical(samples[i_v])
                # remap the values to be in the range [-pi, pi] using the modulo
                samples_pp = samples_pp % (2 * np.pi)
                samples_pp = samples_pp - 2 * np.pi * (samples_pp > np.pi)

                traces += [go.Scatter(x=samples_pp[:, 0], y=samples_pp[:, 1],
                                    showlegend=False, mode='markers',
                                    xaxis='x', yaxis='y', opacity=0.5,
                                    marker=go.scatter.Marker(color=samples_colors,
                                                            size=0.6,))]

            fig.add_traces(traces, rows=1, cols=i_v + 1)

        fig.update_xaxes( 
               tickwidth=2, griddash='dash', tickcolor="black", 
               ticks="inside", 
               tickvals=[-np.pi,0,np.pi], 
                    ticktext=[r"180", "0", r"180"], 
               tickfont=dict( size=14),
               )
        fig.update_yaxes( 
                    tickwidth=2, griddash='dash', tickcolor="black", 
                    ticks="inside", 
                    tickvals=[-np.pi,0,np.pi],
                     ticktext=[r"180", "0", r"180"], 
                    tickfont=dict( size=14),
                    )
        fig.add_hline(y=0, line_width=0.3, line_dash="dash", line_color="black")
        fig.add_vline(x=0, line_width=0.3, line_dash="dash", line_color="black")
        if show: fig.show()
        if save is not None:
            fig.write_html(save + ".html")
            if save_pdf:
                fig.write_image(save + ".pdf", width=2560, height=1600)
        return fig

    @staticmethod
    def map2pp(samples):
        samples_pp = utils.TORUSeuclideantospherical(samples)
        # remap the values to be in the range [-pi, pi] using the modulo
        samples_pp = samples_pp % (2 * np.pi)
        samples_pp = samples_pp - 2 * np.pi * (samples_pp > np.pi)
        return samples_pp

    def plot_contour_flat(self, contours, phis=None, target_samples=None,
                          title="", save=None,
                          show=False, subplots_titles="", marker_size=2,
                          contour_colormap="Rainbow", center=None,
                          kde_colormap="Oranges", samples_colors='rgb(128, 128, 128)',
                          showgrid=False, save_pdf=False):
        fig = make_subplots(
            rows=1, cols=len(contours),
            subplot_titles=subplots_titles, shared_yaxes=True,
            specs=[[{'type': 'xy'} for i in range(len(contours))]])
        fig.update_layout(autosize=False, width=450 * len(contours), height=450)
        scene_dict = dict(showbackground=showgrid, showticklabels=showgrid,
                          showgrid=showgrid, showaxeslabels=showgrid,
                          showline=showgrid, zeroline=showgrid)

        fig.update_scenes(dict(xaxis=scene_dict, yaxis=scene_dict))
        fig.update_layout(coloraxis=dict(colorscale=kde_colormap))
        fig.update_layout(coloraxis2=dict(colorscale=contour_colormap,
                                          colorbar=dict(len=0.5, y=0.6), cmin=0,
                                          cmax=1))
        epsilon = 1e-1
        fig.update_xaxes(range=[-np.pi - epsilon, np.pi + epsilon], )
        fig.update_yaxes(range=[-np.pi - epsilon, np.pi + epsilon], )

        fig.layout.title = f"{self.name} " + title
        fig.layout.scattermode = "overlay"
        fig['layout']['scene'].aspectratio = {"x": 1, "y": 1}

        for i_c, k_c in enumerate(contours):
            if phis is None:
                phis = np.arange(len(contours[k_c]))

            contour_pp = [self.map2pp(contours[k_c][i_cc])
                          for i_cc in range(len(contours[k_c]))]

            traces = list()

            if target_samples is not None:
                samples_pp = self.map2pp(target_samples[i_c])
                traces += [go.Scatter(x=samples_pp[:, 0], y=samples_pp[:, 1],
                                      showlegend=False, mode='markers',
                                      xaxis='x', yaxis='y', opacity=0.7,
                                      marker=go.scatter.Marker(color=samples_colors,
                                                               size=marker_size,
                                                               colorscale="Peach",
                                                               showscale=False)),
                           go.Histogram2dContour(x=samples_pp[:, 0], y=samples_pp[:, 1],
                                                 colorscale="Peach",
                                                 xaxis='x', yaxis='y', showscale=False,
                                                 contours_coloring='heatmap', )
                           ]

            traces += [go.Scatter(x=contour_pp[i_cc][:, 0], y=contour_pp[i_cc][:, 1],
                                  showlegend=k_c == 0, mode='markers',
                                  xaxis='x', yaxis='y',
                                  name=f"{phis[i_cc]:.1f}",
                                  marker=go.scatter.Marker(color=np.repeat(phis[i_cc],
                                                                           contours[
                                                                               k_c][
                                                                               i_cc].shape[
                                                                               0]),
                                                           size=marker_size,
                                                           coloraxis="coloraxis2",
                                                           showscale=False))
                       for i_cc in range(len(contours[k_c]))]
            if center is not None:
                center_pp = self.map2pp(center[k_c])
                traces += [go.Scatter(x=[center_pp[0]], y=[center_pp[1]],
                                      mode='markers', xaxis='x', yaxis='y',
                                      name=f"center",
                                      marker=go.scatter.Marker(size=12, color='red',
                                                               symbol='x',
                                                               showscale=False))]

            fig.add_traces(traces, rows=1, cols=i_c + 1)
        fig.update_layout(showlegend=True,  
                      font=dict(family=latex_font_family),
                      title_font_family=latex_font_family,
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                                      itemsizing='constant', itemwidth=30,
                                      font=dict(
                                          family=latex_font_family, size=12, color="black"),
                                      bgcolor="White", bordercolor="Black",
                                      borderwidth=1, ))

        if show: fig.show()
        if save is not None:
            fig.write_html(save + ".html")
            if save_pdf:
                fig.write_image(save + ".pdf", width=2560, height=1600)
        return fig



def get_manifold(manifold, **kwargs):
    if manifold == 'S1':
        return Sphere(D=2, **kwargs)
    elif manifold == 'S2':
        return Sphere(D=3, **kwargs)
    elif manifold == 'R':
        return Euclidean(D=1, **kwargs)
    elif manifold == 'Torus':
        return Torus(**kwargs)
    else:
        assert False

