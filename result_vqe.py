import argparse
import json
import os
import numpy as np
import torch

from rcpm.manifolds import get_manifold
from rcpm.densities import get_density
from vq_models import ManifoldVQESingle, ManifoldVQEMulti



def json_load(path):
    with open(path, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
    return t_args

def kde_plot(samples_gt, samples_est, title=None, save=None):
    print("Plotting kde")

    f = manifold.plot_samples_3d(
        [samples_gt, samples_est, ],
        title=title, save=save,
        show=False,
        show_samples=False,
        subplots_titles=["target distribution", f"Estimated", ])
    return f


def kde_flat(manifold, samples_gt, samples_est, title=None, save=None, kde_factor=None):
    print("Plotting kde")
    f = None
    if manifold.name in ["S2", ]:
        # mollewide
        f = manifold.plot_kde_mollewide([samples_gt, samples_est, ],
                                        title=title, save=save, kde_factor=kde_factor,
                                        subplots_titles=["target distribution",
                                                         f"Estimated", ],
                                        show=False)
    elif manifold.name in ["Torus", ]:
        # flat torus
        f = manifold.plot_kde_flat([samples_gt, samples_est, ],
                                   title=title, save=save, kde_factor=kde_factor,   
                                   subplots_titles=["target distribution",
                                                    f"Estimated", ],
                                   show=False)

    return f

def likelihood_plot(manifold, vqe, results_dir, N_sample=10000,
                    marker_size=5, eps=5e-2, chunk_size=1500):
    print("Plotting likelihood")
    samples_grid_conj = manifold.grid(N_sample).to(vqe.th_dtype)
    # samples_grid = manifold.grid(20000).to(vqe.th_dtype)
    lh = vqe.likelihood(U_sampled=samples_grid_conj, eps=eps,
                        chunk_size=chunk_size).cpu().detach()
    if manifold.name in ["S2", ]:
        # mollewide
        f = manifold.plot_mollewide([samples_grid_conj, ], samples_colors=lh,
                                        title="Likelihood", 
                                        scatter_size=marker_size, opacity=0.7,
                                        save=os.path.join(results_dir, f'likelihood'),
                                        show=False)
    elif manifold.name in ["Torus", ]:
        # flat torus
        f = manifold.plot_kde_flat([samples_grid_conj, ], samples_colors=lh,
                                   title="Likelihood", 
                                   marker_size=marker_size, opacity=0.7,
                                        save=os.path.join(results_dir, f'likelihood'),
                                        show=False)
    return f

camera_dist = {"S2": 1.6, "Torus": 0.6}
camera_pos = {"S2": np.array([0, 0, 0]), "Torus": np.array([1, 1, 1.5])}


def parse_params():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-m", "--manifold", type=str, default="S2")
    parser.add_argument('-d', "--dir", type=str, default="rezende1")
    parser.add_argument("-t", type=float, default=1.)
    parser.add_argument("--tests",
                        default=[ 'samples', "kde_loss", 'kde_plot', 'contours', 'inverse', 'coverage' ,
                            ],
                        nargs='+')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--kde-factor", type=float, default=0.1)
    parser.add_argument("--best", action='store_true', default=False)


    return parser.parse_args()


if __name__ == "__main__":
    args_test = parse_params()
    tests = args_test.tests
    t_exp = args_test.t
    N_sample = 10000
    results_dir = os.path.join("results", "vqe", args_test.manifold, args_test.dir)
    print(results_dir)
    args = json_load(os.path.join(results_dir, "params.json"))

    device = torch.device(args_test.device) 
    th_dtype = torch.float32

    manifold = get_manifold(args.manifold, kde_factor=args.kde_factor)

    base = get_density(args.base, manifold=manifold)
    if not hasattr(args, 'psi_model') or args.psi_model == "discrete":
        if args.n_layers == 1:
            vqe = ManifoldVQESingle(manifold, base_density=base,
                                    n_components=args.n_components,
                                    init_alpha_mode=args.init_alpha_mode,
                                    init_alpha_linear_scale=args.init_alpha_linear_scale,
                                    init_alpha_minval=args.init_alpha_minval,
                                    init_alpha_range=args.init_alpha_range,
                                    init_points=args.init_points,
                                    fixed_points=args.fixed_points,
                                    cost_gamma=args.cost_gamma,
                                    min_zero_gamma=args.min_zero_gamma,
                                    eps=args.eps,
                                    device=device, th_dtype=th_dtype)
        else:
            vqe = ManifoldVQEMulti(manifold, base_density=base, n_layers=args.n_layers,
                                   n_components=args.n_components,
                                   init_alpha_mode=args.init_alpha_mode,
                                   init_alpha_minval=args.init_alpha_minval,
                                   init_alpha_range=args.init_alpha_range,
                                   init_alpha_linear_scale=args.init_alpha_linear_scale,
                                   cost_gamma=args.cost_gamma,
                                   min_zero_gamma=args.min_zero_gamma,
                                   fixed_points=args.fixed_points,
                                   init_points=args.init_points,
                                   eps=args.eps,
                                   device=device, th_dtype=th_dtype)
    else:
        raise NotImplementedError

    state_dict =  torch.load(os.path.join(results_dir, 'best_state_dict.pth' if args_test.best
                                          else 'state_dict.pth'),
                            map_location={str(args.device): str(device)})
    vqe.load_state_dict(state_dict["model_state_dict"])
    target = state_dict["target_distr"]
    base = state_dict["base_distr"]
    vqe.eval()

    base_samples = base.sample(N_sample)
    eval_target_samples = target.sample(N_sample)
    samples_gt = target.sample(N_sample)
    samples_est_random = vqe.sample(N_sample, base_sample=base_samples).cpu().detach()
    tau_contours = np.linspace(0.1, 1, 10, endpoint=False)
    tau_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .8, .9]
    base_pole = None
    est_base_samples = None

    # Plot 1: Samples
    if "samples" in tests:
        manifold.plot_samples_3d(
            [base_samples, samples_est_random, samples_gt],
            save=os.path.join(args.results_dir, f'result_test'), show=True,
            camera_eye=np.asarray(args.target_loc) * 1.5,
            subplots_titles=["base distribution",
                             f"EST: Randomly sampled on uniform grid.",
                             "target distribution"])
        manifold.plot_samples_3d(
            [samples_est_random, samples_gt],
            save=os.path.join(results_dir, f'samples'), show=True,
            subplots_titles=[f"EST: Randomly sampled on uniform grid.",
                             "target distribution"])

    if "cconvex" in tests:
        print("Plot cconvex")
        vqe.plot_cconvex(save=os.path.join(args.results_dir, f'cconvex'))

    if "kde_loss" in tests:
        kde_loss = manifold.kde_loss(samples_gt, samples_est_random)
        samples_gt2 = target.sample(N_sample)
        kde_loss_gt = manifold.kde_loss(samples_gt, samples_gt2)
        print("kde_loss: ", kde_loss, "kde_loss_gt: ", kde_loss_gt)
        with open(os.path.join(results_dir, "kde_loss.txt"), "wt") as f:
            f.write(f"{kde_loss}; {kde_loss_gt}\n")
    
    if "kde_plot" in tests:
        print("Plot kde")
        kde_plot(samples_gt, samples_est_random, title="",
                     save=os.path.join(results_dir, f'result_kde'))
        kde_flat(manifold, samples_gt, samples_est_random,  title="",
                     save=os.path.join(results_dir, f'kde_flat'), 
                     kde_factor=args_test.kde_factor)
    
    if "likelihood" in tests:
        lh = likelihood_plot(manifold, vqe,
                            results_dir,marker_size=5, eps=args_test.eps)


    # Plot inverse map
    if "inverse" in tests:

        est_base_samples = vqe.sample_inverse(eval_target_samples,
                                              n_base=int(5e3)).cpu().detach()
        manifold.plot_samples_3d(
            [eval_target_samples, est_base_samples, base_samples],
            subplots_titles=["target", "est_base", "GT_base"],
            save=os.path.join(results_dir, f'inverse_sample'), show=False)

    # Plot contours from frechet_mean
    if "contours" in tests and args.manifold in ["S2", "Torus"]:
        if base.is_uniform and base_pole is None:
            base_pole = vqe.sample_inverse(target.frechet_mean().view(1, -1))[0,
                        :].cpu().detach()
        else:
            base_pole = None
        contours_base = manifold.geodesic_contours(base_pole, tau_contours, 1000)
        contours_target = [vqe.sample(base_sample=cb).cpu().detach() for cb in
                           contours_base]
        camera_eye = (manifold.to_3d(base_pole) + camera_pos[manifold.name]) * \
                     camera_dist[manifold.name]
        manifold.plot_contours_3d({"base": contours_base, "target": contours_target},
                                  tau_contours,
                                  target_samples=[base_samples, eval_target_samples],
                                  save=os.path.join(results_dir, f'contour'),
                                  show=False, camera_eye=camera_eye.numpy(),
                                  show_heatmap=True, save_pdf=False)

    # plot contour_theta_betha on base
    if "contour_theta_betha" in tests and base.is_uniform:
        contours_base = [base.contour_theta(n=1000, theta=np.pi / 2),
                         base.contour_theta(n=1000, theta=np.pi / 3),
                         base.contour_phi(n=1000, phi=np.pi / 2),
                         base.contour_phi(n=1000, phi=np.pi / 3)
                         ]
        contours_target = [vqe.sample(base_sample=contour).cpu().detach()
                           for contour in contours_base]
        manifold.plot_contours_3d({"base": contours_base, "target": contours_target},
                                  target_samples=[base_samples, eval_target_samples],
                                  camera_eye=base_pole.numpy() *
                                             camera_dist[manifold.name],
                                  save=os.path.join(results_dir, f'contour_theta_phi'),
                                  show=False)


    # coverage test
    if "coverage" in tests:
        if base.is_uniform and base_pole is None:
            base_pole = vqe.sample_inverse(target.frechet_mean().unsqueeze(0)).cpu().detach()
        samples_gt_cov = target.sample(N_sample, mode="test")
        est_base_samples = vqe.sample_inverse(samples_gt_cov).cpu().detach()
        coverages = manifold.coverage_test_geodesic(est_base_samples, base_pole,
                                                    tau_contours, empirical=True)
        print("coverages: ", coverages)
        error = np.abs(np.array(list(coverages.keys())) -
                       np.array(list(coverages.values())))
        print("error: ", error)
        # mean and std error
        print("mean error: ", error.mean())
        print("std error: ", error.std())
        with open(os.path.join(results_dir, "coverage.txt"), "wt") as f:
            f.write(f"{coverages}\n")
            f.write(f"{error}\n")
            f.write(f"mean error: {error.mean()}\n")
            f.write(f"std error: {error.std()}\n")
