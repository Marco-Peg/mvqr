import json
import os
import numpy as np
import torch
import argparse

from rcpm.manifolds import get_manifold
from rcpm.densities import get_density
from vq_models import ManifoldVQESingle, ManifoldVQEMulti


def parse_params():
    parser = argparse.ArgumentParser(description='Train vqe on sphere.')
    parser.add_argument('-s', '--results-dir', default='test',
                        type=str, help='save directrory')
    parser.add_argument('-d', default=3, type=int)
    parser.add_argument('--eval-samples', default=int(1e3), type=int)
    parser.add_argument('--train-samples', default=int(1e3), type=int)
    parser.add_argument('--num-iters', default=int(15e3), type=int)
    parser.add_argument('--early-stopping', default=False, type=bool)
    parser.add_argument('--eps', default=0., type=float)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--l1-param-weight', default=0.0, type=float)
    parser.add_argument('--identity-iter', default=0, type=int)
    parser.add_argument('--cc-weight', default=10, type=float)
    parser.add_argument('--nu', default=10000, type=int)
    parser.add_argument('--c-batch', default=0, type=int)
    parser.add_argument('--n_components', default=100, type=int)
    parser.add_argument('--cost_gamma', default=0.1, type=float)
    parser.add_argument('--min_zero_gamma', default=None)
    parser.add_argument('--psi_model', default="discrete")
    # discrete alpha
    parser.add_argument('--n-layers', default=2, type=int)
    parser.add_argument('--init_alpha_mode', default='constant')
    parser.add_argument('--init_alpha_linear_scale', default=1.)
    parser.add_argument('--init_alpha_minval', default=0.4)
    parser.add_argument('--init_alpha_range', default=0.01)
    parser.add_argument('--init_points', default="grid")
    parser.add_argument('--fixed-points', action='store_true')
    # nn
    parser.add_argument('--nn_model', default="mlp")
    parser.add_argument('--nl', default="elu")
    parser.add_argument('--hidden_dims', default=[50, 25])
    # manifold
    parser.add_argument('--manifold', default="S2")
    parser.add_argument('--kde_factor', default=0.08)
    # base
    parser.add_argument('--base', default="uniform")
    # target
    parser.add_argument('--target', default="normal")
    parser.add_argument('--target_loc', default=[-0.5206,  1.,  0.5206])
    parser.add_argument('--target_scale', default=[0.3])

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_params()
    args.results_dir = os.path.join("results", "vqe", args.manifold, args.results_dir)
    os.makedirs(args.results_dir, exist_ok=False or True)

    with open(os.path.join(args.results_dir, "params.json"), 'w+') as f:
        json.dump(vars(args), f)

    device = torch.device(args.device)  # torch.device('cuda')
    th_dtype = torch.float32 # torch.float64 # torch.float32

    manifold = get_manifold(args.manifold, kde_factor=args.kde_factor)
    base = get_density(args.base, manifold=manifold)

    if args.psi_model == "discrete":
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

    target = get_density(args.target, manifold=manifold,
                         loc=np.asarray(args.target_loc),
                         scale=args.target_scale)
    eval_target_samples = target.sample(args.eval_samples)
    manifold.plot_samples_3d(
        [base.sample(args.eval_samples), eval_target_samples, ],
        subplots_titles=["base", "target"],
        camera_eye=np.asarray(args.target_loc) * 1.5,
        save=os.path.join(args.results_dir, f'target_samples'), show=False)

    best_state_dict = vqe.train_loop(target, args.train_samples,
                                     results_dir=args.results_dir, lr=args.lr,
                                     num_iters=args.num_iters, intermediate_plot=True,
                                     save_dir=args.results_dir,
                                     identity_iter=args.identity_iter,
                                     early_stopping=args.early_stopping,
                                     l1_param_weight=args.l1_param_weight,
                                     n_u=args.nu, cc_weight=args.cc_weight,
                                     c_batch=args.c_batch,)

    ## save
    best_state_dict = vqe.state_dict()
    torch.save({
        'target_distr': target, 'base_distr': base,
        'model_state_dict': best_state_dict},
        os.path.join(args.results_dir, 'state_dict.pth'))
    vqe.load_state_dict(best_state_dict)


    eval_target_samples = target.sample(10000)
    N_sample = 10000
    base_samples = base.sample(10000)
    samples_gt = target.sample(N_sample)
    samples_est_random = vqe.sample(N_sample, base_sample=base_samples).cpu().detach()

    # Plot 1: Samples
    manifold.plot_samples_3d(
        [base_samples, samples_est_random, samples_gt],
        save=os.path.join(args.results_dir, f'result'), show=True,
        camera_eye=np.asarray(args.target_loc) * 1.5,
        subplots_titles=["base distribution",
                         f"EST: Randomly sampled on uniform grid.",
                         "target distribution"])
