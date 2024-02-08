import json
import os
import torch
import argparse
import numpy as np

from rcpm.manifolds import get_manifold
from rcpm.densities import get_density, get_CondDensity
from vq_models import ManifoldVQRSingle, ManifoldVQRMulti


def define_model(manifold, base, target, device, th_dtype, args):
    if args.psi_model == "discrete":
        if args.n_layers == 1:
            vqr = ManifoldVQRSingle(manifold, base_density=base,
                                    n_components=args.n_components,
                                    beta_dim=args.beta_dim,
                                    cond_size=target.cond_size,
                                    cond_hidden=args.cond_hidden,
                                    n_class_cond=None if target.continous_cond else target.len_cond,
                                    init_alpha_mode=args.init_alpha_mode,
                                    init_alpha_linear_scale=args.init_alpha_linear_scale,
                                    init_alpha_minval=args.init_alpha_minval,
                                    init_alpha_range=args.init_alpha_range,
                                    n_layers=args.n_layers,
                                    fixed_points=args.fixed_points,
                                    init_points=args.init_points,
                                    cost_gamma=args.cost_gamma,
                                    min_zero_gamma=args.min_zero_gamma,
                                    stack=args.stack_beta,
                                    eps=args.eps,
                                    activation=args.activation,
                                    device=device, th_dtype=th_dtype)
        else:
            vqr = ManifoldVQRMulti(manifold, base_density=base,
                                n_components=args.n_components,
                                beta_dim=args.beta_dim,
                                cond_size=target.cond_size,
                                cond_hidden=args.cond_hidden,
                                n_class_cond=None if target.continous_cond else target.len_cond,
                                init_alpha_mode=args.init_alpha_mode,
                                init_alpha_linear_scale=args.init_alpha_linear_scale,
                                init_alpha_minval=args.init_alpha_minval,
                                init_alpha_range=args.init_alpha_range,
                                n_layers=args.n_layers,
                                fixed_points=args.fixed_points,
                                init_points=args.init_points,
                                cost_gamma=args.cost_gamma,
                                min_zero_gamma=args.min_zero_gamma,
                                stack=args.stack_beta,
                                eps=args.eps,
                                activation=args.activation,
                                device=device, th_dtype=th_dtype)
    else:
        raise NotImplementedError

    if args.load_model is not None:
        vqr.load_state_dict(torch.load(os.path.join(args.load_model, 'state_dict.pth'),
                                       map_location=device)['model_state_dict'])

    return vqr


def parse_params():
    parser = argparse.ArgumentParser(description='Train vqe on sphere.')
    parser.add_argument('-s', '--results-dir', default='test',
                        type=str, help='save directrory')
    parser.add_argument('-d', default=3, type=int)
    parser.add_argument('--eval-samples', default=1e3, type=int)
    parser.add_argument('--train-samples', default=5e2, type=int)
    parser.add_argument('--batch-size', default=int(2e2), type=int)
    parser.add_argument('--num-iters', default=int(1e2), type=int)
    parser.add_argument('--eps', default=0, type=float)
    parser.add_argument('--device', default="cuda:4", type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--l1-weight', default=0, type=float)
    parser.add_argument("--cc-weight", default=100, type=float)
    parser.add_argument("--c-batch", default=0, type=int)
    parser.add_argument("--load-model", default=None, type=str)
    parser.add_argument('--n_components', default=1000, type=int)
    parser.add_argument('--cost_gamma', default=1e-1, type=float)
    parser.add_argument('--min_zero_gamma', default=None)
    parser.add_argument('--nu', default=20000, type=int)
    parser.add_argument('--early_stopping', default=False, type=bool)
    parser.add_argument('--identity-iter', default=0, type=int)
    parser.add_argument('--init_points', default="grid")
    parser.add_argument('--fixed-points', default=False, action='store_true')
    # conditioning params
    parser.add_argument('--beta_dim', default=15, type=int)
    parser.add_argument('--stack_beta', default=False, action='store_true')
    parser.add_argument('--cond_hidden', default=( 2, 4, 8), type=int, nargs='+')
    # convex function
    parser.add_argument('--psi_model', default="discrete")
    parser.add_argument('--activation', default="selu")
    # discrete alpha
    parser.add_argument('--n-layers', default=4, type=int)
    parser.add_argument('--init_alpha_mode', default='constant')
    parser.add_argument('--init_alpha_linear_scale', default=1.)
    parser.add_argument('--init_alpha_minval', default=0.4)
    parser.add_argument('--init_alpha_range', default=0.01)
    # nn
    parser.add_argument('--nn_model', default="mlp")
    parser.add_argument('--nl', default="elu")
    parser.add_argument('--hidden_dims', default=[50, 25])
    # manifold
    parser.add_argument('--manifold', default="Torus")
    parser.add_argument('--kde_factor', default=0.1)
    # base
    parser.add_argument('--base', default="uniform")
    # target
    parser.add_argument('--target', default="heart")
    parser.add_argument('--target_loc', default=[-0.5206,  1.0, 0.5206])
    parser.add_argument('--target_scale', default=1, type=float)
    parser.add_argument('--min_cond', default=0.3, type=float)
    parser.add_argument('--max_cond', default=0.8, type=float)

    args = parser.parse_args()

    args.num_iters = int(args.num_iters)

    return args


if __name__ == "__main__":
    args = parse_params()
    args.results_dir = os.path.join("results", "vqr", args.manifold, args.results_dir)

    os.makedirs(args.results_dir, exist_ok=True)

    with open(os.path.join(args.results_dir, "params.json"), 'w+') as f:
        json.dump(vars(args), f)

    device = torch.device(args.device)  # torch.device('cuda')
    th_dtype = torch.float32

    manifold = get_manifold(args.manifold, kde_factor=args.kde_factor)

    base = get_density(args.base, manifold=manifold, kde_factor=args.kde_factor)
    target = get_CondDensity(args.target, manifold=manifold,
                             loc=np.asarray(args.target_loc), scale=args.target_scale,
                             min_cond=args.min_cond, max_cond=args.max_cond,
                             df=torch.tensor([-1., 0, 0], dtype=th_dtype),
                             dtype=th_dtype)

    vqr = define_model(manifold, base, target, device, th_dtype, args)
    base = vqr.base_distr

    best_state_dict = vqr.train_loop(target, batch_size=args.batch_size,
                                     train_samples=args.train_samples,
                                     results_dir=args.results_dir, lr=args.lr,
                                     n_u=args.nu, cc_weight=args.cc_weight,
                                     c_batch=args.c_batch,
                                     l1_param_weight=args.l1_weight,
                                     early_stopping=args.early_stopping,
                                     identity_iter=args.identity_iter,
                                     num_iters=args.num_iters, intermediate_plot=True,
                                     save_dir=args.results_dir, best=True)

    ## save
    torch.save({
        'target_distr': target, 'base_distr': base,
        'model_state_dict': best_state_dict},
        os.path.join(args.results_dir, 'state_dict.pth'))
    vqr.load_state_dict(best_state_dict).to("cpu")
    vqr.device = "cpu"
    vqr.eval()


    # Plot 1: Samples
    N_sample = 1000

    cond_list = np.linspace(target.min_cond, target.max_cond, 4)
    cond_list = [torch.tensor([i_x]) for i_x in cond_list]
    for x_cond in cond_list:
        samples_gt, conds = target.sample(N_sample, x_cond.to(th_dtype))
        cond_plot = f"{conds[0].item():.2f}" if target.continous_cond else (
            target.cond_text(conds))
        base_samples = base.sample(10000)
        samples_est_random = vqr.sample(conds, N=N_sample).cpu().detach()
        manifold.plot_samples_3d(
            [base_samples, samples_est_random, samples_gt],
            title=f"Cond: {cond_plot}",
            save=os.path.join(args.results_dir, f'result_end_{cond_plot}'),
            show=True,  
            subplots_titles=["base distribution",
                             f"EST: Randomly sampled on uniform grid.",
                             "target distribution"])
