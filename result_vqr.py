import argparse
import json
import os
import numpy as np
import torch
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation

from rcpm.manifolds import get_manifold
from rcpm.densities import get_density
from train_vqr import define_model


latex_font_family = 'Times New Roman, Times, serif'


def json_load(path):
    with open(path, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
    return t_args


camera_dist = {"S2": 1.6, "Torus": 15.6}


def kde_plot(samples_gt, samples_est, manifold,
             title=None, save=None, fr_mean=None):
    print("Plotting kde")

    f = manifold.plot_samples_3d(
        [samples_gt, samples_est, ],
        title=title, save=save,
        show=False,
        show_samples=False,
        subplots_titles=["target distribution", f"Estimated", ])
    return f


def kde_flat(samples_gt, samples_est, manifold, 
             title=None, save=None, kde_factor=None, show_samples=False,
             width=1200, height=700):
    print("Plotting kde")
    f = None
    if manifold.name in ["S2", ]:
        R = torch.tensor(Rotation.from_euler('z', -120, degrees=False))
        samples_gt = torch.as_tensor(R.to(samples_gt.dtype) @ samples_gt.T).T
        samples_est = torch.as_tensor(R.to(samples_est.dtype) @ samples_est.T).T
        
        # mollewide
        f = manifold.plot_kde_mollewide([samples_gt, samples_est, ],
                                        show_samples=show_samples,
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
                                   show=False, show_samples=show_samples,
                                   width=1200, height=700)
    return f


def compute_contours(manifold, vqr, cond_val, base_pole, tau_contours,
                     n_samples=1000, vmap=True):
    contours_base = manifold.geodesic_contours(base_pole, tau_contours,
                                               n_samples)
    contours_target = [vqr.sample(torch.tensor([cond_val]),
                                  base_sample=cb, vmap=vmap).cpu().detach()
                       for cb in contours_base]
    return {"base": contours_base, "target": contours_target}

# define a list of 5 colors to use in plotly
colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
          'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
          'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
          'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
          'rgb(188, 189, 34)', 'rgb(23, 190, 207)'
          ]

def plot_contours(manifold, contours_dict, tau_contours, target_samples=None,
                  title=None, save=None):
    print("Plotting contours")

    f = manifold.plot_contours_3d(
        contours_dict, tau_contours,
        title=title, save=save, show=False,
        target_samples=target_samples,
        show_samples=True, kde_colormap="Oranges",
        show_heatmap=False, save_pdf=True, )
    return f


def plot_contours_flat(manifold, contours_dict, tau_contours,
                       target_samples=None, center=None, scatter_size=1,
                       title=None, save=None):
    f = None
    if manifold.name in ["S2", ]:
        # mollewide
        f = manifold.plot_contour_mollewide(contours_dict, tau_contours,
                                            target_samples=target_samples,
                                            save=save, title=title, center=center,
                                            show=False, scatter_size=scatter_size,
                                            subplots_titles=["Ground truth",
                                                             "Estimated", ])
    elif manifold.name in ["Torus", ]:
        # flat torus
        f = manifold.plot_contour_flat(
            contours_dict, tau_contours,
            target_samples=target_samples,
            center=center, save=save, title=title,
            show=False, marker_size=scatter_size,
            subplots_titles=["Ground truth", "Estimated", ])
    return f

def plot_coverages(coverages,save=None):
    """
    est_coverages = dict()
    """
    gt_coverage = np.array(list(coverages.keys()))
    est_coverage = np.array(list(coverages.values()))
    # plot the results
    fig = go.Figure()
    # plot the ground truth as a line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines', line=dict(color='green', dash='dash'),
                             name='Ground Truth'))
    fig.add_trace(go.Scatter(x=gt_coverage, y=est_coverage,
                                mode='markers', name="Estimated",
                                marker=dict(size=12,
                                            color='rgb(31, 119, 180)',
                                            symbol="x-open-dot",
                                            line=dict(width=1)),
                                ))
    fig.update_layout(title='Coverage of the quantile function',
                      font=dict(family=latex_font_family, size=18),
                      title_font_family=latex_font_family,
                      plot_bgcolor='white',
                      xaxis=dict(gridcolor='grey', title='Coverage level',
                                 zerolinecolor='black',
                                 gridwidth=1, tickformat='.2f', dtick=0.25, ),
                      # Change the x-axis
                      yaxis=dict(gridcolor='grey', title='Probability',
                                 zerolinecolor='black',
                                 gridwidth=1, tickformat='.2f', dtick=0.25, ),
                      # Change the y-axis grid color
                      )
    # fig.show()
    if save is not None:
        fig.write_image(save)
    return fig

def plot_coverages_dict(est_coverages, save=None):
    fig = go.Figure()
    # plot the ground truth as a line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                            mode='lines', line=dict(color='green', dash='dash'),
                            name='Ground Truth'))
    for i_c,cond_plot in enumerate(est_coverages.keys()):
        gt_coverage = est_coverages[cond_plot][0]
        est_coverage = est_coverages[cond_plot][1]
        fig.add_trace(go.Scatter(x=gt_coverage, y=est_coverage,
                            mode='markers', name=f"{cond_plot}",
                            marker=dict(size=7,
                                        color=colors[i_c%len(colors)],
                                        symbol="x-open-dot",
                                        line=dict(width=1)),
                            ))
    fig.update_layout(title='Coverage of the quantile function',
                        font=dict(family=latex_font_family, size=18),
                        title_font_family=latex_font_family,
                        plot_bgcolor='white',
                        xaxis=dict(gridcolor='grey', title='Coverage level',
                                     zerolinecolor='black',
                                     gridwidth=1, tickformat='.2f', dtick=0.25, ),
                        # Change the x-axis
                        yaxis=dict(gridcolor='grey', title='Probability',
                                     zerolinecolor='black',
                                     gridwidth=1, tickformat='.2f', dtick=0.25, ),
                        # Change the y-axis grid color
                        )
    if save is not None:
        fig.write_image(save)
    return fig

def marginal_cov(est_coverages, save=None):
    fig = go.Figure()
    # plot the ground truth as a line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                            mode='lines', line=dict(color='green', dash='dash'),
                            name='GT'))
    
    marg_covs = dict()
    err_covs = list()
    for i_c,cond_plot in enumerate(est_coverages.keys()):
        gt_coverage = est_coverages[cond_plot][0]
        est_coverage = est_coverages[cond_plot][1]
        for i_marg in range(0, len(gt_coverage)):
            marg_covs[gt_coverage[i_marg]] = marg_covs.get(gt_coverage[i_marg], []) + [est_coverage[i_marg]]
            err_covs.append(np.abs(gt_coverage[i_marg] - est_coverage[i_marg]))
    str_buffer = "Marginal coverage\n"
    for gt_cov,est_covs in marg_covs.items():
        str_buffer += f"{gt_cov}: {np.mean(est_covs):.2%} +- {np.std(est_covs):.2%}\n"
        fig.add_trace(go.Box(x0=np.asarray(gt_cov),
                             y=np.asarray(est_covs), 
                             name=f"{gt_cov:.0%}",
                             width=0.05,)
                            )
    str_buffer += f"Mean error: {np.mean(err_covs):.2%} +- {np.std(err_covs):.2%}\n"
    fig.update_layout(title='Marginal Coverage of the quantile function',
                        font=dict(family=latex_font_family, size=18),
                        title_font_family=latex_font_family,
                        plot_bgcolor='white',
                        xaxis=dict(gridcolor='grey', title='Coverage level',
                                     zerolinecolor='black',
                                     gridwidth=1, tickformat='.0%', dtick=0.25, ),
                        # Change the x-axis
                        yaxis=dict(gridcolor='grey', title='Probability',
                                     zerolinecolor='black',
                                     gridwidth=1, tickformat='.0%', dtick=0.25, ),
                        # Change the y-axis grid color
                        )
    if save is not None:
        fig.write_image(save)

    print(str_buffer)
    with open(save.replace(".pdf", ".txt"), 'w+') as f:
        f.write(str_buffer)
    return fig

def likelihood_plot(manifold, vqe, cond, results_dir, cond_plot="", 
                    N_sample=10000,marker_size=7, eps=5e-2):
    print("Plotting likelihood")
    samples_grid = manifold.grid(N_sample).to(vqe.th_dtype)
    lh = vqe.likelihood(cond,U_sampled=samples_grid, eps=eps, log=True)
    if manifold.name in ["S2", ]:
        # mollewide
        f = manifold.plot_mollewide([samples_grid, ], samples_colors=lh,
                                        title="Likelihood", 
                                        width=1200, height=700,
                                        scatter_size=marker_size, opacity=0.7,
                                        save=os.path.join(results_dir, f'likelihood_{cond_plot}'),
                                        show=False, show_colorscale=True)
    elif manifold.name in ["Torus", ]:
        # flat torus
        f = manifold.plot_flat([samples_grid, ], samples_colors=[lh],
                                   title="Likelihood", 
                                   marker_size=marker_size, opacity=0.7,
                                        save=os.path.join(results_dir, f'likelihood_{cond_plot}'),
                                        show=False, show_colorscale=True)
    return f

def parse_params():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--manifold", type=str, default="S2")
    parser.add_argument('-d', "--dir", type=str, default="rezende")
    parser.add_argument("--save-dir", type=str, default="eval")
    parser.add_argument("--tests",
                        default=[ 'coverage',
                                 "kde_loss", "contours", "kde_plot", "flat_plot", 'coverage',"samples_plot",
                                 ],
                        nargs='+')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--kde_factor", type=float, default=0.08)
    parser.add_argument('--no-vmap', action='store_false', dest='vmap', default=True)
    parser.add_argument("--chunk-size", default=2500, type=int)
    parser.add_argument('--eval-samples', default=8000, type=int)
    parser.add_argument('--n-conds', default=4, type=int)
    parser.add_argument('--tau-contours', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], type=float, nargs='+')
    parser.add_argument("--best", action='store_true', default=False)
    parser.add_argument("--eps", type=float, default=None)


    return parser.parse_args()


if __name__ == "__main__":
    args_test = parse_params()
    tests = args_test.tests
    eval_samples = args_test.eval_samples
    results_dir = os.path.join("results", "vqr", args_test.manifold, args_test.dir)
    print(results_dir)
    args = json_load(os.path.join(results_dir, "params.json"))

    device = torch.device(args_test.device)  
    th_dtype = torch.float32

    manifold = get_manifold(args.manifold, kde_factor=args.kde_factor)
    base = get_density(args.base, manifold=manifold)

    state_dict = torch.load(os.path.join(args.results_dir, 'best_state_dict.pth' if args_test.best
                                          else 'state_dict.pth'),
                            map_location={str(args.device): str(device)})
    target = state_dict["target_distr"]

    vqr = define_model(manifold, base, target, device, th_dtype, args)

    vqr.load_state_dict(state_dict["model_state_dict"])
    vqr.eval()
    base = state_dict["base_distr"]
    base.is_uniform = True
    manifold.kde_factor = args_test.kde_factor

    if args_test.n_conds==0:
        cond_list = np.arange(target.min_cond, target.max_cond,)
    else:
        cond_list = np.linspace(target.min_cond, target.max_cond, args_test.n_conds)
    cond_list = [torch.tensor([i_x]).to(device) for i_x in cond_list]

    base_samples = base.sample(eval_samples)
    tau_contours = args_test.tau_contours
    est_coverages = dict()

    results_dir = os.path.join(results_dir, args_test.save_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Plot inverse map
    for x_cond in cond_list:
        samples_gt, cond_val = target.sample(eval_samples, x_cond.item())
        cond_plot = f"{cond_val.item():.2f}" if target.continous_cond else (
            target.cond_text(cond_val))
        print(f"Cond: {cond_plot}")
        samples_est = None
        est_base_samples = None

        if set(args_test.tests).intersection(["kde_loss", "kde_plot", "contours",
                                              "flat_plot", "samples_plot"]):
            print(f"vmap {args_test.vmap}, chunk_size {args_test.chunk_size}")
            samples_est = vqr.sample(cond_val,
                                     base_sample=base_samples,
                                     vmap=args_test.vmap,
                                     chunk_size=args_test.chunk_size
                                     ).cpu().detach()
        if set(args_test.tests).intersection(["contours", "coverage"]):
            fr_mean = target.frechet_mean(cond_val)
            if not isinstance(fr_mean, torch.Tensor):
                fr_mean = torch.tensor(fr_mean)
            base_pole = vqr.sample_inverse(
                fr_mean.unsqueeze(0).to(th_dtype),
                cond_val, eps=args_test.eps)[0, :].cpu().detach()
        if set(args_test.tests).intersection(["kde_plot"]):
            fr_mean = target.frechet_mean(cond_val)
            if isinstance(fr_mean, torch.Tensor):
                fr_mean = fr_mean.cpu().detach().numpy()
                
        if "samples_plot" in tests:
            f =manifold.plot_samples_3d([samples_gt, samples_est, ], 
                                        title=f"Cond: {cond_plot}", 
                                        save=os.path.join(results_dir, f'samples_{cond_plot}'),
                                            show=False,
                                            subplots_titles=[
                                                "GT distribution.",  f"Forward"])

        if "kde_plot" in tests:
            kde_plot(samples_gt, samples_est, manifold, title=f"Cond: {cond_plot}",
                     save=os.path.join(results_dir, f'result_kde_{cond_plot}'),
                     fr_mean=fr_mean)
            kde_flat(samples_gt, samples_est, manifold, title=f"Cond: {cond_plot}",
                     save=os.path.join(results_dir, f'kde_flat_{cond_plot}'))

        if "flat_plot" in tests:
            print("Plotting flat")
            if args.manifold in ["S2", ]:
                # mollewide
                manifold.plot_mollewide([samples_gt, samples_est, ],
                                        title=f"Cond: {cond_plot}",
                                        save=os.path.join(results_dir,
                                                          f'moll_{cond_plot}'),
                                        show=False, scatter_size=2, )
            elif args.manifold in ["Torus", ]:
                # flat torus
                f = manifold.plot_flat([samples_gt, samples_est, ],
                                       save=os.path.join(results_dir,
                                                         f'flat_{cond_plot}'),
                                       show=False, marker_size=2, show_hist=True,
                                       title=f"Cond: {cond_plot}",
                                       subplots_titles=["Ground truth", "Estimated", ])
                
        if "kde_loss" in tests:
            print("Computing kde loss")

            kde_loss = manifold.kde_loss(samples_gt, samples_est)
            print("kde_loss: ", kde_loss)
            with open(os.path.join(results_dir, f"kde_loss{cond_plot}.txt"),
                      "wt") as f:
                f.write(f"{kde_loss}")
        
        if "likelihood" in tests:
            lh = likelihood_plot(manifold, vqr, cond_val, 
                            results_dir, cond_plot= cond_plot,
                            N_sample=eval_samples,
                              marker_size=5, eps=args_test.eps)

        if "inverse" in tests:
            print("Plotting inverse map")
            inv_samples = vqr.sample_inverse(samples_gt, cond_val,
                                                  vmap=args_test.vmap,
                                                  chunk_size=args_test.chunk_size,
                                                  eps=args_test.eps).cpu().detach()
            manifold.plot_samples_3d(
                [samples_gt, inv_samples],
                title=f"Cond: {cond_plot}",
                subplots_titles=["target", "inverse", "GT_base"],
                save=os.path.join(results_dir, f'inverse_sample_{cond_plot}'),
                show=False)

        # Plot contours from frechet_mean
        if "contours" in tests and args.manifold in ["S2", "Torus"]:
            contours_dict = compute_contours(manifold, vqr, cond_val, base_pole,
                                             tau_contours, n_samples=250,
                                             vmap=args_test.vmap)

            plot_contours(manifold, contours_dict, tau_contours,
                          target_samples=[base_samples, samples_gt],
                          title=f"Cond: {cond_plot}",
                          save=os.path.join(results_dir, f'contour_{cond_plot}'))

            plot_contours_flat(manifold, contours_dict, tau_contours,
                               target_samples=[base_samples, samples_gt],
                               center={"base": base_pole, "target": fr_mean},
                               title=f"Cond: {cond_plot}",
                               save=os.path.join(results_dir,
                                                 f'contour_flat_{cond_plot}'))

        if "contour_theta_betha" in tests and base.is_uniform:
            print("Plotting contours theta betha")
            # plot contour_theta_betha on base
            contours_base = [base.contour_theta(n=1000, theta=np.pi / 2),
                             base.contour_theta(n=1000, theta=np.pi / 3),
                             base.contour_phi(n=1000, phi=np.pi / 2),
                             base.contour_phi(n=1000, phi=np.pi / 3)
                             ]
            contours_target = [vqr.sample(torch.tensor([cond_val]).to(device).to(
                th_dtype), base_sample=contour)
                for contour in contours_base]
            manifold.plot_contours_3d(
                {"base": contours_base, "target": contours_target},
                title=f"{cond_plot}",
                show_samples=False, kde_colormap="Oranges",
                target_samples=[base_samples, samples_gt],
                save=os.path.join(results_dir,
                                  f'contour_theta_phi_{cond_plot}'),
                show=False)

        if "coverage" in tests:
            print("Computing coverage")
            samples_gt_cov, cond_val = target.sample(eval_samples, x_cond.item(), mode="test")
            if est_base_samples is None:
                est_base_samples = vqr.sample_inverse(samples_gt_cov,
                                                      cond_val,
                                                      vmap=args_test.vmap,
                                                      chunk_size=args_test.chunk_size,
                                                      eps=args_test.eps).cpu().detach()
            coverages = manifold.coverage_test_geodesic(est_base_samples, base_pole,
                                                        tau_contours, empirical=True)
            est_coverages[cond_plot] = (np.array(list(coverages.keys())), np.array(list(coverages.values())))

            print(f"Num samples: {samples_gt_cov.shape[0]}")
            print("coverages: ", coverages)
            error = np.abs(np.array(list(coverages.keys())) -
                           np.array(list(coverages.values())))
            print("error: ", error)
            print("mean error: ", np.mean(error))
            with open(os.path.join(results_dir, f"coverage_{cond_plot}.txt"),
                      "wt") as f:
                f.write(f"{coverages}\n")
                f.write(f"{error}\n")
            plot_coverages(coverages,save=os.path.join(results_dir, f"coverage_{cond_plot}.pdf"))
            
        print("#############################################")
    if "coverage" in tests:
        plot_coverages_dict(est_coverages, save=os.path.join(results_dir, f"coverage.pdf"))
        marginal_cov(est_coverages, save=os.path.join(results_dir, f"marginal_coverage.pdf"))
    