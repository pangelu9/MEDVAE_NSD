"""Render the main 3-panel figure (Panels A, B, C) — fully self-contained.

  Panel A : UMAP of the shared latent space for MED-VAE / SRM / Procrustes,
            coloured by stimulus category.
  Panel B : Category encoding — combined silhouette (bar + bootstrap CI) and
            leave-one-subject-out multi-label decoding (exact-match, balanced).
  Panel C : Alignment (component correlation, RSA) and reconstruction
            (within- / cross-trial voxel correlation).

Inputs (all produced by the release pipeline — see README):
  * Panel A   : visualisation/data/encoder_data_2d_{VAE,srm,procrustes}.pkl   (visualise_latent.py)
  * Panel B/C : <results>/alignment_eval_{method}_32d_not_all8_streams.pkl     (evaluation/evaluate_methods.py)
  * stats     : visualisation/data/figure_stats.json                          (stat_tests.py)

Usage:
  python render_figure.py [--results <dir>] [--out figure.eps]
"""
import os
import json
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from matplotlib.patches import Rectangle
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
DEFAULT_RESULTS = os.environ.get(
    'CCN_RESULTS_DIR', os.path.join(os.path.dirname(HERE), 'evaluation', 'results'))

METHODS = ['vae', 'srm', 'procrustes']
METHOD_COLORS = {'vae': '#8E86C0', 'srm': '#C99494', 'procrustes': '#88B8A9'}
METHOD_LABELS = {'vae': 'MED-VAE', 'srm': 'SRM', 'procrustes': 'Procrustes'}
CATEGORY_NAMES = np.array(['accessory', 'animal', 'appliance', 'electronic', 'food',
                           'furniture', 'indoor', 'kitchen', 'outdoor',
                           'person', 'sports', 'vehicle'])


# ============================== colour helpers ==============================
def create_semantic_color_palette(category_names):
    color_map = {
        'person': '#98D4F3', 'accessory': '#368BBC', 'sports': '#1A4D7A',
        'food': '#FFBF00', 'kitchen': '#96291F',
        'outdoor': '#5FEBA8', 'vehicle': '#2A7A5E',
        'appliance': '#AB59DE', 'furniture': '#55266C',
        'electronic': '#B99FCB', 'indoor': '#513F62', 'animal': '#997344',
    }
    arr = np.zeros((len(category_names), 4))
    for i, cat in enumerate(category_names):
        if cat in color_map:
            arr[i] = list(mcolors.hex2color(color_map[cat])) + [1.0]
    return arr


def blend_colors(color_indices, base_colors):
    cols = np.array([base_colors[i] for i in color_indices])
    blended = np.mean(cols, axis=0)
    hsv = rgb_to_hsv(blended[:3].reshape(1, 1, 3))
    hsv[0, 0, 1] = min(1.0, hsv[0, 0, 1] * 1.15)
    return tuple(hsv_to_rgb(hsv).flatten().tolist() + [1.0])


def find_common_label_combinations(all_labels, category_names, base_colors, top_k=8):
    combos = []
    for label in all_labels:
        active = tuple(sorted(np.where(label > 0.5)[0]))
        if len(active) > 1:
            combos.append(active)
    out = []
    for combo, count in Counter(combos).most_common(top_k):
        out.append({'indices': combo,
                    'names': [category_names[i] for i in combo],
                    'count': count, 'color': blend_colors(combo, base_colors)})
    return out


# ============================== Panel A: UMAP ==============================
def plot_umap_panel(axes, encoder_data_list, model_names, top_k_combos=8):
    single_colors = create_semantic_color_palette(CATEGORY_NAMES)
    all_common_combos = []
    for idx, (encoder_data, model_name, ax) in enumerate(zip(encoder_data_list, model_names, axes)):
        all_latents = np.vstack([d['representations'] for d in encoder_data.values()])
        all_labels = np.concatenate([d['labels'] for d in encoder_data.values()])
        common_combos = find_common_label_combinations(all_labels, CATEGORY_NAMES, single_colors, top_k=top_k_combos)
        if idx == 0:
            all_common_combos = common_combos
        rng = np.random.default_rng(42 + idx)
        jit = 0.05
        DEFAULT_SIZE = plt.rcParams['lines.markersize'] ** 2
        combo_pts = {i: {'x': [], 'y': []} for i in range(len(common_combos))}
        unmatched = {c: {'x': [], 'y': []} for c in range(len(CATEGORY_NAMES))}
        for lat, lab in zip(all_latents, all_labels):
            active = tuple(sorted(np.where(lab > 0.5)[0]))
            matched = False
            for ci, combo in enumerate(common_combos):
                if active == combo['indices']:
                    combo_pts[ci]['x'].append(lat[0]); combo_pts[ci]['y'].append(lat[1]); matched = True; break
            if not matched:
                for cat in active:
                    off = rng.uniform(-jit, jit, size=2)
                    unmatched[cat]['x'].append(lat[0] + off[0]); unmatched[cat]['y'].append(lat[1] + off[1])
        for cat in range(len(CATEGORY_NAMES)):
            if unmatched[cat]['x']:
                ax.scatter(unmatched[cat]['x'], unmatched[cat]['y'], color=single_colors[cat], s=DEFAULT_SIZE * 0.8, alpha=0.7, lw=0)
        for ci, combo in enumerate(common_combos):
            if combo_pts[ci]['x']:
                ax.scatter(combo_pts[ci]['x'], combo_pts[ci]['y'], color=combo['color'], s=DEFAULT_SIZE * 1.2, alpha=0.85, lw=0.8, edgecolors='#C4C4C4')
        ax.set_title(model_name, fontsize=22, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('UMAP2', fontsize=14)
        ax.set_xlabel('UMAP1', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_aspect('auto')
    return all_common_combos, CATEGORY_NAMES


# ============================== significance brackets ==============================
def _add_sig_bracket(ax, x1, x2, y, h, text, fontsize=9):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c='black', clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.1, text, ha='center', va='bottom', fontsize=fontsize, clip_on=False)


# ============================== Panel B: Category encoding ==============================
def plot_category_encoding_panel(ax, category_encoding, silh_bar, sig_results):
    """metric 0 (silhouette) = bar + bootstrap CI; metrics 1,2 (exact/balanced) = violins."""
    metric_order = [('silhouette', None), ('within_exact', 'generalisation'), ('within_balanced', 'generalisation')]
    metric_labels = ['Silhouette', 'Exact match', 'Balanced acc.']
    violin_width, method_gap, group_spacing = 0.40, 1.1, 4.0
    metric_positions, metric_max_y = {}, {}
    for mi, (metric, dataset) in enumerate(metric_order):
        base = mi * group_spacing
        positions, max_y = [], -np.inf
        for k, method in enumerate(METHODS):
            pos = base + k * method_gap
            positions.append(pos)
            if mi == 0:
                val, err = silh_bar[method]
                ax.bar(pos, val, width=violin_width, color=METHOD_COLORS[method], edgecolor='black', linewidth=0.5, zorder=2)
                ax.errorbar(pos, val, yerr=err, fmt='none', ecolor='black', elinewidth=1.0, capsize=3, zorder=3)
                max_y = max(max_y, val + err)
            else:
                data = np.asarray(category_encoding[metric].get((method, dataset), []), float)
                if data.size:
                    max_y = max(max_y, data.max())
                    parts = ax.violinplot([data], positions=[pos], widths=violin_width, showmeans=True, showmedians=False)
                    for pc in parts['bodies']:
                        pc.set_facecolor(METHOD_COLORS[method]); pc.set_edgecolor('black'); pc.set_linewidth(0.3); pc.set_alpha(1.0)
                    for pn in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                        if pn in parts:
                            parts[pn].set_edgecolor('black'); parts[pn].set_linewidth(0.6)
        metric_positions[mi] = positions
        if max_y > -np.inf:
            metric_max_y[mi] = max_y
    _draw_brackets_and_axes(ax, metric_positions, metric_max_y, sig_results, metric_labels,
                            title='Category Encoding', legend_y0=0.18, divider_between=(0, 1))


# ============================== Panel C: Alignment + Reconstruction ==============================
def plot_comparison_panel(ax, alignment, reconstruction, sig_results):
    metric_order = [('comp_corr', alignment, 'pairwise'), ('rsa_vals', alignment, 'pairwise'),
                    ('voxel_corr_mean', reconstruction, 'generalisation'),
                    ('voxel_corr_cross_trial', reconstruction, 'generalisation')]
    metric_labels = ['Comp. corr.', 'RSA', 'Voxel corr.\n(within-trial)', 'Voxel corr.\n(cross-trial)']
    violin_width, method_gap, group_spacing = 0.40, 1.1, 4.0
    metric_positions, metric_max_y = {}, {}
    for mi, (metric, store, dataset) in enumerate(metric_order):
        base = mi * group_spacing
        positions, max_y = [], -np.inf
        for k, method in enumerate(METHODS):
            pos = base + k * method_gap
            positions.append(pos)
            data = np.asarray(store[metric].get((method, dataset), []), float)
            if data.size:
                max_y = max(max_y, data.max())
                parts = ax.violinplot([data], positions=[pos], widths=violin_width, showmeans=True, showmedians=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(METHOD_COLORS[method]); pc.set_edgecolor('black'); pc.set_linewidth(0.3); pc.set_alpha(1.0)
                for pn in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                    if pn in parts:
                        parts[pn].set_edgecolor('black'); parts[pn].set_linewidth(0.6)
        metric_positions[mi] = positions
        if max_y > -np.inf:
            metric_max_y[mi] = max_y
    _draw_brackets_and_axes(ax, metric_positions, metric_max_y, sig_results, metric_labels,
                            title=('Alignment', 'Reconstruction'), legend_y0=0.95, divider_between=(1, 2))


def _draw_brackets_and_axes(ax, metric_positions, metric_max_y, sig_results, metric_labels,
                            title, legend_y0, divider_between):
    ax.set_ylim(auto=True)
    y_lo, y_hi = ax.get_ylim(); y_range = y_hi - y_lo
    bracket_h, bracket_gap = y_range * 0.015, y_range * 0.055
    for mi, comparisons in sig_results.items():
        if mi not in metric_positions:
            continue
        pos = metric_positions[mi]
        base_y = metric_max_y.get(mi, y_hi * 0.8) + y_range * 0.02
        for (a, b, st) in comparisons:
            level = 0 if (b - a) == 1 else 1
            _add_sig_bracket(ax, pos[a], pos[b], base_y + bracket_gap * level, bracket_h, st,
                             fontsize=(8 if st == 'n.s.' else 10))
    if metric_max_y:
        ax.set_ylim(y_lo, max(y_hi, max(metric_max_y.values()) + bracket_gap * 2.2))
    ax.set_ylabel('Performance', fontsize=16, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    n = len(metric_labels)
    centers = [i * 4.0 + 1.1 for i in range(n)]
    ax.set_xticks(centers); ax.set_xticklabels(metric_labels, fontsize=15, fontweight='bold')
    div = 0.5 * (centers[divider_between[0]] + centers[divider_between[1]])
    ax.axvline(x=div, color='gray', linestyle='--', linewidth=0.9, alpha=0.45)
    if isinstance(title, tuple):
        ax.text(np.mean(centers[:divider_between[1]]), 1.06, title[0], transform=ax.get_xaxis_transform(),
                ha='center', fontsize=22, fontweight='bold')
        ax.text(np.mean(centers[divider_between[1]:]), 1.06, title[1], transform=ax.get_xaxis_transform(),
                ha='center', fontsize=22, fontweight='bold')
    else:
        ax.text(np.mean(centers), 1.06, title, transform=ax.get_xaxis_transform(),
                ha='center', fontsize=22, fontweight='bold')
    ax.set_xlim(centers[0] - 1.5, centers[-1] + 1.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    for i, method in enumerate(METHODS):
        ax.text(0.94, legend_y0 - i * 0.07, METHOD_LABELS[method], transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='right', va='center')
        ax.add_patch(Rectangle((0.95, legend_y0 - 0.03 - i * 0.07), 0.03, 0.05,
                     facecolor=METHOD_COLORS[method], edgecolor='black', linewidth=0.5, transform=ax.transAxes))


# ============================== data loading ==============================
def load_panel_A():
    files = {'vae': 'encoder_data_2d_VAE.pkl', 'srm': 'encoder_data_2d_srm.pkl',
             'procrustes': 'encoder_data_2d_procrustes.pkl'}
    return {m: pickle.load(open(os.path.join(DATA, f), 'rb')) for m, f in files.items()}


def load_panel_BC(results_dir):
    alignment = {'comp_corr': {}, 'rsa_vals': {}}
    reconstruction = {'voxel_corr_mean': {}, 'voxel_corr_cross_trial': {}}
    category = {'within_exact': {}, 'within_balanced': {}}
    for m in METHODS:
        d = pickle.load(open(os.path.join(results_dir, f'alignment_eval_{m}_32d_not_all8_streams.pkl'), 'rb'))
        subs = d['subjects']
        alignment['comp_corr'][(m, 'pairwise')] = np.asarray(d['per_component_corr'])
        rsa = np.asarray(d['rsa_matrix_pearson'])
        alignment['rsa_vals'][(m, 'pairwise')] = rsa[np.triu_indices(rsa.shape[0], k=1)]
        reconstruction['voxel_corr_mean'][(m, 'generalisation')] = np.array(
            [d['recon_per_subject'][s]['voxel_correlation'] for s in subs])
        ct = d['cross_trial']['cross_trial_per_subject']
        reconstruction['voxel_corr_cross_trial'][(m, 'generalisation')] = np.array(
            [ct[s]['mean_voxel_corr'] for s in subs if s in ct])
        category['within_exact'][(m, 'generalisation')] = np.asarray(d['decoding']['per_subject_exact'], float)
        category['within_balanced'][(m, 'generalisation')] = np.asarray(d['decoding']['per_subject_balanced'], float)
    return alignment, reconstruction, category


def load_stats():
    s = json.load(open(os.path.join(DATA, 'figure_stats.json')))
    silh_bar = {m: (s['silhouette_bars'][m]['point'],
                    (s['silhouette_bars'][m]['ci'][1] - s['silhouette_bars'][m]['ci'][0]) / 2.0)
                for m in METHODS}
    sig_B = {int(k): [tuple(x) for x in v] for k, v in s['sig_B'].items()}
    sig_C = {int(k): [tuple(x) for x in v] for k, v in s['sig_C'].items()}
    return silh_bar, sig_B, sig_C


# ============================== figure assembly ==============================
def create_figure(umap_data, alignment, reconstruction, category, silh_bar, sig_B, sig_C, out_path):
    fig = plt.figure(figsize=(24, 13))
    gs_main = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.85], hspace=0.18, figure=fig)

    # Row 1: Panel A (UMAP) + Panel B (Category Encoding)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0], width_ratios=[2.5, 1.2], wspace=0.12)
    gs_umap = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_top[0], height_ratios=[2, 0.9], hspace=0.18, wspace=0.08)
    ax_umap = [fig.add_subplot(gs_umap[0, i]) for i in range(3)]
    ax_leg = fig.add_subplot(gs_umap[1, :]); ax_leg.axis('off')
    gs_b = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_top[1], height_ratios=[0.05, 2, 0.75])
    ax_b = fig.add_subplot(gs_b[1])

    # Row 2: Panel C (Alignment + Reconstruction), full width
    ax_c = fig.add_subplot(gs_main[1])

    # ----- Panel A -----
    enc_list = [umap_data['vae'], umap_data['srm'], umap_data['procrustes']]
    common_combos, cat_names = plot_umap_panel(ax_umap, enc_list, ['MED-VAE', 'SRM', 'Procrustes'])
    single_colors = create_semantic_color_palette(cat_names)
    DEFAULT_SIZE = plt.rcParams['lines.markersize'] ** 2
    col_x = [0.02, 0.14, 0.26, 0.42, 0.68]
    row_y = [0.78, 0.64, 0.50, 0.36]
    msz = DEFAULT_SIZE * 1.2
    layout = [['accessory', 'furniture', 'person'], ['animal', 'indoor', 'sports'],
              ['appliance', 'kitchen', 'vehicle'], ['electronic', 'outdoor', 'food']]
    for r, row_cats in enumerate(layout):
        for c, cat in enumerate(row_cats):
            ci = int(np.where(cat_names == cat)[0][0])
            ax_leg.scatter(col_x[c], row_y[r], c=[single_colors[ci]], s=msz, transform=ax_leg.transAxes, zorder=3)
            ax_leg.text(col_x[c] + 0.02, row_y[r], cat, fontsize=16, va='center', transform=ax_leg.transAxes)
    ax_leg.text(0.55, 0.93, 'Most frequent combinations', fontsize=14, fontweight='bold', va='top', ha='center', transform=ax_leg.transAxes)
    half = (len(common_combos) + 1) // 2
    for combos_col, c in [(common_combos[:half], 3), (common_combos[half:], 4)]:
        for r, combo in enumerate(combos_col[:4]):
            ax_leg.scatter(col_x[c], row_y[r], c=[combo['color']], s=msz * 1.3, edgecolors='#C4C4C4',
                           linewidths=1.5, transform=ax_leg.transAxes, zorder=3)
            ax_leg.text(col_x[c] + 0.02, row_y[r], ' + '.join(combo['names']), fontsize=16, va='center', transform=ax_leg.transAxes)
    ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)

    # ----- Panels B & C -----
    plot_category_encoding_panel(ax_b, category, silh_bar, sig_B)
    plot_comparison_panel(ax_c, alignment, reconstruction, sig_C)

    fig.text(0.00, 0.98, 'A', fontsize=24, fontweight='bold', va='top')
    fig.text(0.67, 0.98, 'B', fontsize=24, fontweight='bold', va='top')
    fig.text(0.00, 0.46, 'C', fontsize=24, fontweight='bold', va='top')
    plt.subplots_adjust(left=0.02, right=0.985, top=0.96, bottom=0.04)

    for ext in ('eps', 'pdf', 'png'):
        p = os.path.splitext(out_path)[0] + '.' + ext
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print('saved', p)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=DEFAULT_RESULTS, help='dir with alignment_eval_{method}_32d_not_all8_streams.pkl')
    ap.add_argument('--out', default=os.path.join(HERE, 'figure_abc.eps'))
    a = ap.parse_args()
    umap_data = load_panel_A()
    alignment, reconstruction, category = load_panel_BC(a.results)
    silh_bar, sig_B, sig_C = load_stats()
    print('comp_corr:', {m: round(float(np.mean(alignment["comp_corr"][(m, "pairwise")])), 4) for m in METHODS})
    print('silhouette:', {m: round(silh_bar[m][0], 4) for m in METHODS})
    create_figure(umap_data, alignment, reconstruction, category, silh_bar, sig_B, sig_C, a.out)


if __name__ == '__main__':
    main()
