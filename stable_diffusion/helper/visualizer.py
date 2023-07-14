import math
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="ticks")

def reformat_label(label, use_dagger):
    pattern = r"\((-?\d+\.\d+), '(.*)'\)"
    matches = re.findall(pattern, label)
    if matches:
        order = float(matches[0][0])
        scheduler = matches[0][1]
    else:
        return None, None

    if scheduler == "ghvb":
        beta = order % 1 if order%1 > 0 else 1
        beta = round(beta, 1)
        if beta < 1.0:
            new_label = f"GHVB{order}$^\dagger$" if use_dagger \
                else f"GHVB{order}"
            linestyle = "-"
        elif (beta == 1.0) and (order == 1.0):
            new_label = "DDIM"
            linestyle = "--"
        elif (beta == 1.0):
            new_label = f'PLMS{math.ceil(order)}'
            linestyle = "--"
    elif ("w/ HB" in scheduler) or ("w/ NT" in scheduler):
        beta = order % 1 if order%1 > 0 else 1
        beta = round(beta, 1)
        if beta < 1.0:
            new_label = f"{scheduler} {beta}$^\dagger$" if use_dagger \
                else f"{scheduler} {beta}"
            linestyle = "-"
        elif (beta == 1.0) and (order == 1.0):
            new_label = "DDIM"
            linestyle = "--"
        elif (beta == 1.0):
            new_label = f'{scheduler.split(" ")[0]}'
            linestyle = "--"
    else:
        raise ValueError
    return new_label, linestyle

def reformat_legend(handles, labels, use_dagger, legend_order):
    new_handles = []
    new_labels = []
    for handle, label in zip(handles, labels):
        new_label, new_handle = label, handle
        new_label, linestyle = reformat_label(new_label, use_dagger=use_dagger)
        if new_label is not None:
            new_handle.set_linestyle(linestyle)
            new_labels.append(new_label)
            new_handles.append(new_handle)

    if legend_order is not None:
        new_labels = list(np.array(new_labels)[legend_order])
        new_handles = list(np.array(new_handles)[legend_order])

    return new_handles, new_labels

def plot_metric(
    df, mask, metric, xlabel, ylabel, ax, errorbar=('ci', 95),
    markers=False, start_with_dashed=True, xscale="linear", yscale="linear",
    xlabel_style="plain", ylabel_style="plain", legend_order=None, legend_size=14,
):
    # plot graph
    markersize = 0 if not markers else 4
    if start_with_dashed:
        dashes = [(2, 2), ""] # start with dashed
    else:
        dashes = ["", (2, 2)] # start with solid
    g = sns.lineplot(
        df[mask], x="step", y=metric,
        hue=df[mask].apply(lambda x: (x.label, x.scheduler), axis=1), style="category",
        legend="brief", palette="tab10", errorbar=errorbar, 
        markers=markers, markersize=markersize, dashes=dashes, ax=ax
    )

    handles, labels = g.get_legend_handles_labels()
    new_handles, new_labels = reformat_legend(
        handles, labels, use_dagger=True, legend_order=legend_order,
    )
    g.legend(new_handles, new_labels, loc='best', prop={'size': legend_size})

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    ax.ticklabel_format(axis="x", style=xlabel_style, scilimits=(0,0))
    ax.ticklabel_format(axis="y", style=ylabel_style, scilimits=(0,0))
    ax.grid("on")

def plot_q(
    df, mask, xlabel, ylabel, ax, errorbar=('ci', 95),
    legend_order=None, legend_size=14,
):
    g = sns.pointplot(
        df[mask], x="step", y="q",
        hue=df[mask].apply(lambda x: (x.label, x.scheduler), axis=1),
        palette="tab10", errwidth=2, capsize=.05, 
        seed=0, scale=0.8, ax=ax, errorbar=errorbar
    )

    handles, labels = g.get_legend_handles_labels()
    new_handles, new_labels = reformat_legend(
        handles, labels, use_dagger=True, legend_order=legend_order,
    )
    g.legend(new_handles, new_labels, loc='best', prop={'size': legend_size})

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    ax.grid("on")