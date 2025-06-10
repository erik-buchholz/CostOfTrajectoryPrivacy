#!/usr/bin/env python3
""" """
import os
from typing import List, Dict, Optional, Union

from matplotlib import pyplot as plt
from matplotlib.container import BarContainer

from privtrajgen.evaluation.config import final_evaluation
from privtrajgen.plot.tables import load_results, get_metric_values
from privtrajgen.utils import mean_conf_half, printc


def create_bar_plot(cases: List[List[Optional[str]]],
                    metric: str,
                    title: str,
                    output_dir: str,
                    adjust: Dict[str, float],
                    filename: Optional[str] = None,
                    group_names: Optional[List[str]] = None,
                    labels: Optional[List[str]] = None,
                    y_label: Optional[str] = None,
                    group_offset: float = -0.004,
                    yticks: Optional[List[Union[int, float]]] = None,
                    column_width_pt: float = 241.14749,
                    fig_height_cm: float = 3.5,
                    font_size: int = 6,
                    capsize: int = 3,
                    legend_rows_first: bool = False,
                    fine_textues: bool = True,
                    legend_y: float = 1.52,
                    title_offset: float = 0.12,
                    labelpad: Optional[float] = 0
                    ) -> None:
    """
    Create a grouped bar plot for the specified metric across different cases.

    :param cases: List of groups of case identifiers (each inner list is a group).
    :param metric: Metric name to plot.
    :param title: Plot title.
    :param output_dir: Directory to save the output figures.
    :param adjust: Dictionary of adjustments for `plt.subplots_adjust`.
    :param filename: Optional filename to save the figure (without extension). Defaults to the title.
    :param group_names: Optional list of group names for labelling under each group.
    :param labels: Optional list of case labels (used in the legend).
    :param y_label: Optional label for the y-axis. Defaults to the metric name.
    :param group_offset: Vertical offset for group labels.
    :param yticks: Optional list of y-tick values.
    :param column_width_pt: Width of the figure in points (used for conversion to inches).
    :param fig_height_cm: Height of the figure in centimeters (used for conversion to inches).
    :param font_size: Font size for group labels and ticks.
    :param capsize: Size of the error bar caps.
    :param legend_rows_first: If True, the legend will be arranged in rows first, otherwise in columns.
    :param fine_textues: If True, use finer textures for the bars.
    :param legend_y: Vertical position of the legend.
    :param title_offset: Vertical offset for the title.
    :param labelpad: Padding for the y-axis label.
    """

    # Assert input consistency
    assert len(cases) > 0, "No cases to plot"
    assert group_names is None or len(cases) == len(group_names), "Number of cases and group names do not match"
    # Get length of each group
    group_size = len(cases[0])
    for group in cases:
        assert len(group) == group_size, "Groups must have the same size"
    assert labels is None or len(labels) == group_size, "Number of labels must match group size"

    fig_width = pt_to_inch(column_width_pt)
    fig_height = cm_to_inch(fig_height_cm)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    colors = plt.colormaps.get_cmap('tab20')
    if fine_textues:
        textures = [None] + ['//////', '\\\\\\\\\\', '||||||', '------', 'xxxxxx', '++++++', 'oooooo', '******',
                             '......', '++++++']
    else:
        textures = [None] + ['////', '\\\\\\\\', '||||', '----', 'xxxx', '++++', 'oooo', '****', '....', '++++']

    results = {}
    for group in cases:
        for case in group:
            if case is None:
                continue
            config = final_evaluation[case]
            all_res = load_results(config)
            metric_values = get_metric_values(all_res, metric)
            results[case] = metric_values

    bar_width = 1.2
    group_spacing = 1.0
    current_x = group_spacing
    x_positions = []  # list of lists with actual x-locations per group

    # Iterate over groups and calculate x positions for each bar
    for group in cases:
        group_x = []
        for case in group:
            if case is not None:
                group_x.append(current_x)
                current_x += bar_width
        if group_x:
            x_positions.append(group_x)
            current_x += group_spacing
        else:
            x_positions.append([])

    ids = []
    x_ticks = []
    bars: List[BarContainer] = [None for _ in range(group_size)]
    for j, group in enumerate(cases):
        pos_idx = 0
        for i, case in enumerate(group):
            if case is None:
                continue
            metric_values = results[case]
            mean, half = mean_conf_half(metric_values, 0.95)
            color = colors(i)
            texture = textures[i]
            x_loc = x_positions[j][pos_idx]
            pos_idx += 1
            bar = ax.bar(x_loc, mean, yerr=half, capsize=capsize, width=bar_width, color=color, hatch=texture, align='edge')
            bars[i] = bar
            ids.append(case.upper())
            x_ticks.append(x_loc + bar_width / 2)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(ids)

    if group_names:
        for i, name in enumerate(group_names):
            group_xs = x_positions[i]
            if group_xs:
                left = group_xs[0]
                right = group_xs[-1] + bar_width
                centre_x = (left + right) / 2
                ax.text(centre_x, group_offset, name, ha='center', va='center', fontsize=font_size)

    if labels:
        # First or last group might be PrivTrace and is not complete --> Use centre group as model
        handles = [bars[i] for i in range(group_size)]
        ncol = group_size // 2
        if legend_rows_first:
            nrows = (group_size + ncol - 1) // ncol

            new_handles = [None] * group_size
            new_labels = [None] * group_size

            for idx, (h, lab) in enumerate(zip(handles, labels)):
                row = idx // ncol
                col = idx % ncol
                new_idx = row + col * nrows
                new_handles[new_idx] = h
                new_labels[new_idx] = lab

            handles = new_handles
            labels = new_labels
        ax.legend(handles, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, legend_y))

    ax.set_ylabel(y_label if y_label else metric, labelpad=labelpad)
    ax.set_title(title, y=legend_y - title_offset)
    ax.set_ylim(bottom=0)
    if yticks:
        ax.set_yticks(yticks)

    filename = filename or title.lower().replace(" ", "_")
    filename1 = f"{filename}.pdf"
    filename2 = f"{filename}.png"

    plt.subplots_adjust(**adjust)
    plt.savefig(os.path.join(output_dir, filename1), pad_inches=0.05, dpi=300)
    printc("Saved to path: ", os.path.join(output_dir, filename1))
    plt.savefig(os.path.join(output_dir, filename2), pad_inches=0.05, dpi=300)


def pt_to_cm(pt: float) -> float:
    return pt * 0.0352778


def cm_to_inch(cm: float) -> float:
    """Matplotlib figsize accepts inches"""
    return cm / 2.54


def pt_to_inch(pt: float) -> float:
    return cm_to_inch(pt_to_cm(pt))
