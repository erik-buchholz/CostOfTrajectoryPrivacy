#!/usr/bin/env python3
"""Helper Code to visualize the different configurations."""
import logging
from enum import Enum
from typing import List, Any, Dict, Optional, Sequence, Union, Tuple

import numpy as np
from scipy.stats import ttest_rel

from privtrajgen.config import ConditionalInfoSource, NoiseType
from privtrajgen.evaluation.config import BaseConfig, DiffTrajConfig, UNetVAEConfig, UNetGANConfig, final_evaluation, \
    Config, PrivTraceConfig
from privtrajgen.metrics import load_case_metrics, keys
from privtrajgen.utils import mean_conf_half

log = logging.getLogger(__name__)


###############################################################################
# Enums
###############################################################################
class ThreadModel(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

    def __str__(self) -> str:
        """Return the string representation of the ThreadModel."""
        return self.value

###############################################################################
# Constants
###############################################################################
NUM_RUNS = 5
WARN_IF_MISSING = True  # Activate for final evaluation!
THREAT_MODELS = [ThreadModel.A, ThreadModel.B, ThreadModel.C, ThreadModel.D]
N_BASE_CASES = 24
BASE_CASES = list(range(1, N_BASE_CASES + 1))
CONF = 0.95  # Confidence level for the confidence interval if used
JSD_GRAN = 64  # Granularity for JSD to show (we measured multiple, but only show 1 for simplicity)
HS_GRAN = 128  # Granularity for Hotspot Analysis to show (we measured multiple, but only show 1 for simplicity)

# % Printer friendly colours -> https://colorbrewer2.org/#type=diverging&scheme=Spectral&n=4
# \definecolor{printred}{RGB}{215,25,28}
# \definecolor{printorange}{RGB}{253,174,97}
# \definecolor{printgreen}{RGB}{171,221,164}
# \definecolor{printblue}{RGB}{43,131,186}
THREAT_COLOR = {
    ThreadModel.A: "printred",
    ThreadModel.B: "printorange",
    ThreadModel.C: "printblue",
    ThreadModel.D: "printgreen"
}

METRICS_SHORT_NAMES = {
    keys.JSD_DISTRIBUTION: "JSD",
    keys.SWD_DISTRIBUTION: "SWD",
    keys.HD_POINTS: "HD (P)",
    keys.RANGE_QUERY: "Range",
    keys.HOTSPOT_ANALYSIS: "Hotspot",
    keys.HD_AGGREGATED: "HD (T)",
    keys.ED_AGGREGATED: "Haversine",
    keys.DTW_AGGREGATED: "DTW",
    keys.TTD: "TTD",
    keys.DIAMETER: "TD",
    keys.TRIP_ERROR: "TE"
}

METRICS_RANGE = {
    keys.JSD_DISTRIBUTION: (0, 1),
    keys.SWD_DISTRIBUTION: (0, float('inf')),
    keys.HD_POINTS: (0, float('inf')),
    keys.RANGE_QUERY: (0, 1),
    keys.HOTSPOT_ANALYSIS: (0, 1),
    keys.HD_AGGREGATED: (0, float('inf')),
    keys.ED_AGGREGATED: (0, float('inf')),
    keys.DTW_AGGREGATED: (0, float('inf')),
    keys.TTD: (0, 1),
    keys.DIAMETER: (0, 1),
    keys.TRIP_ERROR: (0, 1)
}


class MetricDirection(Enum):
    SMALLER = -1
    LARGER = 1

    def smaller_is_better(self) -> bool:
        return self == MetricDirection.SMALLER

    def larger_is_better(self) -> bool:
        return self == MetricDirection.LARGER


METRICS_DIRECTION = {  # -1: Smaller is better, 1: Larger is better
    keys.JSD_DISTRIBUTION: MetricDirection.SMALLER,
    keys.SWD_DISTRIBUTION: MetricDirection.SMALLER,
    keys.HD_POINTS: MetricDirection.SMALLER,
    keys.RANGE_QUERY: MetricDirection.SMALLER,
    keys.HOTSPOT_ANALYSIS: MetricDirection.LARGER,
    keys.HD_AGGREGATED: MetricDirection.SMALLER,
    keys.ED_AGGREGATED: MetricDirection.SMALLER,
    keys.DTW_AGGREGATED: MetricDirection.SMALLER,
    keys.TTD: MetricDirection.SMALLER,
    keys.DIAMETER: MetricDirection.SMALLER,
    keys.TRIP_ERROR: MetricDirection.SMALLER
}

PAPER_METRICS = [
    keys.JSD_DISTRIBUTION,
    keys.SWD_DISTRIBUTION,
    keys.HD_POINTS,
    keys.RANGE_QUERY,  # Use MRE in paper
    keys.HOTSPOT_ANALYSIS,
    keys.HD_AGGREGATED,
    keys.ED_AGGREGATED,
    keys.DTW_AGGREGATED,
    keys.TTD,
    keys.DIAMETER,
    keys.TRIP_ERROR
]


def get_metric_header(metric: str) -> str:
    r"""
    Return the header for the metric:
    'Short-Name (\uparrow|\downarrow)'
    """

    short_name = METRICS_SHORT_NAMES[metric]
    direction = r"$\uparrow$" if METRICS_DIRECTION[metric].larger_is_better() else r"$\downarrow$"
    return f"{short_name} ({direction})"

###############################################################################
# LaTeX Helper Functions
###############################################################################
def ballnumber(id: str) -> str:
    return f"\\ballnumber{{{id}}}"


def bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def italic(text: str) -> str:
    return f"\\textit{{{text}}}"


def textcolor(txt: str, color: str) -> str:
    return f"\\textcolor{{{color}}}{{{txt}}}"


def highlight_x(text: str) -> str:
    """Highlight the 'x' cases (those with eps_c = 20.0)."""
    return italic(textcolor(text, "RoyalBlue"))


def cellcolor(color: str) -> str:
    return f"\\cellcolor{{{color}}}"


def bool_to_latex(value: bool) -> str:
    r"""
    Repalces True by $\checkmark$ and False by \xmark.

    :param value: Boolean value to convert.
    :return: LaTeX representation.
    """
    return r"$\checkmark$" if value else r"\xmark"


###############################################################################
# Compare Values
###############################################################################
def is_base_case(case: str) -> bool:
    # Check if 'x' or 'a' is in the case and the number is smaller equal to N_BASE_CASES
    if 'x' in case or 'a' in case or int(''.join(filter(str.isdigit, str(case)))) not in BASE_CASES:
        return False
    return True


def is_x_case(case: str) -> bool:
    return 'x' in case or 'a' in case


def is_better(value: float, best_value: float, metric: str) -> bool:
    """
    Check if the value is better than the best value for the metric.
    """
    if METRICS_DIRECTION[metric].smaller_is_better():
        return value < best_value
    else:
        return value > best_value


def is_better_equal(value: float, best_value: float, metric: str) -> bool:
    """
    Check if the value is better than the best value for the metric.
    """
    return is_better(value, best_value, metric) or value == best_value


def ttest(values1: List[float], values2: List[float], alpha: float = 0.05) -> bool:
    """
    Perform a two-sided paired t-test and determine if the difference is significant.

    :param values1: Metric values from the first case.
    :param values2: Metric values from the second case.
    :param alpha: Significance level (default: 0.05).
    :return: True if the difference is statistically significant, False otherwise.
    """
    stat, p_value = ttest_rel(values1, values2)
    return p_value < alpha


def ttest_group(
        values: Dict[str, List[float] | None],
        best_case: str,
        other_cases: Sequence[str],
) -> bool:
    """
    Perform a paired t-test for the best case against all other cases.
    """
    for case in other_cases:
        if case == best_case:
            continue
        # Perform paired t-test
        v = values[best_case]
        vo = values[case]
        if v is None or vo is None:
            raise ValueError(f"None value in {best_case} or {case}.")
        if not ttest(v, vo):
            return False
    return True
###############################################################################
# Codes
###############################################################################

def encode_id(id: Union[str, int]) -> str:
    """
    Encode the ID for LaTeX.

    :param id: ID to encode.
    :return: Encoded ID.
    """
    # Make string if integer
    txt = str(id)
    # Make uppercase
    txt = txt.upper()
    # Replace 'x' (if present) by 'a' (lower case!)
    txt = txt.replace("X", "a")
    return txt


def get_threat_model(config: Config) -> ThreadModel:
    """
    Return the threat model ID for the configuration.

    - A: No DP
    - B: DP-SGD but unconditional conditional information
    - C: No DP-SGD with conditional DP (or unconditional model)
    - D: DP-SGD with conditional DP (or unconditional model)

    :param config: Configuration to get the threat model for.
    :return: Threat model ID.
    """
    if isinstance(config, BaseConfig):
        dpsgd = config.dp.dp
        cond_dp = not config.conditional or config.cond.dp

        if not dpsgd and not cond_dp:
            return ThreadModel.A
        elif dpsgd and not cond_dp:
            return ThreadModel.B
        elif not dpsgd and cond_dp:
            return ThreadModel.C
        elif dpsgd and cond_dp:
            return ThreadModel.D
        else:
            raise ValueError(f"Impossible Path.")
    elif isinstance(config, PrivTraceConfig):
        return ThreadModel.D
    else:
        raise ValueError(f"Unknown configuration type: {type(config)}. Expected BaseConfig or PrivTraceConfig.")

def get_conf_code(config: Config) -> str:
    """
    Return the code for the configuration.
    Each setup is identified by an ID and a code `[1–6][N|D]`, where the digit indicates the conditional type, and `N/D` signifies whether DP-SGD is off or on.
    Conditional types:
        1. **None:** Unconditional models.
        2. **DTcond:** Original DiffTraj embedding.
        3. **DTcond + lap:** DiffTraj embedding with Laplace mechanism.
        4. **Sample + lap:** Full sample with Laplace mechanism.
        5. **Sample + gau:** Full sample with Gaussian mechanism.
        6. **Sample + V:** Full sample with vmf mechanism.

    :param config: Configuration to get the code for.
    :return: Code for the configuration: `[1–6][N|D]`
    """
    if isinstance(config, BaseConfig):
        cond_code: int
        dp_code: str = "D" if config.dp.dp else "N"
        if not config.conditional:
            cond_code = 1
        else:
            if config.cond.source == ConditionalInfoSource.DIFFTRAJ and not config.cond.dp:
                cond_code = 2
            elif config.cond.source == ConditionalInfoSource.DIFFTRAJ and config.cond.dp:
                cond_code = 3
            elif config.cond.source == ConditionalInfoSource.SAMPLE and config.cond.noise == NoiseType.LAPLACE:
                cond_code = 4
            elif config.cond.source == ConditionalInfoSource.SAMPLE and config.cond.noise == NoiseType.GAUSSIAN:
                cond_code = 5
            elif config.cond.source == ConditionalInfoSource.SAMPLE and config.cond.noise == NoiseType.VMF:
                cond_code = 6
            else:
                raise ValueError(f"Unknown conditional configuration: {config.cond}")
        return f"{cond_code}{dp_code}"
    elif isinstance(config, PrivTraceConfig):
        return "PT"  # PrivTrace does not have a code, but is a special case
    else:
        raise ValueError(f"Unknown configuration type: {type(config)}. Expected BaseConfig or PrivTraceConfig.")


def get_model_str(config: Config) -> str:
    """
    Return the model string for the configuration.

    :param config: Configuration to get the model string for.
    :return: Model string.
    """
    if isinstance(config, DiffTrajConfig):
        return "DiffTraj"
    elif isinstance(config, UNetVAEConfig):
        return "UNetVAE"
    elif isinstance(config, UNetGANConfig):
        return "UNetGAN"
    elif isinstance(config, PrivTraceConfig):
        return "PrivTrace"
    else:
        raise ValueError(f"Unknown configuration: {config}")


def get_cond_dp_str(config: BaseConfig, thesis: bool = False) -> str:
    r"""
    Return the conditional DP string for the configuration.
    unconditional: '--'
    None: None
    Laplace: \gls{lap}
    Gaussian: \gls{gau}
    VMF: \gls{V}

    :param config: Configuration to get the conditional DP string for.
    :param thesis: If False, use paper command. If True, use thesis commands.
    :return: Conditional DP string.
    """
    if not config.conditional:
        return "--"
    elif config.cond.dp:
        if config.cond.noise == NoiseType.LAPLACE:
            return r"$Lap$" if thesis else r"\gls{lap}"
        elif config.cond.noise == NoiseType.GAUSSIAN:
            return r"$\mathcal{N}$" if thesis else r"\gls{gau}"
        elif config.cond.noise == NoiseType.VMF:
            return r"$\mathcal{V}$" if thesis else r"\gls{V}"
        else:
            raise ValueError(f"Unknown conditional noise type: {config.cond.noise}")
    else:
        return "None"


def get_cond_info_type(config: BaseConfig) -> str:
    """
    Return the conditional information type for the configuration.

    - None: No conditional information
    - DiffTraj: DiffTraj conditional information
    - Sample: Full sample as conditional information

    :param config: Configuration to get the conditional information type for.
    :return: Conditional information type.
    """
    if not config.conditional:
        return "None"
    elif config.cond.source == ConditionalInfoSource.DIFFTRAJ:
        return r"\DTcond"
    elif config.cond.source == ConditionalInfoSource.SAMPLE:
        return r"\Sample"
    else:
        raise ValueError(f"Unknown conditional information source: {config.cond.source}")


def get_multicolumn_row(n_cols: int, text: str) -> str:
    """
    Return a multicolumn row for LaTeX table.

    :param n_cols: Number of columns to span.
    :param text: Text to display.
    :return: Multicolumn row.
    """
    return fr"\multicolumn{{{n_cols}}}{{c}}{{ \textbf{{ {text} }} }} \\ \midrule"


def to_latex(rows: List[Any], header_columns: List[str], column_alignment: List[str],
             subheaders: Optional[Dict[str, str]] = None, highlight_special: bool = True,
             ) -> str:
    """
    Print the rows as a LaTeX table.
    - rows containing multicolumns are added afterward, followed by a separating line (\\midrule).
    - The header row should be bold.
    - column alignment contains 'c', 'l', 'r', etc.
    - LaTeX special characters are NOT escaped.

    :param rows: List of rows to print.
    :param header_columns: List of header columns.
    :param column_alignment: List of column alignments.
    :param subheaders: Dictionary of subheaders; {before_row_id: subheader_text}
    """
    # Begin LaTeX table
    table_lines: List[str] = list()
    table_lines.append(r"%%%%%%%% BEGIN TABLE %%%%%%%%")
    table_lines.append(f"\\begin{{tabular}}{{{''.join(column_alignment)}}}")
    table_lines.append("\\toprule")

    # Add header row with bold columns
    header_row = " & ".join([f"\\textbf{{{col}}}" for col in header_columns]) + " \\\\"
    table_lines.append(header_row)
    table_lines.append(r"\midrule")

    # Process each row
    for row in rows:
        row_id = row[0]
        if subheaders is not None and row_id in subheaders:  # Multicolumn row
            if subheaders[row_id] is not None:
                table_lines.append("")
                table_lines.append(get_multicolumn_row(len(header_columns), subheaders[row_id]))
            else:
                # Print a separating line without header
                table_lines.append(r"\midrule")
                # Add an empty line
                table_lines.append("")
        # Do encoding for a rows
        if 'a' in row_id and highlight_special:
            row[0] = r"\rowcolor{gray!10}" + highlight_x(row[0])
        # Regular row
        formatted_row = " & ".join(map(str, row)) + " \\\\"
        table_lines.append(formatted_row)

    # End LaTeX table
    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append(r"%%%%%%%% END TABLE %%%%%%%%")

    # Prepend a tab to all lines except for the first two and the last two lines
    table_lines = table_lines[:2] + ["    " + line for line in table_lines[2:-2]] + table_lines[-2:]


    # Join lines and print
    latex_table = "\n".join(table_lines)
    return latex_table


def load_results(config: BaseConfig | PrivTraceConfig, num_runs: int = NUM_RUNS, warn: bool = WARN_IF_MISSING) -> Sequence[dict]:
    """
    Try to load num_runs results for the configuration.
    However, fail gracefully!
    If there is no run for a config, log an error and continue.
    If there are runs, but less than num_runs, log a warning and continue.

    :param config: Configuration to load the results for.
    :param num_runs: Number of runs to load.
    :param warn: Whether to log a warning if not all runs are available.
    :return: List of results.
    """
    results = []
    for run in range(1, num_runs + 1):
        # Set the run id in the case
        config.run_id = run
        try:
            metrics = load_case_metrics(config)
            results.append(metrics)
        except FileNotFoundError:
            if run == 1:
                log.error(f"No results for {config.name}.")
            elif warn:
                log.warning(f"Could not load results for {config.name} run {run}.")
            break
    return results


def get_metric_values(results: Sequence[dict], metric: str) -> List[float]:
    """
    Get the values for the metric from the results.

    :param results: List of results.
    :param metric: Metric to get the values for.
    :return: All values for the metric.
    """
    if metric == keys.JSD_DISTRIBUTION:
        values = [res[metric][str(JSD_GRAN)] for res in results]
    elif metric == keys.RANGE_QUERY:
        values = [res[metric][keys.MRE] for res in results]
    elif metric in keys.HOTSPOT_ANALYSIS:
        values = [res[metric][str(HS_GRAN)] for res in results]
    elif metric in keys.TRAJ_PRESERVATION_METRICS_AGGREGATED:
        values = [res[metric][keys.MEAN] for res in results]
    else:
        values = [res[metric] for res in results]
    assert len(values) == NUM_RUNS, f"Number of values ({len(values)}) does not match NUM_RUNS ({NUM_RUNS})."
    return values


def get_metric_mean_value(results: Sequence[dict], metric: str) -> Tuple[float, float]:
    """
    Get the values for the metric from the results.

    :param results: List of results.
    :param metric: Metric to get the values for.
    :return: Mean and half-width of the confidence interval
    """
    values = get_metric_values(results, metric)

    m, c = mean_conf_half(values)
    assert m == np.mean(values), f"Mean value {m} does not match np.mean(values) {np.mean(values)}."
    return m, c


def format_num(value: float, digits: int = 3) -> str:
    """
    Format the float value.
    Show {digits} significant figures.

    :param value: Value to format.
    :param digits: Number of digits to show.
    """
    if value >= 100 and value < 10000:
        return f"{value:.0f}"
    return f"{value:.{digits}g}"


###############################################################################
# All Results Tables
###############################################################################
def print_result_table(cases: Sequence[str],
                       subheaders: Dict[str, str],
                       metrics: List[str] = PAPER_METRICS,
                       ) -> None:
    header = [
        "ID",
        # "Code",
        r"\glsshort{TM}",
        # Metrics
    ]
    rows = []
    header_columns = header + [get_metric_header(metric) for metric in metrics]

    best_values = {
        T: {
            metric: float("-inf") if METRICS_DIRECTION[metric].larger_is_better() else float("inf") for metric
            in metrics
        } for T in THREAT_MODELS
    }

    values: Dict[str, Dict[str, Optional[float]]] = {
        case: {metric: None for metric in metrics} for case in cases
    }

    ###############################################################################
    # Load results and compute best values for each metric
    ###############################################################################
    for case in cases:
        # Get the configuration
        config = final_evaluation[case]

        # Get the threat model
        tm = get_threat_model(config)

        # Load results
        results = load_results(config)

        if len(results) > 0:
            # Keep buffer row if no results are available yet
            for metric in metrics:
                v, _ = get_metric_mean_value(results, metric)
                # Save the value
                values[case][metric] = v
                # Check if this is the best value - only base cases considered!
                if is_base_case(case):
                    if METRICS_DIRECTION[metric].larger_is_better():
                        best_values[tm][metric] = max(best_values[tm][metric], v)
                    else:
                        best_values[tm][metric] = min(best_values[tm][metric], v)
                del v  # Make sure we don't accidentally use this value

    ###############################################################################
    # Assemble the rows and highlight the best values
    ###############################################################################
    for i, case in enumerate(cases):
        # Get the configuration
        config = final_evaluation[case]

        # Get the threat model
        tm = get_threat_model(config)

        # Identify the row
        row = [encode_id(case),
               # get_conf_code(config),
               cellcolor(THREAT_COLOR[tm]) + ballnumber(tm)]

        for metric in metrics:
            value = values[case][metric]

            if value is None:
                row.append("--")
            elif is_base_case(case) and is_better(value, best_values[tm][metric], metric):
                raise ValueError(
                    f"Value {value} is better than best value {best_values[tm][metric]} for {case} and {metric}")
            elif value == best_values[tm][metric] and is_base_case(case) and i < 36:
                # Highlight the best value if this is a base case
                row.append(textcolor(bold(format_num(value)), THREAT_COLOR[tm]))
            elif is_x_case(case) and is_better_equal(value, best_values[tm][metric], metric):
                # This is an X case - bold if better or equal to the best value for this threat model and metric
                row.append(bold(format_num(value)))
            else:
                row.append(format_num(value))

        rows.append(row)

    column_alignment = ['c'] * len(header_columns)
    print(to_latex(rows, header_columns, column_alignment, subheaders=subheaders))


###############################################################################
# Comparative Table
###############################################################################
def determine_group_indices(case: str, matching_cases: Sequence[Sequence[str]]) -> Optional[Union[int, List[int]]]:
    """
    Determine the group indices/index for the case.
    """
    group_idx = []
    for idx, group in enumerate(matching_cases):
        if case in group:
            group_idx.append(idx)
    if len(group_idx) == 0:
        return None
    elif len(group_idx) == 1:
        return group_idx[0]
    else:
        return group_idx


def determine_group_idx(case: str, matching_cases: Sequence[Sequence[str]]) -> int:
    """Same as determine_group_indices but only returns the index and raises an error otherwise."""
    group_idx = determine_group_indices(case, matching_cases)
    if group_idx is None:
        raise ValueError(f"Case {case} is not a member of exactly one group.")
    elif not isinstance(group_idx, int):
        raise ValueError(f"Case {case} is a member of multiple groups.")
    return group_idx


def comparative_table(cases: Sequence[str],
                      classes: Sequence[Sequence[str]],
                      subheaders: Dict[str, str],
                      metrics: List[str] = PAPER_METRICS,
                      additional_headers: List[str] = list(),
                      additional_values: Optional[Dict[str, Dict[str, str]]] = None,
                      conf_int: bool = False,
                      t_test: bool = False,
                      ) -> None:
    """
    Print a comparative table for the given cases.

    :param cases: Cases to include in the table in the order they should appear.
    :param classes: Sequence of classes to compare. Each class is a sequence of cases.
        For example, matching_cases = [[case1, case2, case3], [case4, case5]], would compare case1, case2, and
        case3, to each other, and case4 and case5 to each other.
        For each class, the best value for each metric is highlighted.
    :param subheaders: Subheaders to include in the table.
    :param metrics: Metrics to include in the table.
    :param additional_headers: Additional headers to include in the table.
    :param additional_values: Additional values to include in the table.
        {case: {header: value}}
    :param conf_int: Whether to include confidence intervals.
    :param t_test: Whether to include t-test results.
    """
    rows = []
    header = ["ID", "Code", ] + additional_headers
    header_columns = header + [get_metric_header(metric) for metric in metrics]

    # Stores the currently best value for each metric
    best_values = {
        group_idx: {
            metric: float("-inf") if METRICS_DIRECTION[metric].larger_is_better() else float("inf") for metric
            in metrics
        } for group_idx in range(len(classes))
    }

    # Stores which case is best for each metric
    best_cases: Dict[int, Dict[str, Optional[str]]] = {
        group_idx: {metric: None for metric in metrics} for group_idx in range(len(classes))}

    # Stores the values for each case and metric
    values: Dict[str, Dict[str, Optional[List[float]]]] = {
        case: {metric: None for metric in metrics} for case in cases
    }

    # Stores the confidence intervals for each case and metric
    conf: Dict[str, Dict[str, Optional[float]]] = {
        case: {metric: None for metric in metrics} for case in cases
    }

    ###############################################################################
    # Load results and compute best values for each metric
    ###############################################################################
    for case in cases:
        # Get the configuration
        config = final_evaluation[case]

        # Load results
        results = load_results(config)

        # Determine the group index (should be member of exactly one group)
        group_idx = determine_group_idx(case, classes)
        assert isinstance(group_idx, int), f"Case {case} is not a member of exactly one group."

        if len(results) > 0:
            # Keep buffer row if no results are available yet
            for metric in metrics:
                buf = get_metric_values(results, metric)
                # Save the value
                values[case][metric] = buf
                # Save confidence interval
                m, c = mean_conf_half(buf, confidence=CONF)
                conf[case][metric] = c
                # Check if this is the best value - only base cases considered!
                if METRICS_DIRECTION[metric].larger_is_better():
                    best_values[group_idx][metric] = max(best_values[group_idx][metric], m)
                else:
                    best_values[group_idx][metric] = min(best_values[group_idx][metric], m)
                # Update the best case in case this is the best value
                if is_better_equal(m, best_values[group_idx][metric], metric):
                    best_cases[group_idx][metric] = case
                # Make sure we don't accidentally use this value
                del m

    ###############################################################################
    # Assemble the rows and highlight the best values
    ###############################################################################

    for case in cases:
        # Get the configuration
        config = final_evaluation[case]

        # Identify the row
        row = [encode_id(case), get_conf_code(config)]

        # Add additional values if available
        if additional_values:
            row += [additional_values[case][header] for header in additional_headers]

        for metric in metrics:
            res = values[case][metric]
            if res is None:
                txt = "--"
                break
            m, c = mean_conf_half(res, confidence=CONF)

            group_idx = determine_group_idx(case, classes)
            if is_better(m, best_values[group_idx][metric], metric):
                raise ValueError(
                    f"Value {m} is better than best value {best_values[group_idx][metric]} for {case} and {metric}")
            elif m == best_values[group_idx][metric]:
                # Check if this value is significantly better compared to any other in this group
                group_values = {case: values[case][metric] for case in classes[group_idx]}
                significant: bool = True if not t_test else ttest_group(group_values, case, classes[group_idx])
                # Highlight the best value
                if significant:
                    txt = bold(format_num(m))
                else:
                    txt = format_num(m)
            else:
                txt = format_num(m)

            # Add confidence interval if available
            if conf_int:
                txt = '$' + txt + r" \pm " + format_num(c) + '$'
            row.append(txt)

        rows.append(row)

    column_alignment = ['c'] * len(header_columns)
    print(to_latex(rows, header_columns, column_alignment, subheaders=subheaders))
