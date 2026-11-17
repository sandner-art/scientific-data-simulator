# scidatasim/visualization.py
# Figures are drawn with matplotlib.figure.Figure, never pyplot: the simulator only writes files,
# so it must not pick or start a GUI backend (Tk on Windows can fail to start, or open windows).
# Interactive HTML versions come from plotly, an optional extra ("standard" profile): without it
# only the matplotlib PNGs are written.
import logging
import matplotlib
from matplotlib.figure import Figure
import os
from typing import Dict, Any, List
from .utils import DataType
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
_PLOTLY_MISSING_LOGGED = False


def _plotly():
    """``(plotly.express, plotly.graph_objects)``, or None when plotly is not installed."""
    global _PLOTLY_MISSING_LOGGED
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        if not _PLOTLY_MISSING_LOGGED:
            logger.info("plotly is not installed: interactive HTML plots are skipped "
                        "(pip install 'scientific-data-simulator[standard]')")
            _PLOTLY_MISSING_LOGGED = True
        return None
    return px, go


def _write_static(fig, path: str, static_format: str, failed: List[str], attempts: int = 2) -> bool:
    """Write a static image of a plotly figure through kaleido and Chrome.

    That route can fail for reasons outside the simulator: Chrome missing, or a race in kaleido
    (1.5.1 sometimes ran a figure before its page script had loaded). The data and the HTML plot
    are already written by then, so retry once, then note the file in ``failed`` and go on with
    the other plots; ``generate_plots`` reports the failures together at the end.
    """
    error = None
    for _ in range(attempts):
        try:
            fig.write_image(path, format=static_format)
            return True
        except Exception as e:  # kaleido raises RuntimeError subclasses; report, do not lose the run
            error = e
    failed.append(f"{os.path.basename(path)} ({type(error).__name__}: {str(error).strip()[:200]})")
    return False


def generate_plots(results: Dict[str, Dict[str, Any]], output_dir: str, static_format: str = "svg"):
    """
    Generates plots based on the results and their DataDescriptors.

    Args:
        results: A dictionary of experiment results.
        output_dir: The directory where plots should be saved.
        static_format: The format for static image export (svg, pdf, png, jpeg, webp).
                       Defaults to "svg".  Set to None to disable static export.
    """

    if static_format not in [None, "svg", "pdf", "png", "jpeg", "webp"]:
        raise ValueError(f"Invalid static_format: {static_format}.  Must be one of None, 'svg', 'pdf', 'png', 'jpeg', 'webp'.")


    failed: List[str] = []  # static exports that did not succeed

    # Group data items by their 'group' field
    grouped_data: Dict[str, List[Dict[str, Any]]] = {}
    for data_name, data_info in results.items():
        descriptor = data_info['descriptor']
        if descriptor.group not in grouped_data:
            grouped_data[descriptor.group] = []
        grouped_data[descriptor.group].append(data_info)

    for group_name, data_list in grouped_data.items():
        if group_name == "time_series":
            _time_series_plot(results, group_name, data_list, output_dir, static_format, failed)

        elif group_name == 'histogram':
            # Matplotlib
            figure = Figure()
            ax = figure.add_subplot()
            for data_info in data_list:
                descriptor = data_info['descriptor']
                data = data_info['data']
                ax.hist(data, bins='auto', label=descriptor.name, alpha=0.7) # Transparency for multiple histograms
            ax.set_xlabel("Value") # Generic
            ax.set_ylabel("Frequency")
            ax.set_title(f"Histogram ({group_name})")
            ax.legend()
            ax.grid(True)
            figure.savefig(os.path.join(output_dir, f"{group_name}_matplotlib.png"))

            # Plotly (optional)
            plotly = _plotly()
            if plotly is None:
                continue
            px = plotly[0]
            hist_df = pd.DataFrame()
            for data_info in data_list:
                descriptor = data_info['descriptor']
                data = data_info['data']
                hist_df[descriptor.name] = pd.Series(data) # Use Series, as data can be different length


            fig = px.histogram(hist_df, nbins=30, #  fixed number of bins
                              labels={'value': "Value", 'variable': 'Series'},
                              title=f"Histogram ({group_name})",
                              marginal="rug",  # Add marginal distributions
                              opacity=0.7) # transparency
            fig.write_html(os.path.join(output_dir, f"{group_name}_plotly.html"))
            if static_format:
                _write_static(fig, os.path.join(output_dir, f"{group_name}_plotly.{static_format}"),
                              static_format, failed)

        # Add support for other groups as needed

    # Add combined plot for Predator-Prey (and similar scenarios)
    if "prey_population" in results and "predator_population" in results:
        _generate_combined_plot(results, output_dir, static_format, failed)

    if failed:
        raise RuntimeError("static plot export failed for " + "; ".join(failed)
                           + ". The data and the other plots are written; `sds doctor` checks kaleido and Chrome.")

def _time_series_plot(results, group_name, data_list, output_dir, static_format, failed):
    """Plot each line/scatter series of the group against its own x axis
    (``descriptor.x_axis``, default ``time``). Observed series that were
    sampled at a subset of points carry their own coordinate, so latent and
    observed series can share one figure."""

    series = []
    for data_info in data_list:
        descriptor = data_info['descriptor']
        if descriptor.plot_type not in ("line", "scatter"):
            continue
        x_name = descriptor.x_axis or 'time'
        if x_name not in results:
            print(f"Warning: X-axis data '{x_name}' not found for '{descriptor.name}'. Skipping it.")
            continue
        series.append((descriptor, results[x_name], data_info['data']))
    if not series:
        return

    x_descriptor = series[0][1]['descriptor']
    x_label = x_descriptor.units if x_descriptor.units else (series[0][0].x_axis or 'time')

    # Observed series share the colour of the latent series they observe.
    palette = matplotlib.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])
    colours, next_colour = {}, 0
    for descriptor, _, _ in series:
        base = descriptor.name[:-len("_observed")] if descriptor.name.endswith("_observed") else descriptor.name
        if base not in colours:
            colours[base] = palette[next_colour % len(palette)]
            next_colour += 1

    figure = Figure()
    ax = figure.add_subplot()
    for descriptor, x_info, data in series:
        base = descriptor.name[:-len("_observed")] if descriptor.name.endswith("_observed") else descriptor.name
        if descriptor.plot_type == "scatter":
            ax.plot(x_info['data'], data, linestyle="none", marker="o", markersize=3,
                    color=colours[base], alpha=0.8, label=descriptor.name)
        else:
            ax.plot(x_info['data'], data, color=colours[base], label=descriptor.name)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Value")
    ax.set_title(f"Time Series Plot ({group_name})")
    ax.legend()
    ax.grid(True)
    figure.savefig(os.path.join(output_dir, f"{group_name}_matplotlib.png"))

    plotly = _plotly()
    if plotly is None:
        return
    go = plotly[1]
    fig = go.Figure()
    for descriptor, x_info, data in series:
        mode = "markers" if descriptor.plot_type == "scatter" else "lines"
        fig.add_trace(go.Scatter(x=np.asarray(x_info['data']), y=np.asarray(data), mode=mode, name=descriptor.name))
    fig.update_layout(title=f"Time Series Plot ({group_name})", xaxis_title=x_label,
                      yaxis_title="Value", legend_title_text="Series")
    fig.write_html(os.path.join(output_dir, f"{group_name}_plotly.html"))
    if static_format:
        _write_static(fig, os.path.join(output_dir, f"{group_name}_plotly.{static_format}"), static_format, failed)


def _generate_combined_plot(results: Dict[str, Dict[str, Any]], output_dir: str, static_format: str = "svg",
                            failed: List[str] = None):
    """Generates a combined plot of prey and predator populations."""
    failed = [] if failed is None else failed

    # Matplotlib
    figure = Figure()
    ax = figure.add_subplot()
    ax.plot(results['time']['data'], results['prey_population']['data'], label="Prey Population")
    ax.plot(results['time']['data'], results['predator_population']['data'], label="Predator Population")
    if 'observed_data' in results:
        if 'prey_population' in results['observed_data']['data'].columns:
            ax.plot(results['observed_data']['data']['time'], results['observed_data']['data']['prey_population'], label="Observed Prey", linestyle='--')
        if 'predator_population' in results['observed_data']['data'].columns:
            ax.plot(results['observed_data']['data']['time'], results['observed_data']['data']['predator_population'], label="Observed Predator", linestyle='--')

    ax.set_xlabel(results['time']['descriptor'].units if results['time']['descriptor'].units else 'Time')
    ax.set_ylabel("Population")
    ax.set_title("Predator-Prey Population Dynamics")
    ax.legend()
    ax.grid(True)
    figure.savefig(os.path.join(output_dir, "combined_populations_matplotlib.png"))

    # Plotly (optional)
    plotly = _plotly()
    if plotly is None:
        return
    px = plotly[0]
    df = pd.DataFrame({
      'time': results['time']['data'],
      'Prey Population': results['prey_population']['data'],
      'Predator Population': results['predator_population']['data']
    })
    if 'observed_data' in results:
        obs_df = results['observed_data']['data']
        if 'prey_population' in obs_df.columns:
          df['Observed Prey'] = obs_df.set_index('time')['prey_population']
          df['Observed Prey'] = pd.to_numeric(df['Observed Prey'], errors='coerce')
        if 'predator_population' in obs_df.columns:
          df['Observed Predator'] = obs_df.set_index('time')['predator_population']
          df['Observed Predator'] = pd.to_numeric(df['Observed Predator'], errors='coerce')


    fig = px.line(df, x='time', y = [col for col in df.columns if col != 'time'],
                    labels={'time': results['time']['descriptor'].units if results['time']['descriptor'].units else 'Time',
                            'value': 'Population',
                            'variable': 'Population Type'},
                    title="Predator-Prey Population Dynamics")
    fig.write_html(os.path.join(output_dir, "combined_populations_plotly.html"))
    if static_format:
        _write_static(fig, os.path.join(output_dir, f"combined_populations_plotly.{static_format}"),
                      static_format, failed)