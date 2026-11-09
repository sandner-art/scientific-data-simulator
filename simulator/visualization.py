# simulator/visualization.py
# Figures are drawn with matplotlib.figure.Figure, never pyplot: the simulator only writes files,
# so it must not pick or start a GUI backend (Tk on Windows can fail to start, or open windows).
import matplotlib
from matplotlib.figure import Figure
import plotly.express as px
import plotly.io as pio  # Import plotly.io for static image export
import os
from typing import Dict, Any, List
from .utils import DataType
import pandas as pd
import numpy as np

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


    # Group data items by their 'group' field
    grouped_data: Dict[str, List[Dict[str, Any]]] = {}
    for data_name, data_info in results.items():
        descriptor = data_info['descriptor']
        if descriptor.group not in grouped_data:
            grouped_data[descriptor.group] = []
        grouped_data[descriptor.group].append(data_info)

    for group_name, data_list in grouped_data.items():
        if group_name == "time_series":
            _time_series_plot(results, group_name, data_list, output_dir, static_format)

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

            # Plotly
            # Combine into DataFrame for plotting
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
                # Explicitly set mathjax to None when using a static renderer
                fig.write_image(os.path.join(output_dir, f"{group_name}_plotly.{static_format}"), format=static_format)

        # Add support for other groups as needed

    # Add combined plot for Predator-Prey (and similar scenarios)
    if "prey_population" in results and "predator_population" in results:
        _generate_combined_plot(results, output_dir, static_format) # Pass static_format

def _time_series_plot(results, group_name, data_list, output_dir, static_format):
    """Plot each line/scatter series of the group against its own x axis
    (``descriptor.x_axis``, default ``time``). Observed series that were
    sampled at a subset of points carry their own coordinate, so latent and
    observed series can share one figure."""
    import plotly.graph_objects as go

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

    fig = go.Figure()
    for descriptor, x_info, data in series:
        mode = "markers" if descriptor.plot_type == "scatter" else "lines"
        fig.add_trace(go.Scatter(x=np.asarray(x_info['data']), y=np.asarray(data), mode=mode, name=descriptor.name))
    fig.update_layout(title=f"Time Series Plot ({group_name})", xaxis_title=x_label,
                      yaxis_title="Value", legend_title_text="Series")
    fig.write_html(os.path.join(output_dir, f"{group_name}_plotly.html"))
    if static_format:
        fig.write_image(os.path.join(output_dir, f"{group_name}_plotly.{static_format}"), format=static_format)


def _generate_combined_plot(results: Dict[str, Dict[str, Any]], output_dir: str, static_format: str = "svg"):
    """Generates a combined plot of prey and predator populations."""

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

    # Plotly
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
      # Explicitly set mathjax to None when using a static renderer
      fig.write_image(os.path.join(output_dir, f"combined_populations_plotly.{static_format}"), format=static_format)