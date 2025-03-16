# simulator/visualization.py
import matplotlib.pyplot as plt
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
            first_data_info = data_list[0]
            first_descriptor = first_data_info['descriptor']
            # Determine x_axis_name:
            x_axis_name =  'time'
            if 'time' not in results: # Check time axis
                x_axis_name = first_descriptor.x_axis if first_descriptor.x_axis else None
                if x_axis_name is None:
                    print(f"Warning: X-axis data not found and 'time' data not found. Skipping time series plot for {group_name}.")
                    continue # Skip the entire group
            if x_axis_name not in results:
                print(f"Warning: X-axis data '{x_axis_name}' not found. Skipping time series plot for {group_name}.")
                continue

            x_axis_data = results[x_axis_name]['data']
            x_axis_descriptor = results[x_axis_name]['descriptor']

            # --- Matplotlib ---
            plt.figure()
            for data_info in data_list:
                descriptor = data_info['descriptor']
                if descriptor.plot_type == "line" and descriptor.group == "time_series":
                    data = data_info['data']
                    plt.plot(x_axis_data, data, label=descriptor.name)

            plt.xlabel(x_axis_descriptor.units if x_axis_descriptor.units else x_axis_name)
            plt.ylabel("Value")  # Generic y-axis label
            plt.title(f"Time Series Plot ({group_name})")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{group_name}_matplotlib.png"))
            plt.close()

            # --- Plotly ---
            # Use a DataFrame for easier plotting
            df = pd.DataFrame()
            df[x_axis_name] = x_axis_data  # Always include the x-axis
            for data_info in data_list:
                descriptor = data_info['descriptor']
                if descriptor.plot_type == 'line' and descriptor.group == "time_series":
                    data = data_info['data']
                    df[descriptor.name] = data

            # Corrected y label
            fig = px.line(df, x=x_axis_name, y=[col for col in df.columns if col != x_axis_name], # all columns except x
                            labels={'x': x_axis_descriptor.units if x_axis_descriptor.units else x_axis_name,
                                    'y': "Value",  # Generic y-axis label
                                    'variable': 'Series'},
                            title=f"Time Series Plot ({group_name})")
            fig.write_html(os.path.join(output_dir, f"{group_name}_plotly.html"))
            if static_format:
                # Explicitly set mathjax to None when using a static renderer
                fig.write_image(os.path.join(output_dir, f"{group_name}_plotly.{static_format}"), format=static_format, mathjax=None)


        elif group_name == 'histogram':
            # Matplotlib
            plt.figure()
            for data_info in data_list:
                descriptor = data_info['descriptor']
                data = data_info['data']
                plt.hist(data, bins='auto', label=descriptor.name, alpha=0.7) # Transparency for multiple histograms
            plt.xlabel("Value") # Generic
            plt.ylabel("Frequency")
            plt.title(f"Histogram ({group_name})")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{group_name}_matplotlib.png"))
            plt.close()

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
                fig.write_image(os.path.join(output_dir, f"{group_name}_plotly.{static_format}"), format=static_format, mathjax=None)

        # Add support for other groups as needed

    # Add combined plot for Predator-Prey (and similar scenarios)
    if "prey_population" in results and "predator_population" in results:
        _generate_combined_plot(results, output_dir, static_format) # Pass static_format

def _generate_combined_plot(results: Dict[str, Dict[str, Any]], output_dir: str, static_format: str = "svg"):
    """Generates a combined plot of prey and predator populations."""

    # Matplotlib
    plt.figure()
    plt.plot(results['time']['data'], results['prey_population']['data'], label="Prey Population")
    plt.plot(results['time']['data'], results['predator_population']['data'], label="Predator Population")
    if 'observed_data' in results:
        if 'prey_population' in results['observed_data']['data'].columns:
            plt.plot(results['observed_data']['data']['time'], results['observed_data']['data']['prey_population'], label="Observed Prey", linestyle='--')
        if 'predator_population' in results['observed_data']['data'].columns:
            plt.plot(results['observed_data']['data']['time'], results['observed_data']['data']['predator_population'], label="Observed Predator", linestyle='--')

    plt.xlabel(results['time']['descriptor'].units if results['time']['descriptor'].units else 'Time')
    plt.ylabel("Population")
    plt.title("Predator-Prey Population Dynamics")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "combined_populations_matplotlib.png"))
    plt.close()

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
      fig.write_image(os.path.join(output_dir, f"combined_populations_plotly.{static_format}"), format=static_format, mathjax=None)