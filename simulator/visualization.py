# simulator/visualization.py
import matplotlib.pyplot as plt
import plotly.express as px
import os
from typing import Dict, Any
from .utils import DataType  # Import DataType


def generate_plots(results: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Generates plots based on the results and their DataDescriptors.

    Args:
        results: A dictionary of experiment results, as returned by
                 ExperimentLogic.get_results().
        output_dir: The directory where plots should be saved.
    """

    for data_name, data_info in results.items():
        descriptor = data_info['descriptor']
        data = data_info['data']

        if descriptor.group == "time_series":
            # --- MODIFIED: Use x_axis if provided for Y data, otherwise default to 'time' for x data ---
            x_axis_name =  'time'  # Default x-axis name
            if 'time' not in results and descriptor.plot_type != "histogram": # if no time, we check if there is other option for x.
                x_axis_name = descriptor.x_axis if descriptor.x_axis else None # first check user-specified axis
                if x_axis_name is None: # If user did not specify, we through warning.
                    print(f"Warning: X-axis data not found and 'time' data not found. Skipping time series plot for {data_name}.")
                    continue
            if x_axis_name not in results and descriptor.plot_type != "histogram":
                print(f"Warning: X-axis data '{x_axis_name}' not found. Skipping time series plot for {data_name}.")
                continue

            if descriptor.plot_type != "histogram": # we plot histogram only if not timeseries
                x_axis_data = results[x_axis_name]['data']
                x_axis_descriptor = results[x_axis_name]['descriptor']
            # --- END MODIFIED ---

            if descriptor.plot_type == "line":
                # Matplotlib (static)
                plt.figure()
                plt.plot(x_axis_data, data)  # Use x_axis_data
                plt.xlabel(x_axis_descriptor.units if x_axis_descriptor.units else x_axis_name)
                plt.ylabel(descriptor.units if descriptor.units else descriptor.name)
                plt.title(f"{descriptor.name} vs. {x_axis_name}")  # Updated title
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f"{data_name}_matplotlib.png"))
                plt.close()

                # Plotly (interactive)
                fig = px.line(x=x_axis_data, y=data,  # Use x_axis_data
                              labels={'x': x_axis_descriptor.units if x_axis_descriptor.units else x_axis_name,
                                      'y': descriptor.units if descriptor.units else descriptor.name},
                              title=f"{descriptor.name} vs. {x_axis_name}")  # Updated title
                fig.write_html(os.path.join(output_dir, f"{data_name}_plotly.html"))

            elif descriptor.plot_type == "scatter":
                # Similar changes for scatter plots...
                plt.figure()
                plt.scatter(x_axis_data, data)  # Use x_axis_data
                plt.xlabel(x_axis_descriptor.units if x_axis_descriptor.units else x_axis_name)
                plt.ylabel(descriptor.units if descriptor.units else descriptor.name)
                plt.title(f"{descriptor.name} vs. {x_axis_name}")  # Updated title
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f"{data_name}_matplotlib.png"))
                plt.close()

                # Plotly (interactive)
                fig = px.scatter(x=x_axis_data, y=data,
                                     labels={'x': x_axis_descriptor.units if x_axis_descriptor.units else x_axis_name,
                                             'y': descriptor.units if descriptor.units else descriptor.name},
                                     title=f"{descriptor.name} vs. {x_axis_name}")
                fig.write_html(os.path.join(output_dir, f"{data_name}_plotly.html"))

            elif descriptor.group == "histogram":
                # ... (rest of the histogram plotting code - no changes needed here) ...
                plt.figure()
                plt.hist(data, bins='auto')  # 'auto' for automatic bin selection
                plt.xlabel(descriptor.units if descriptor.units else descriptor.name)
                plt.ylabel("Frequency")
                plt.title(f"Histogram of {descriptor.name}")
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f"{data_name}_matplotlib.png"))
                plt.close()

                # Plotly
                fig = px.histogram(x=data, labels={'x': descriptor.units if descriptor.units else descriptor.name},
                                  title=f"Histogram of {descriptor.name}")
                fig.write_html(os.path.join(output_dir, f"{data_name}_plotly.html"))