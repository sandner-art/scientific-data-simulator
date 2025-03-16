# simulator/visualization.py
import matplotlib.pyplot as plt
import plotly.express as px
import os
from typing import Dict, Any
from .utils import DataType  # Import DataType
import pandas as pd

def generate_plots(results: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Generates plots based on the results and their DataDescriptors.
    """

    for data_name, data_info in results.items():
      descriptor = data_info['descriptor']
      data = data_info['data']

      if descriptor.group == "time_series":
          x_axis_name =  'time'
          if 'time' not in results:
              x_axis_name = descriptor.x_axis if descriptor.x_axis else None
              if x_axis_name is None:
                  print(f"Warning: X-axis data not found and 'time' data not found. Skipping time series plot for {data_name}.")
                  continue
          if x_axis_name not in results:
                print(f"Warning: X-axis data '{x_axis_name}' not found. Skipping time series plot for {data_name}.")
                continue
          x_axis_data = results[x_axis_name]['data']
          x_axis_descriptor = results[x_axis_name]['descriptor']

          if descriptor.plot_type == "line":
            # Check if observed data exists, and plot accordingly
            if "observed_data" in results and data_name in results['observed_data']['data'].columns: # Check if the column exists

                # Matplotlib
                plt.figure()
                plt.plot(x_axis_data, data, label=f"Simulated {data_name}")
                plt.plot(results['observed_data']['data']['time'], results['observed_data']['data'][data_name], label=f"Observed {data_name}", linestyle='--')
                plt.xlabel(x_axis_descriptor.units if x_axis_descriptor.units else x_axis_name)
                plt.ylabel(descriptor.units if descriptor.units else descriptor.name)
                plt.title(f"{descriptor.name} vs. {x_axis_name}")
                plt.legend()  # Add a legend
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f"{data_name}_matplotlib.png"))
                plt.close()

                # Plotly
                # Convert to DataFrame for easier plotting with Plotly
                df = pd.DataFrame({
                  x_axis_name: x_axis_data,
                  f"Simulated {data_name}": data
                })

                # Add observed data, if exist.
                if 'observed_data' in results:
                    if data_name in results['observed_data']['data'].columns:
                        observed_df = results['observed_data']['data']
                        # Check if the lengths of the time arrays are compatible, take minimum.
                        min_len = min(len(x_axis_data), len(observed_df['time']))
                        df = df.iloc[:min_len] # Trim df
                        df[f"Observed {data_name}"] = observed_df[data_name].iloc[:min_len] # Trim data

                fig = px.line(df, x=x_axis_name, y=[col for col in df.columns if col != x_axis_name],
                              labels={'x': x_axis_descriptor.units if x_axis_descriptor.units else x_axis_name,
                                      'value': descriptor.units if descriptor.units else descriptor.name,
                                      'variable': 'Series'},
                              title=f"{descriptor.name} vs. {x_axis_name}")
                fig.write_html(os.path.join(output_dir, f"{data_name}_plotly.html"))
            else: # No observed data
                # Matplotlib (static)
                plt.figure()
                plt.plot(x_axis_data, data)
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