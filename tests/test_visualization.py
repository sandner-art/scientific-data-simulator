# tests/test_visualization.py
import pytest
from simulator.visualization import generate_plots
from simulator.utils import DataDescriptor, DataType
import os
import numpy as np
import pandas as pd

# Fixture to create dummy results for testing
@pytest.fixture
def example_results():
    return {
        "time": {
            "data": np.array([0, 1, 2, 3, 4]),
            "descriptor": DataDescriptor("time", DataType.NDARRAY, shape=(5,), units="seconds", group="time_series")
        },
        "value1": {
            "data": np.array([10, 12, 11, 14, 13]),
            "descriptor": DataDescriptor("value1", DataType.NDARRAY, shape=(5,), units="m", group="time_series", plot_type="line", x_axis="time")
        },
        "value2": {
            "data": np.array([20, 18, 19, 16, 17]),
            "descriptor": DataDescriptor("value2", DataType.NDARRAY, shape=(5,), units="cm", group="time_series", plot_type="line", x_axis="time")
        },
        "value_hist":{
            "data": np.array([1,1,4,5,6,7,8,8,8]),
            "descriptor": DataDescriptor("value_hist", DataType.NDARRAY, group='histogram', plot_type='histogram')
        }
    }
@pytest.fixture
def example_results_no_time():
    return {
        "step": {
            "data": np.array([0, 1, 2, 3, 4]),
            "descriptor": DataDescriptor("step", DataType.NDARRAY, shape=(5,), units="steps", group="time_series")
        },
        "value1": {
            "data": np.array([10, 12, 11, 14, 13]),
            "descriptor": DataDescriptor("value1", DataType.NDARRAY, shape=(5,), units="m", group="time_series", plot_type="line", x_axis="step")
        },
        "value2": {
            "data": np.array([20, 18, 19, 16, 17]),
            "descriptor": DataDescriptor("value2", DataType.NDARRAY, shape=(5,), units="cm", group="time_series", plot_type="line", x_axis="step")
        }
    }

@pytest.fixture
def example_results_predator_prey():
    # Create dummy data for predator-prey (including observed data)
    time_data = np.arange(0, 10)
    prey_simulated = np.array([100, 90, 80, 75, 72, 70, 68, 71, 75, 80])
    predator_simulated = np.array([20, 22, 25, 28, 30, 31, 30, 28, 25, 22])
    prey_observed = np.array([102, 88, 78, 77, 75, 73, 70, 72, 77, 82])
    predator_observed = np.array([19, 21, 26, 29, 32, 33, 31, 29, 26, 23])

    observed_df = pd.DataFrame({
        'time': time_data,
        'prey_population': prey_observed,
        'predator_population': predator_observed
    })
    return {
        "time": {
            "data": time_data,
            "descriptor": DataDescriptor("time", DataType.NDARRAY, shape=time_data.shape, units="steps", group="time_series")
        },
        "prey_population": {
            "data": prey_simulated,
            "descriptor": DataDescriptor("prey_population", DataType.NDARRAY, shape=prey_simulated.shape, units="individuals", group="time_series", plot_type="line", x_axis="time")
        },
        "predator_population": {
            "data": predator_simulated,
            "descriptor": DataDescriptor("predator_population", DataType.NDARRAY, shape=predator_simulated.shape, units="individuals", group="time_series", plot_type="line", x_axis="time")
        },
        'observed_data': {
            'data': observed_df,
            'descriptor': DataDescriptor("observed_data", DataType.DATAFRAME, group='observed')
        }
    }

def test_generate_plots_time_series(example_results, tmp_path):
    """Test generating time series plots."""
    output_dir = str(tmp_path)
    generate_plots(example_results, output_dir)

    # Check that the expected files were created
    assert os.path.exists(os.path.join(output_dir, "time_series_matplotlib.png"))
    assert os.path.exists(os.path.join(output_dir, "time_series_plotly.html"))
    assert os.path.exists(os.path.join(output_dir, "time_series_plotly.svg")) # Check for default static plot

def test_generate_plots_histogram(example_results, tmp_path):
    output_dir = str(tmp_path)
    generate_plots(example_results, output_dir)

    assert os.path.exists(os.path.join(output_dir, "histogram_matplotlib.png"))
    assert os.path.exists(os.path.join(output_dir, "histogram_plotly.html"))

def test_generate_plots_no_time(example_results_no_time, tmp_path):
    output_dir = str(tmp_path)
    generate_plots(example_results_no_time, output_dir)
    assert os.path.exists(os.path.join(output_dir, "time_series_matplotlib.png"))
    assert os.path.exists(os.path.join(output_dir, "time_series_plotly.html"))

def test_generate_plots_predator_prey(example_results_predator_prey, tmp_path):
    """Test combined plot generation for predator-prey."""
    output_dir = str(tmp_path)
    generate_plots(example_results_predator_prey, output_dir)

    # Check for combined plot files
    assert os.path.exists(os.path.join(output_dir, "combined_populations_matplotlib.png"))
    assert os.path.exists(os.path.join(output_dir, "combined_populations_plotly.html"))
    assert os.path.exists(os.path.join(output_dir, "combined_populations_plotly.svg"))

    # Check for individual plot files as well, in case of future modifications
    assert os.path.exists(os.path.join(output_dir, "time_series_matplotlib.png"))
    assert os.path.exists(os.path.join(output_dir, "time_series_plotly.html"))

def test_generate_plots_custom_format(example_results, tmp_path):
    output_dir = str(tmp_path)
    generate_plots(example_results, output_dir, static_format='png')
    assert os.path.exists(os.path.join(output_dir, "time_series_plotly.png")) # check for png
    assert not os.path.exists(os.path.join(output_dir, "time_series_plotly.svg")) # check of no svg

def test_generate_plots_invalid_format(example_results, tmp_path):
    output_dir = str(tmp_path)
    with pytest.raises(ValueError):
        generate_plots(example_results, output_dir, static_format='invalid')