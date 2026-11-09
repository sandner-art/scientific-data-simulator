# tests/test_visualization.py
import pytest
from simulator.visualization import generate_plots
from simulator.utils import DataDescriptor, DataType
import os
import numpy as np
import pandas as pd

def _static_export_available():
    """Static plotly export needs kaleido (and, for kaleido >= 1.0, Chrome)."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        pio.to_image(go.Figure(), format="png")
        return True
    except Exception:
        return False


requires_static_export = pytest.mark.skipif(
    not _static_export_available(), reason="static plotly export unavailable (kaleido/Chrome)")


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
            "descriptor": DataDescriptor("step", DataType.NDARRAY, shape=(5,), units="steps", group="time_series") # Removed x_axis
        },
        "value1": {
            "data": np.array([10, 12, 11, 14, 13]),
            "descriptor": DataDescriptor("value1", DataType.NDARRAY, shape=(5,), units="m", group="time_series", plot_type="line") # remove x_axis
        },
        "value2": {
            "data": np.array([20, 18, 19, 16, 17]),
            "descriptor": DataDescriptor("value2", DataType.NDARRAY, shape=(5,), units="cm", group="time_series", plot_type="line") # remove x_axis
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
    generate_plots(example_results, output_dir, static_format=None)

    # Check that the expected files were created
    assert os.path.exists(os.path.join(output_dir, "time_series_matplotlib.png"))
    assert os.path.exists(os.path.join(output_dir, "time_series_plotly.html"))
    assert not os.path.exists(os.path.join(output_dir, "time_series_plotly.svg"))


@requires_static_export
def test_generate_plots_default_static_svg(example_results, tmp_path):
    output_dir = str(tmp_path)
    generate_plots(example_results, output_dir)
    assert os.path.exists(os.path.join(output_dir, "time_series_plotly.svg"))  # default static format

def test_generate_plots_histogram(example_results, tmp_path):
    output_dir = str(tmp_path)
    generate_plots(example_results, output_dir, static_format=None)

    assert os.path.exists(os.path.join(output_dir, "histogram_matplotlib.png"))
    assert os.path.exists(os.path.join(output_dir, "histogram_plotly.html"))

def test_generate_plots_no_default_time(example_results_no_time, tmp_path):
    output_dir = str(tmp_path)
    generate_plots(example_results_no_time, output_dir, static_format=None)
    assert not os.path.exists(os.path.join(output_dir, "time_series_matplotlib.png"))
    assert not os.path.exists(os.path.join(output_dir, "time_series_plotly.html"))

def test_generate_plots_no_x(example_results_no_time, tmp_path):
    output_dir = str(tmp_path)
    generate_plots(example_results_no_time, output_dir, static_format=None)
    assert not os.path.exists(os.path.join(output_dir, "time_series_matplotlib.png"))
    assert not os.path.exists(os.path.join(output_dir, "time_series_plotly.html"))

def test_generate_plots_predator_prey(example_results_predator_prey, tmp_path):
    """Test combined plot generation for predator-prey."""
    output_dir = str(tmp_path)
    generate_plots(example_results_predator_prey, output_dir, static_format=None)

    # Check for combined plot files
    assert os.path.exists(os.path.join(output_dir, "combined_populations_matplotlib.png"))
    assert os.path.exists(os.path.join(output_dir, "combined_populations_plotly.html"))

    # Check for individual plot files as well, in case of future modifications
    assert os.path.exists(os.path.join(output_dir, "time_series_matplotlib.png"))
    assert os.path.exists(os.path.join(output_dir, "time_series_plotly.html"))

@requires_static_export
def test_generate_plots_predator_prey_static(example_results_predator_prey, tmp_path):
    output_dir = str(tmp_path)
    generate_plots(example_results_predator_prey, output_dir)
    assert os.path.exists(os.path.join(output_dir, "combined_populations_plotly.svg"))


@requires_static_export
def test_generate_plots_custom_format(example_results, tmp_path):
    output_dir = str(tmp_path)
    generate_plots(example_results, output_dir, static_format='png')
    assert os.path.exists(os.path.join(output_dir, "time_series_plotly.png")) # check for png
    assert not os.path.exists(os.path.join(output_dir, "time_series_plotly.svg")) # check of no svg

def test_generate_plots_invalid_format(example_results, tmp_path):
    output_dir = str(tmp_path)
    with pytest.raises(ValueError):
        generate_plots(example_results, output_dir, static_format='invalid')

def test_plots_never_load_pyplot(tmp_path):
    """The simulator only writes image files: it must not pick a GUI backend (pyplot chose Tk on
    a Windows runner whose Tcl was broken, and on a desktop it can open windows)."""
    import subprocess
    import sys
    code = (
        "import sys, numpy as np\n"
        "from simulator.utils import DataDescriptor, DataType\n"
        "from simulator.visualization import generate_plots\n"
        "t = np.arange(5.0)\n"
        "results = {'time': {'data': t, 'descriptor': DataDescriptor('time', DataType.NDARRAY, group='time_series')},\n"
        "           'v': {'data': t ** 2, 'descriptor': DataDescriptor('v', DataType.NDARRAY, group='time_series',\n"
        "                 plot_type='line', x_axis='time')},\n"
        "           'h': {'data': t, 'descriptor': DataDescriptor('h', DataType.NDARRAY, group='histogram',\n"
        "                 plot_type='histogram')}}\n"
        f"generate_plots(results, {str(tmp_path)!r}, static_format=None)\n"
        "assert 'matplotlib.pyplot' not in sys.modules, 'pyplot was imported'\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert {"time_series_matplotlib.png", "histogram_matplotlib.png"} <= set(os.listdir(tmp_path))
