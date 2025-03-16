# Example 3: Predator-Prey Simulation

This example demonstrates a simulation of predator-prey population dynamics using the Lotka-Volterra equations.  It uses the `experiments.predator_prey.logic.PredatorPreyExperiment` class.

## Configuration

The `config.yaml` file allows you to configure the following parameters:

*   `n_steps`: The number of simulation steps.
*   `initial_prey`: The initial population of prey.
*   `initial_predators`: The initial population of predators.
*   `prey_growth_rate`: The growth rate of the prey population.
*   `prey_death_rate`: The rate at which prey are killed by predators.
*   `predator_growth_rate`: The growth rate of the predator population (dependent on prey).
*   `predator_death_rate`: The natural death rate of predators.

## Running the Example

To run this example, navigate to the project root directory in your terminal and execute:

```bash
python -m examples.example_3.run_experiment

```

This will generate an experiment_record.json file and plot files in the experiments_output directory.

