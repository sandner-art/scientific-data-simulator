    # Predator-Prey Experiment

    This directory contains the `ExperimentLogic` implementation for a predator-prey simulation using the Lotka-Volterra equations.

    ## Files

    *   `logic.py`: Contains the `PredatorPreyExperiment` class.

    ## Model Description

    The simulation models the population dynamics of prey and predators.  The model uses the classic Lotka-Volterra equations:

    ```
    dPrey/dt =  prey_growth_rate * Prey - prey_death_rate * Prey * Predators
    dPredators/dt = predator_growth_rate * Prey * Predators - predator_death_rate * Predators
    ```

    where:

    *   `Prey` is the population of prey.
    *   `Predators` is the population of predators.
    *   `prey_growth_rate` is the intrinsic growth rate of the prey.
    *   `prey_death_rate` represents the predation rate.
    *   `predator_growth_rate` represents the efficiency of converting prey into predator offspring.
    *   `predator_death_rate` is the natural death rate of predators.

    
