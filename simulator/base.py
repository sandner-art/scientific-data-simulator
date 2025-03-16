# simulator/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

class ExperimentLogic(ABC):
    """
    Abstract base class for defining experiment logic.
    """

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initializes the simulation state.

        Args:
            config: A dictionary of configuration parameters.

        Returns:
            A dictionary representing the initial state of the simulation.
        """
        pass

    @abstractmethod
    def run_step(self, state: Dict[str, Any], step: int) -> Dict[str, Any]:
        """
        Runs a single step of the simulation.

        Args:
            state: A dictionary representing the current state of the simulation.
            step: The current simulation step number.

        Returns:
            A dictionary representing the updated state of the simulation.
        """
        pass

    @abstractmethod
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the simulation results.

        Returns:
            A dictionary where keys are data names and values are dictionaries
            containing the data and a DataDescriptor.
        """
        pass

    def visualize(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Optionally provides custom visualization logic.
        """
        pass  # Default implementation does nothing


class LLMClient(ABC):
    """
    Abstract base class for interacting with LLM APIs.
    """

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generates text from the LLM based on a prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters specific to the LLM provider
                      (e.g., temperature, max_tokens, model_name).

        Returns:
            The generated text.
        """
        pass

    @abstractmethod
    def generate_structured_output(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
      """
      Generates structured output (e.g., JSON) from the LLM.

      Args:
          prompt: The input prompt
          schema: a dictionary that describes expected structure.
          **kwargs: additional parameters specific to the LLM provider.

      Returns:
          a dictionary that contains generated data.
      """
      pass

    @abstractmethod
    def get_usage_info(self) -> Dict[str, Any]:
        """
        Gets information about API usage (e.g., number of tokens used).

        Returns:
             A dictionary of usage information.
        """
        pass