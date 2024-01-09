import dataclasses
import operator
from typing import Callable


@dataclasses.dataclass
class AlertingThreshold:
    """
    Thresholds to be used to define minimum satisfactory model performance metrics.
    The comparison operator can be anything from the `operator` module, with the
    comparison being read as: `calculated_metric AlertingThreshold.comparison_operator
    AlertingThreshold.threshold`.

    Attributes:
        metric_name (str): The name of the model performance metric.
        threshold_value (float): The threshold value to compare against.
        comparison_operator (Callable[[float, float], bool]): A comparison operator
        function to evaluate the metric.
    """

    metric_name: str
    threshold_value: float
    comparison_operator: Callable[[float, float], bool]

    def compare(self, calculated_value: float) -> bool:
        """
        Compare the calculated metric value with the threshold value using the specified operator.

        Args:
            calculated_value (float): The value of the calculated model performance metric.

        Returns:
            bool: True if the calculated metric meets the specified criteria based on the
            operator and threshold, False otherwise.
        """
        return self.comparison_operator(calculated_value, self.threshold_value)


# TODO: Declare all metric thresholds to be used to generate performance monitoring alerts
PERFORMANCE_MONITORING_THRESHOLDS = [
    AlertingThreshold(
        metric_name="accuracy",
        threshold_value=0.7,
        comparison_operator=operator.lt,
    )
]
