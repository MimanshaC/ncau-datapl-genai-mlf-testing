from pathlib import Path
from typing import List
from typing import Type


def make_test_artifact(artifact_type: Type) -> Type:
    """Create test artifacts to be able to test components that use Input/Output artifacrs

    Args:
        artifact_type (Type): type of artifact to create

    Returns:
        Type: Artifact of specified type with adapted funcionality
    """

    class TestArtifact(artifact_type):
        metadata = {}

        def _get_path(self):
            return super()._get_path() or self.uri

        def log_confusion_matrix(self, categories: List[str], matrix: List[List[int]]):
            return f"{categories}\n{matrix}"

        def log_metric(self, metric: str, value: float):
            self.metadata.update({metric: value})

    return TestArtifact


def get_test_resource_folder() -> Path:
    """Get the resource folder within the tests for testing purposes

    Returns:
        Path: path of resource folder
    """
    return Path(__file__).parent.parent.joinpath("resources")
