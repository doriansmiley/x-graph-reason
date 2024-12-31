import pytest
import numpy as np
import json
from unittest.mock import MagicMock, patch, call
from KnowledgeGraph import ExternalContinualLearner  # Assuming the class is in a file named ExternalContinualLearner.py

@pytest.fixture
def mocked_openai_client():
    """
    Fixture to mock the OpenAI client instance.
    """
    client = MagicMock()
    # Mock the embeddings.create method's response
    client.embeddings.create.return_value = MagicMock(
        data=[
            MagicMock(embedding=[0.1, 0.2, 0.3])
        ]
    )
    # mock completions
    client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "class": "Claim",
                        "relationships": {
                            "denialReason": "Preauthorization required",
                            "adjustment": "Charge exceeds allowed amount"
                        },
                        "resolutions": {
                            "denialReason": "Submit preauthorization forms",
                            "adjustment": "Verify charge limits"
                        }
                    })
                )
            )
        ]
    )
    
    return client

@pytest.fixture
def ecl():
    client = MagicMock()
    # Mock the embeddings.create method's response
    client.embeddings.create.return_value = MagicMock(
        data=[
            MagicMock(embedding=[0.1, 0.2, 0.3])
        ]
    )
    # mock completions
    client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "class": "Claim",
                        "relationships": {
                            "denialReason": "Preauthorization required",
                            "adjustment": "Charge exceeds allowed amount"
                        },
                        "resolutions": {
                            "denialReason": "Submit preauthorization forms",
                            "adjustment": "Verify charge limits"
                        }
                    })
                )
            )
        ]
    )
    """
    Fixture to initialize the ExternalContinualLearner instance.
    """
    return ExternalContinualLearner(client)


def test_update_class(ecl):
    """
    Test the `update_class` method to ensure class embeddings are updated correctly.
    """
    tag_embeddings_class1 = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [1.2, 2.2, 3.2]])
    tag_embeddings_class2 = np.array([[4.0, 5.0, 6.0], [4.5, 5.5, 6.5]])

    ecl.update_class("class1", tag_embeddings_class1)
    ecl.update_class("class2", tag_embeddings_class2)

    expected_mean_class1 = np.mean(tag_embeddings_class1, axis=0)
    expected_mean_class2 = np.mean(tag_embeddings_class2, axis=0)

    assert np.allclose(ecl.class_means["class1"], expected_mean_class1), "Class1 mean embedding is incorrect."
    assert np.allclose(ecl.class_means["class2"], expected_mean_class2), "Class2 mean embedding is incorrect."
    assert ecl.shared_covariance is not None, "Shared covariance matrix should not be None after updates."
    assert ecl.num_classes == 2, "Number of classes should be 2."


def test_get_top_k_classes(ecl):
    """
    Test the `get_top_k_classes` method to ensure it correctly retrieves the top-k closest classes.
    """
    tag_embeddings_class1 = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [1.2, 2.2, 3.2]])
    tag_embeddings_class2 = np.array([[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]])
    tag_embeddings_class3 = np.array([[20.0, 20.0, 20.0]])

    ecl.update_class("class1", tag_embeddings_class1)
    ecl.update_class("class2", tag_embeddings_class2)
    ecl.update_class("class3", tag_embeddings_class3)

    input_embedding = np.array([1.4, 2.4, 3.4])

    top_k_classes = ecl.get_top_k_classes(input_embedding, k=2)
    assert top_k_classes == ["class1", "class2"], "Top-k classes are incorrect."


def test_empty_ecl_behavior(ecl):
    """
    Test behavior when trying to query an empty ECL.
    """
    input_embedding = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="No classes have been added to the learner."):
        ecl.get_top_k_classes(input_embedding, k=3)
