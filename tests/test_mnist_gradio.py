import numpy as np
import pytest
import src.mnist_gradio as mnist_gradio

@pytest.fixture(autouse=True)
def patch_fetch(monkeypatch):
    """
    Provide a small fake dataset for faster tests by monkeypatching fetch_openml.
    """
    data = {
        'data': np.random.rand(100, 784),
        'target': np.array([i % 10 for i in range(100)])
    }
    def dummy_fetch(*args, **kwargs):
        return data
    monkeypatch.setattr(mnist_gradio, 'fetch_openml', dummy_fetch)
    yield


def test_get_classifier_attributes():
    model = mnist_gradio.get_classifier()
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')


def test_predict_outputs():
    # black image
    image = np.zeros((28, 28), dtype=float)
    label, prob_dict = mnist_gradio.predict(image)
    assert isinstance(label, int)
    assert 0 <= label <= 9
    assert isinstance(prob_dict, dict)
    assert len(prob_dict) == 10
    assert abs(sum(prob_dict.values()) - 1.0) < 1e-6
