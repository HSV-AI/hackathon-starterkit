import numpy as np
import os

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression

# Global classifier, initialized lazily
clf = None

def get_classifier():
    """
    Load the MNIST dataset (subset) and train a logistic regression classifier if not already trained.
    """
    global clf
    if clf is None:
        # Fetch MNIST data from OpenML
        data = fetch_openml('mnist_784', version=1, as_frame=False)
        X = data['data'] / 255.0
        y = data['target'].astype(int)
        # Use a subset for faster training
        X_train, y_train = X[:5000], y[:5000]
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
    return clf

def predict(image):
    """
    Predict the digit drawn in the input image.

    Args:
        image (numpy.ndarray): 28x28 or RGB image array.

    Returns:
        int: Predicted digit label.
        dict: Probability distribution over digits 0-9.
    """
    classifier = get_classifier()
    arr = np.array(image)
    # Convert RGB to grayscale by averaging channels
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    # Ensure shape is 28x28
    arr = np.resize(arr, (28, 28))
    # Flatten and normalize
    flat = arr.reshape(1, -1) / 255.0
    proba = classifier.predict_proba(flat)[0]
    label = int(proba.argmax())
    prob_dict = {str(i): float(proba[i]) for i in range(len(proba))}
    return label, prob_dict

def launch():
    """
    Launch the Gradio interface for interactive digit classification.
    """
    # Import Gradio here to avoid requiring it for tests
    import gradio as gr

    # Ensure classifier is trained before launching
    get_classifier()
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Sketchpad(shape=(28, 28), invert_colors=True, label="Draw a digit"),
        outputs=[
            gr.Textbox(label="Predicted Label"),
            gr.Label(num_top_classes=3, label="Probabilities")
        ],
        title="MNIST Digit Classifier",
        description="Draw a digit (0-9) and the model will predict it.",
        allow_flagging="never"
    )
    port = int(os.environ.get('PORT', 50742))
    interface.launch(server_name="0.0.0.0", server_port=port, share=False)

if __name__ == "__main__":
    launch()
