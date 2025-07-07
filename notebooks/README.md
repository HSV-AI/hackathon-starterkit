
# Notebooks

This folder contains various Jupyter notebooks for different data analysis and machine learning tasks.

## Available Notebooks

1. **01_Jupyter_Intro.ipynb**: Introduction to Jupyter notebooks and basic usage.
2. **02_Data_Analysis.ipynb**: Comprehensive data analysis techniques and examples.
3. **03_Map_Display.ipynb**: Visualization of geographical data using maps.
4. **04_MUSK_Classification.ipynb**: Classification model for MUSK dataset.
5. **05_Questions_Answers.ipynb**: Interactive Q&A session with code examples.
6. **iris_classification.ipynb**: Classification of the Iris dataset.

## Running Notebooks in Google Colab with GPU

Google Colab is a free Jupyter notebook environment that runs in the cloud and supports GPU acceleration. Here's how to use it with a GPU:

1. **Open Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com/).

2. **Upload a Notebook**:
   - Click on "File" in the menu, then "Upload notebook".
   - Select a notebook from your local machine or from Google Drive.

3. **Enable GPU**:
   - Click on "Runtime" in the menu.
   - Select "Change runtime type".
   - In the dialog that appears, set "Hardware accelerator" to "GPU".
   - Click "Save".

4. **Run the Notebook**:
   - Once the GPU is enabled, you can run the notebook cells as usual.
   - Cells that use GPU acceleration will run much faster.

5. **Check GPU Availability**:
   - You can verify if the GPU is available by running the following code in a cell:
     ```python
     import tensorflow as tf
     print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
     ```

## Notes

- GPU availability in Colab is subject to Google's resource limits. If no GPU is available, you may need to wait and try again later.
- Some notebooks may not require GPU acceleration and can be run on CPU for faster iteration.
- For persistent storage, consider saving your work to Google Drive by mounting it in Colab.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Contributing

If you have improvements or new notebooks to add, please submit a pull request.
