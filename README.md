# Cat vs Dog Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs using Python. The model is trained on a dataset of labeled cat and dog images to distinguish between the two classes effectively.

## Project Overview
This project follows a tutorial from GeeksforGeeks:
[Cat-Dog Classification using CNN in Python](https://www.geeksforgeeks.org/cat-dog-classification-using-convolutional-neural-network-in-python/)

The implementation includes:
- Data preprocessing and augmentation
- Building a CNN model using TensorFlow/Keras
- Training the model on a dataset of cat and dog images
- Evaluating the model's performance

## Technologies Used
- Python
- TensorFlow/Keras
- NumPy
- OpenCV
- Matplotlib
- Pandas
- GeeksforGeeks tutorial as a reference

## How to Run the Project
1. Install the required dependencies using:
   ```bash
   pip install tensorflow numpy opencv-python matplotlib pandas
   ```
2. Open the Jupyter Notebook (`CatDog_CNN.ipynb`).
3. Run the notebook step by step to preprocess data, train the model, and evaluate its accuracy.

## Dataset
The dataset consists of labeled images of cats and dogs. You can use datasets like:
- [Kaggle Cat vs Dog Dataset](https://www.kaggle.com/datasets/subho117/cat-and-dog-classification-using-cnn)
- Custom datasets with images organized into `cat/` and `dog/` folders

## Model Architecture
The CNN model consists of:
- Convolutional Layers
- Max Pooling Layers
- Fully Connected (Dense) Layers
- Activation functions (ReLU, Softmax)

## Results
- The model is trained to classify images as either 'Cat' or 'Dog' with a high accuracy.
- Performance can be improved with hyperparameter tuning and more training data.

## References
- [GeeksforGeeks Tutorial](https://www.geeksforgeeks.org/cat-dog-classification-using-convolutional-neural-network-in-python/)

## License
This project is for educational purposes and follows the guidelines of the referenced tutorial.


