
# **Multiclass Classification with Iris Dataset**

## **Project Description**
This project demonstrates the application of various classification techniques to predict the species of flowers in the Iris dataset. The dataset contains three classes of flowers (Setosa, Versicolor, and Virginica) with four features: sepal length, sepal width, petal length, and petal width. We employ multiple classification algorithms to predict the flower species and compare their performances:

1. **Softmax Regression with Argmax**  
2. **Support Vector Machine (SVM)**
3. **One-vs-All (OvA) Strategy**
4. **One-vs-One (OvO) Strategy**

The goal is to assess which classification technique works best for multiclass problems.

## **Dataset**

The Iris dataset consists of 150 samples, each containing four features (sepal length, sepal width, petal length, and petal width) and a class label representing one of three Iris species:

| Feature Name      | Description                                                   |
|-------------------|---------------------------------------------------------------|
| `sepal_length`    | Length of the sepal in cm                                     |
| `sepal_width`     | Width of the sepal in cm                                      |
| `petal_length`    | Length of the petal in cm                                     |
| `petal_width`     | Width of the petal in cm                                      |
| `species`         | Class label (Setosa, Versicolor, Virginica)                    |

### **Dataset Sample**

| sepal_length | sepal_width | petal_length | petal_width | species   |
|--------------|-------------|--------------|-------------|-----------|
| 5.1          | 3.5         | 1.4          | 0.2         | Setosa    |
| 4.9          | 3.0         | 1.4          | 0.2         | Setosa    |
| 4.7          | 3.2         | 1.3          | 0.2         | Setosa    |
| 4.6          | 3.1         | 1.5          | 0.2         | Setosa    |
| 5.0          | 3.6         | 1.4          | 0.2         | Setosa    |

## **Classification Techniques Used**

1. **Softmax Regression with Argmax**  
   - Softmax regression, also known as multinomial logistic regression, is used for multiclass classification. We apply it to compute probabilities for each class and use the **argmax** function to select the class with the highest probability as the prediction.
  
2. **Support Vector Machine (SVM)**  
   - The SVM is trained to classify the data into multiple classes by constructing hyperplanes that maximize the margin between the classes.

3. **One-vs-All (OvA)**  
   - In this approach, a binary classifier is trained for each class where the samples of the current class are positive, and all others are negative. We use the classifier with the highest confidence score for the prediction.

4. **One-vs-One (OvO)**  
   - This strategy involves training a binary classifier for every pair of classes. Each classifier is trained to distinguish between two classes, and predictions are made by voting from all classifiers.

## **Model Evaluation**

After training all the models, we evaluate their performance using the following metrics:
- **Accuracy**: Percentage of correct predictions.
- **Precision, Recall, F1-Score**: For each class, to measure the performance of the model in detail.

### **Sample Model Results:**

| Model                     | Accuracy | Precision | Recall | F1-Score |
|---------------------------|----------|-----------|--------|----------|
| Softmax Regression         | 95.33%   | 0.95      | 0.95   | 0.95     |
| Support Vector Machine (SVM) | 97.33%   | 0.97      | 0.97   | 0.97     |
| One-vs-All (OvA)           | 96.67%   | 0.96      | 0.96   | 0.96     |
| One-vs-One (OvO)           | 96.00%   | 0.96      | 0.96   | 0.96     |

## **Technologies and Tools Used**
- **Programming Language:** Python  
- **Libraries:**  
  - `scikit-learn` for implementing classification algorithms.  
  - `pandas` for data handling and preprocessing.  
  - `matplotlib` and `seaborn` for visualization.  
- **Development Environment:** Jupyter Notebook  

## **How to Use the Project**

1. Clone this repository:  
   ```bash
   git clone https://github.com/SebasKHE/Multiclass_Classification-with-Iris-Dataset  .git
   ```
2. Install the required dependencies:  
   ```bash
   pip install scikit-learn pandas matplotlib seaborn
   ```
3. Open the Jupyter notebook:  
   ```bash
   jupyter notebook notebooks/iris_classification.ipynb
   ```
4. Run the cells to train and compare the models.

## **Conclusion**

This project provides a comparison of different multiclass classification techniques on the Iris dataset. By training and evaluating various models such as **Softmax Regression**, **SVM**, **One-vs-All**, and **One-vs-One**, we observe the differences in their performance and accuracy. The SVM classifier, in this case, showed the best accuracy, though other models also performed quite well.

---
