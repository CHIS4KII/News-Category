# News Category Classification with a Convolutional Neural Network (CNN)

This repository contains a project that builds, trains, and evaluates a deep learning model to classify news articles into one of 15 categories based on their headlines and short descriptions. The model uses a Convolutional Neural Network (CNN) architecture, a powerful technique for text classification tasks.

![Word Cloud of News Categories](images/wordcloud.png)

## 📋 Project Overview

The goal of this project is to automate the process of news categorization. Given a news headline and a brief description, the model predicts its most likely category (e.g., POLITICS, ENTERTAINMENT, SPORTS). This is a multi-class text classification problem addressed using Natural Language Processing (NLP) and TensorFlow/Keras.

The notebook demonstrates a complete machine learning workflow:
1.  **Data Loading & Cleaning**: Loading the dataset and performing comprehensive text preprocessing.
2.  **Exploratory Data Analysis (EDA)**: Visualizing the distribution of news categories.
3.  **Feature Engineering**: Combining text fields and preparing them for the model.
4.  **Model Building**: Designing and implementing a CNN model tailored for text data.
5.  **Training & Evaluation**: Training the model with early stopping and evaluating its performance with a detailed classification report.
6.  **Saving Artifacts**: Saving the trained model, tokenizer, and label encoder for future use.

---

## 📊 Dataset

The project utilizes the **News Category Dataset Version 3** from Kaggle, which contains news headlines from HuffPost between 2012 and 2022.

- **Source**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- **Original Size**: 209,527 news articles across 42 categories.
- **Preprocessing**: To manage class imbalance and focus the model, the dataset was filtered to include only the **top 15 most frequent categories**.

![Pie Chart of Category Distribution](images/pie_chart.png)

---

## ⚙️ Methodology

The project follows a structured approach to solve the classification problem:

### 1. Text Preprocessing
The `headline` and `short_description` columns were combined into a single text feature. A cleaning pipeline was then applied to this text, which included:
- Converting text to lowercase.
- Removing punctuation and special characters.
- Tokenizing text into individual words.
- Removing common English stopwords using NLTK.
- **Lemmatization**: Reducing words to their base or dictionary form (e.g., "running" -> "run") using `WordNetLemmatizer`.

### 2. Model Preparation
- **Label Encoding**: The 15 text-based category labels were converted into numerical format using `sklearn.preprocessing.LabelEncoder`.
- **Tokenization**: The cleaned text was tokenized using `tensorflow.keras.preprocessing.text.Tokenizer`, converting words into integer sequences. The vocabulary was limited to the top 10,000 words.
- **Padding**: All sequences were padded to a uniform length of 100 to ensure consistent input size for the model.
- **Data Splitting**: The data was split into training (80%) and testing (20%) sets, stratified to maintain the original class distribution in both sets.

### 3. CNN Model Architecture
A Sequential Keras model was built with the following layers, which is a robust architecture for text classification:
1.  **Embedding Layer**: Maps integer sequences to dense vector representations of 128 dimensions.
2.  **Dropout (0.2)**: Regularization layer to prevent overfitting.
3.  **Conv1D Layer**: A 1D convolutional layer with 128 filters and a kernel size of 5. It acts as a feature detector, identifying patterns (similar to n-grams) in the text.
4.  **GlobalMaxPooling1D**: Pools the feature maps to reduce dimensionality and capture the most important features.
5.  **Dropout (0.5)**: A second, more aggressive dropout layer for further regularization.
6.  **Dense Layer (Output)**: A fully connected layer with a `softmax` activation function to output probabilities for each of the 15 classes.

The model was compiled with the `adam` optimizer and `categorical_crossentropy` loss function. **Early Stopping** was used to monitor validation loss and prevent overfitting by stopping the training when performance on the validation set ceased to improve.

---

## 📈 Results

The model was trained for up to 20 epochs with a batch size of 64. It achieved an **overall accuracy of approximately 74%** on the unseen test data. The detailed performance for each class is shown below:

```
               precision    recall  f1-score   support

  BLACK VOICES       0.62      0.38      0.47       917
      BUSINESS       0.67      0.55      0.60      1198
        COMEDY       0.66      0.46      0.54      1080
 ENTERTAINMENT       0.69      0.80      0.74      3473
  FOOD & DRINK       0.77      0.78      0.77      1268
HEALTHY LIVING       0.47      0.36      0.41      1339
 HOME & LIVING       0.74      0.73      0.74       864
     PARENTING       0.59      0.73      0.65      1758
       PARENTS       0.50      0.23      0.32       791
      POLITICS       0.85      0.90      0.87      7121
  QUEER VOICES       0.84      0.72      0.77      1269
        SPORTS       0.79      0.73      0.75      1015
STYLE & BEAUTY       0.85      0.81      0.83      1963
        TRAVEL       0.81      0.79      0.80      1980
      WELLNESS       0.67      0.79      0.73      3589

      accuracy                           0.74     29625
     macro avg       0.70      0.65      0.67     29625
  weighted avg       0.73      0.74      0.73     29625
```
- **High-Performing Categories**: The model performs exceptionally well on distinct categories like `POLITICS`, `STYLE & BEAUTY`, and `TRAVEL`.
- **Areas for Improvement**: Categories with lower scores, such as `PARENTS` and `HEALTHY LIVING`, may overlap significantly with others (like `PARENTING` and `WELLNESS`), making them harder to distinguish.

---

## 🛠️ Technologies Used

- **Python 3**: Core programming language.
- **Jupyter Notebook**: For interactive development and analysis.
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn, WordCloud
- **Machine Learning**: Scikit-learn (for preprocessing and evaluation)
- **Deep Learning**: TensorFlow, Keras (for model building and training)
- **NLP**: NLTK (for stopwords and lemmatization)

---

## 🚀 How to Run

To replicate this project on your local machine, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    Make sure you have a `requirements.txt` file (as described above) in your repository.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data**
    Run the following in a Python interpreter to download the necessary NLTK packages:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

5.  **Get the Dataset**
    - Download the `News_Category_Dataset_v3.json` file from the [Kaggle link](https://www.kaggle.com/datasets/rmisra/news-category-dataset).
    - Place the JSON file in the root directory of the project, or update the file path in the Jupyter Notebook.

6.  **Run the Notebook**
    - Launch Jupyter Notebook:
      ```bash
      jupyter notebook
      ```
    - Open `News_Category.ipynb` and run the cells.

---

## 💾 Saved Artifacts

The following trained components have been saved and can be found in the repository. This allows for easy inference on new data without retraining.

- `news_category_cnn_model.h5`: The trained Keras model weights and architecture.
- `tokenizer.pickle`: The Keras `Tokenizer` object fitted on the training data.
- `label_encoder.pickle`: The Scikit-learn `LabelEncoder` object for converting between text labels and integer classes.
