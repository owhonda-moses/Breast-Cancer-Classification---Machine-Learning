# BREAST CANCER 
## INTRODUCTION

> In this project, we analyzed the Wisconsin Breast Cancer dataset to classify breast masses as benign or malignant based on multiple features extracted from fine needle aspirate (FNA) images. The dataset required extensive preprocessing due to outliers, invalid data, null values, and feature selection to ensure the robustness of subsequent analyses. The original dataset contained 571 samples of 30 numerical measurements, drilled down to 518 samples after preprocessing.
We employed various techniques, including linear regression imputations to handle missing values, dimensionality reduction, unsupervised, and supervised machine learning algorithms to extract insights and build predictive models.
The steps taken in data preprocessing and building and analyzing the models are documented.


### PREPROCESSING

Visual and programmatic assessments were done and copies of the dataframes were made. Several content related issues were identified and the data was cleaned sequentially. Mean impuation and linear regression techniques were used to handle null values in the data. After cleaning the data, further preprocessing for machine learning tasks was performed like splitting the data into training and test sets, and applying transformations to scale our features.

### UNSUPERVISED MACHINE LEARNING
The K-Means Clustering algorithm was employed to cluster the data into benign and malignant clusters. Additionally, dimensionality reduction techniques such as Principal Component Analysis (PCA) and more advanced methods like Kernel PCA and Supervised UMAP (Uniform Manifold Approximation and Projection) were employed to assess potential improvements in clustering.

### SUPERVISED MACHINE LEARNING
Logistic regression and Random Forest classifier models were used to classify breast tumors as either benign or malignant. Evaluation metrics such as accuracy, precision, recall, F1-score, and validation curves were used to assess each model's performance with reduced features and full features



### KEY FINDINGS

The samples contained nearly 30% more benign classes than malignant ones.
Features related to concave points and worst measurements emerged as highly influential for diagnosis classification. We found that dimensionality reduction techniques aided in enhancing clustering and classification accuracy; supervised UMAP demonstrated superior cluster definition than PCA, kPCA, t-SNE, and manually extracted features.

Logistic Regression exhibited consistent performance across training and test sets, outperforming Random Forest Classifier in tumor classification, offering clearer boundaries between classes.
The Random Forest Classifier showed signs of overfitting initially but improved with hyperparameter tuning.


#### PACKAGES USED:

- JuPyter notebook
- pandas
- sklearn
- numpy
- matplotlib
- seaborn
- umap
- missingno


_Some notable references for this project include -_
- https://stackoverflow.com/
- https://github.com/
- https://www.geeksforgeeks.org/
- https://umap-learn.readthedocs.io/en/latest/
- https://wiki.python.org/
- https://www.analyticsvidhya.com/
- https://towardsdatascience.com/
- https://kaggle.com/


Note: _The original dataset on Kaggle contains clean data but the version used in this project contained low qaulity data._