# Team 20: 7641 Project 

## Final Report



### Introduction / Background


The music industry has experienced significant evolution over the past two decades with the advent of streaming services. With streaming and social media changing consumption patterns, the evolution of culture can be directly pulsed through changes in music preferences [1]. Therefore, for streaming services like Spotify, understanding the dynamics of song popularity and genres has become essential to curating the customer experience. Our project aims to delve into this musical transformation by analyzing the "The Spotify Hit Predictor Dataset (1960-2019)" dataset, which is composed of over 38000 songs throughout the decades [6].


### Problem Definition


By leveraging machine learning, we aim to uncover patterns that contribute to song popularity and identify prevalent genres by building appropriate classifiers, offering insights into music trends over this period.

#### Supervised Learning:


In the context of supervised learning, our primary goal is to construct regression models that predict song popularity and genre based on features extracted from the dataset. These model will help us identify key characteristics shared by top hit songs that ultimately to the selection of the top 100 songs overall. 

#### Unsupervised Learning:


In the context of unsupervised learning, we emphasize the use of feature selection and reduction techniques to identify key features in our dataset that are significant for hit and genre classification. Our approach now includes Principal Component Analysis (PCA) and Uniform Manifold Approximation and Projection (UMAP) as our primary tools for dimensionality reduction, aiding in reducing overfitting. Additionally, we employ Linear Discriminant Analysis (LDA) for feature selection. Notably, we have decided to exclude t-Distributed Stochastic Neighbor Embedding (t-SNE) and Singular Value Decomposition (SVD) from our methodology, focusing instead on the more effective combination of PCA and UMAP for our unsupervised learning strategy.


### Quick Overview of Models and Progress

  - [X]  **Genre Classification**
    - [X]  Feature Reduction
      - [X]  PCA
      - [X]  UMAP
      - [X]  LDA
    - [X]  Classifier Models
      - [X]  Random Forest 
      - [X]  Naive Bayes
      - [X]  SGD
      - [X]  Logistic Regression

- [X] **Hit Classification**
    - [X]  Feature Reduction
      - [X]  PCA
      - [X]  UMAP
   - [X]  Classifier Models
      - [X]  Random Forest 
      - [X]  Naive Bayes
      - [X]  SGD
      - [X]  Logistic Regression
      - [X]  LDA

## Data Collection


### Data Exploration and Cleaning


The data exploration and cleaning phase provided valuable insights into the distribution of genres with our updated dataset the "The Spotify Hit Predictor Dataset (1960-2019)". We preprocessed our dataset and dropped duplicate values. A histogram of the number of songs for each genre was generated, visualizing the diversity and prevalence of different musical genres. 


<img width="600" alt="image" src="https://github.gatech.edu/storage/user/54546/files/0485ad6f-6153-410a-98a0-3dc7287f16cb">


To enhance the dataset's robustness, certain columns were identified for exclusion due to their perceived lack of relevance to audio characteristics and genre classification. 'Artist,' 'song,' 'duration_s,' and 'year' columns were dropped to streamline the dataset for more effective unsupervised learning models.

<img width="1022" alt="image" src="https://github.gatech.edu/storage/user/54546/files/3e255b00-6adb-41cc-b2f4-5dcdc3bf55c0">


This refined visualization allowed us to concentrate on genres that had a more substantial representation in the dataset, contributing to a more accurate and meaningful analysis. We also normalized all the entries in the dataframe using StandardScaler, which standardizes the data by transforming each feature to have a mean of 0 and a standard deviation of 1, ensuring consistent scaling across all columns.

Normalizing the values:
<img width="1235" alt="image" src="https://github.gatech.edu/storage/user/54546/files/1608cfc4-5a63-44bf-84af-9712c514bfa1">


Finally, the exploration of feature relationships was complemented by a heatmap illustrating below.
This heatmap visually conveyed the strength and direction of correlations between different audio features, providing insights into potential multicollinearity issues. We observed that the correlation between the remaining features doesn't seem to be high and so we will move forward with all the features.

Correlation between features

<img width="677" alt="image" src="https://github.gatech.edu/storage/user/54546/files/a8166fe2-1108-4271-af59-21a9de90476b">



## Methods 


In this project, we employed a variety of methods, algorithms, and models to address the challenges associated with understanding the dynamics of song popularity, identifying prevalent genres, and classifying songs based on mood using the "The Spotify Hit Predictor Dataset (1960-2019)". The primary focus was on leveraging machine learning techniques to uncover patterns in song data. Our project focuses on 2 distinct sections: Genre Classification and Hit Classification. Below, we outline the key methods employed in different stages of the project:

### **Dimensionality Reduction / Feature Selection**

**Feature Reduction Methods**

The initial dataset underwent a comprehensive preprocessing phase to enhance its suitability for analysis. One crucial step involved the application of Principal Component Analysis (PCA) to the data frame. PCA was employed to reduce dimensionality, eliminate highly correlated features, and preserve essential information within the dataset. We implemented LDA and swapped out T-SNE for UMAP.

**Brief: Principal Component Analysis (PCA)**

**Objective**

The primary goal of PCA was to transform the original dataset using an orthogonal transformation. This transformation aimed to create a lower-dimensional linear space with the following objectives:
Maximizing Variance: The transformation sought to maximize the variance of the projected data, ensuring that crucial information was retained.
Minimizing Mean Squared Distance: The process minimized the mean squared distance between data points and their projections in the reduced-dimensional space.

**Procedure**

The application of PCA resulted in the generation of a set of values termed "principal components." These components exhibit low correlation and effectively capture the essential information present in the original dataset.

**Determining the Number of Principal Components**

To identify the appropriate number of principal components, we employed proportion of variance plots, which is a form of Scree plot. This approach involves looking at the cumulative proportion of variance explained by each PC and selecting a number of PCs that can collectively describe 80% of the total variance. This method allows us to capture a substantial amount of information while potentially reducing dimensionality effectively. It is more flexible and shows that we should select 8 principal components. Thus we will move forward with 8 PCs. The results obtained from PCA set the foundation for subsequent analyses, providing a reduced-dimensional representation of the data that retains critical information while mitigating issues associated with multicollinearity.


<img width="585" alt="image" src="https://github.gatech.edu/storage/user/54546/files/625b0256-d754-4f6e-b53b-7762c5bac716">

**Plots for Genre Classification**


PCA:

<img width="585" alt="image" src="https://github.gatech.edu/storage/user/54546/files/4653c471-e9e1-4746-9630-68637c6643e9">

LDA:


<img width="585" alt="image" src="https://github.gatech.edu/storage/user/54546/files/a223aed8-888d-4259-82ca-5df371e562e2">

UMAP:

<img width="585" alt="image" src="https://github.gatech.edu/storage/user/54546/files/43d88faa-c548-457d-99cc-d88083cdb171">


**Plots for Hit Classification**

PCA: 

<img width="583" alt="image" src="https://github.gatech.edu/storage/user/53949/files/d321170d-ae39-4a38-9a74-bb9f3b157598">

<img width="527" alt="image" src="https://github.gatech.edu/storage/user/53949/files/1ad73c83-ffef-4b35-83dd-4bba22bba756">

UMAP: 

<img width="538" alt="image" src="https://github.gatech.edu/storage/user/53949/files/8d7444ab-ba2a-4f78-8471-e41d1d69b490">

### **Learning Model Selection and Evaluation:**

**Genre Classification**

We explored various machine learning models for genre classification, including Random Forest, Naive Bayes, Stochastic Gradient Descent (SGD), and Logistic Regression. The models were evaluated using cross-validation to assess their accuracy, precision, recall, and F1 scores. Random Forest and Logistic Regression emerged as the top-performing models based on the F1 score, demonstrating their effectiveness in classifying songs into different genres.

<img width="585" alt="image" src="https://github.gatech.edu/storage/user/54546/files/59a58629-81c6-4aca-a8ef-0214fc7aa4b9">


<img width="585" alt="image" src="https://github.gatech.edu/storage/user/54546/files/df4cf7b8-dc04-4b43-8bc4-1c355052a98d">


Compared to our previous low values, all of our models now have good accuracy. The accuracy could have been higher but we have a lack of well distributed data. The current data does not have an equal representation of the different genre classes. It appears that Random Forest Classifier has the highest accuracy (87.3%). We will now calculate the recall and precision values. Then, we will calculate the F1 score based on that. 

Based on the images above, Random Forest Classifier has the best F1 score (87.7) followed by SGD Classifier, then Logistic Regression and Naive Bayes.

### **Visualizing with K-Means Clustering:**

In addition to supervised learning, we explored unsupervised learning through KMeans clustering. The Elbow Method was employed to determine the optimal number of clusters, and the dataset was partitioned into six clusters. We did not find much success with using this approach to classify genres as it makes more sense for us to use the given genre labels rather than ambiguous cluster labeling. 
  
  
  <img width="1015" alt="image" src="https://github.gatech.edu/storage/user/54546/files/f21dabfc-0806-4380-ae9d-7721bd60c9e3">
  

 <img width="481" alt="image" src="https://github.gatech.edu/storage/user/54546/files/5a43a57f-1af2-4d66-b8d0-625f0d5dae8f">
 
 
**Hit Classification**

In our implementation of hit classification, we considered various machine learning models, including Random Forest, Naive Bayes, Stochastic Gradient Descent (SGD), and Logistic Regression. These models underwent rigorous cross-validation to evaluate their accuracy, precision, recall, and F1 scores, with a focus on the F1 score for a comprehensive assessment.

Visual representations of the evaluation results and metrics are provided for clarity.

Compared to initial assessments, where values were lower, our models now exhibit commendable accuracy despite challenges with data distribution. Notably, the UMAP pipeline emerged as the top performer. The results below highlight the relative performance of each model, with Random Forest leading in F1 scores, closely followed by SGD, Logistic Regression, and Naive Bayes. This concise evaluation provides insights into the nuanced trade-offs between precision and recall in the context of hit classification.

UMAP Pipeline Results:

<img width="495" alt="image" src="https://github.gatech.edu/storage/user/53949/files/bbc61014-91f3-456e-a5e3-62b7bfb30d5f">

PCA-LDA: 

<img width="480" alt="image" src="https://github.gatech.edu/storage/user/53949/files/88a2fdb8-e214-43aa-9bfe-7b7e2b1404ea">

UMAP-LDA: 

<img width="485" alt="image" src="https://github.gatech.edu/storage/user/53949/files/630993da-16ac-4154-86bb-865e89448a72">


## **Result/Discussion:**

**Genre Classification**


  <img width="500" alt="image" src="https://github.gatech.edu/storage/user/54546/files/cc03debb-2b4e-445e-80d9-6cc584d6fb6c">

We tested with our trained models with our test set and found that random forest performed the best, with an F1 score of 0.88. Please note, we found that the best performance happened when we did not use PCA, LDA, or UMAP, which may be due to some error in our implementation we will have to debug. Regardless, we believe F1-Score is an appropriate measure here since our dataset is very imbalanced. Looking back at the distribution of genres across the dataset, pop dominates the others leading to genres like latin being much harder to classify. Overall, we have created our two-step pipeline of feature selection and model training, which can classify genre and we extnd it to hit prediction below.


**Hit Classification**

<img width="553" alt="image" src="https://github.gatech.edu/storage/user/53949/files/8505295c-03df-4fab-b332-31cd21a2d59b">

Hit prediction performs much better due to it being a binary classification problem, so that is much simpler. The results from the PCA pipleline are attached above with a random forest F1 score of 0.96.

## **Conclusion**


In conclusion, our project aimed to unravel the dynamics of song popularity and genre classification within the context of the evolving music industry, particularly focusing on the "The Spotify Hit Predictor Dataset (1960-2019)" dataset. Through a comprehensive exploration of machine learning models, dimensionality reduction techniques, and thorough data analysis, we focused on genre classification and hit classification. For genre classification, Random Forest and Logistic Regression emerged as top performers, showcasing their efficacy in discerning musical genres based on various features. However, the performance of our models was influenced by the imbalanced distribution of genres in the dataset. On the other hand, our hit classification endeavors led us to employ diverse models, with our UMAP pipeline standing out as the top performer in capturing relevant patterns. Despite encountering challenges and potential errors in feature reduction techniques, we have established robust pipelines for genre and hit classification. Moving forward, our emphasis will be on refining our methodologies, addressing data distribution issues, and exploring larger datasets to enhance the accuracy and generalizability of our models. Overall, our project represents a significant step towards understanding the intricate relationships between music features, genres, and popularity trends.


### Updated Timeline


[Gannt Chart Timeline](https://docs.google.com/spreadsheets/d/1DY6M5F-5bqtjDR2IK07es8zp60YzUyz3/edit?usp=sharing&ouid=115370411170529593050&rtpof=true&sd=true)


### Contribution Table


| Tasks | Aashna | Akshat | Anika | Devang | Raj|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Final Report | X |  | X | | |
| Update Methods |  | X |  | X |  |
| Hit Classification |  | X |  | X | X |
| Midterm Video | X | | X | | |


### Checkpoints


Our dataset is the "The Spotify Hit Predictor Dataset (1960-2019)" dataset containing >38000 instances with 20 variables.  

  - Things we accomplished:
    - Data processing and cleanup complete
    - Song genre prediction architectures implemented with initial tunes
    - Relevant plots generated
    - Data prepared for supervised model training
    - Report Written
    

### References

[1]	M. Mauch, R. M. MacCallum, M. Levy, and A. M. Leroi, "The evolution of popular music: USA 1960–2010," Royal Society Open Science, vol. 2, no. 5, p. 150081, 2015. doi: 10.1098/rsos.150081.

[2]	M. Kovehra, "Top Hits Spotify from 2000-2019," Kaggle, [Online]. Available: https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019/data.

[3]	S. Bruckert, Spotipy, Github.com, [Online]. Available: https://github.com/spotipy-dev/spotipy.

[4]	I. Dimolitsas, S. Kantarelis, and A. Fouka, "SpotHitPy: A Study For ML-Based Song Hit Prediction Using Spotify," arXiv preprint arXiv:2301.07978, 2023.

[5]	J. S. Gulmatico, et al., "SpotiPred: A Machine Learning Approach Prediction of Spotify Music Popularity by Audio Features," in 2022 Second International Conference on Power, Control and Computing Technologies (ICPC2T), Raipur, India, 2022, pp. 1-5, doi: 10.1109/ICPC2T53885.2022.9776765.

[6] A. Farooq, "The Spotify Hit Predictor Dataset (1960-2019)," Kaggle, [Online]. Available: https://www.kaggle.com/datasets/akiboy96/spotify-genre-joined.

