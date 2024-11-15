# Team 20: 7641 Project 

## Midterm



### Introduction / Background


The music industry has experienced significant evolution over the past two decades with the advent of streaming services. With streaming and social media changing consumption patterns, the evolution of culture can be directly pulsed through changes in music preferences [1]. Therefore, for streaming services like Spotify, understanding the dynamics of song popularity and genres has become essential to curating the customer experience. Our project aims to delve into this musical transformation by analyzing the "Top Hits Spotify from 2000-2019" dataset, which is composed of the 100 chart-topping songs from each year [2].


### Problem Definition


By leveraging machine learning, we aim to uncover patterns that contribute to song popularity and identify prevalent genres by building appropriate classifiers, offering insights into music trends over this period.

#### Supervised Learning:


In the context of supervised learning, our primary goal is to construct regression models that predict song popularity and genre based on features extracted from the dataset. These model will help us identify key characteristics shared by top hit songs that ultimately to the selection of the top 100 songs overall. 

#### Unsupervised Learning:


In the context of unsupervised learning, our focus extends to usage of feature selection and reduction techniques. Through unsupervised techniques, we aim to uncover relevant features within our dataset that contribute to hit and genre classification. We will primarily explore PCA and t-SNE, and compare them with supervised methods such as LDA feature selection.

### Quick Overview of Models and Progress

  - [X]  **Genre Classification**
    - [X]  Feature Reduction
      - [X]  PCA
      - [X]  t-SNE
      - [X]  LDA
    - [X]  Classifier Models
      - [X]  Random Forest 
      - [X]  Naive Bayes
      - [X]  SGD
      - [X]  Logistic Regression

- [ ] **Hit Classification**

## Data Collection


### Data Exploration and Cleaning


The data exploration and cleaning phase provided valuable insights into the distribution of genres within the "Top Hits Spotify from 2000-2019" dataset. A histogram of the number of songs for each genre was generated, visualizing the diversity and prevalence of different musical genres. 

<img width="1329" alt="image" src="https://github.gatech.edu/storage/user/58277/files/f2b45f35-2b01-442b-a5e0-4a36fa249193">

<img width="600" alt="image" src="https://github.gatech.edu/storage/user/58277/files/80af79b9-9b74-4fa7-925a-1494c8becf9a">


This graphical representation allowed us to observe the distribution of songs across various genres, forming the foundation for subsequent genre-focused analyses.We can drop the songs which solely belong to genres Folk/Acoustic, World/Traditional, blues, classical, country, easy listening and jazz because we have very little data about songs from these genres as you can see above (song count <= 20). Our model would consider these to be outliers or provide wrong predictions for these genres. Therefore, we will work with the remaining 7 genres.
To enhance the dataset's robustness, certain columns were identified for exclusion due to their perceived lack of relevance to audio characteristics and genre classification. 'Artist,' 'song,' 'duration_ms,' 'year,' 'popularity,' and 'genre' columns were dropped to streamline the dataset for more effective unsupervised learning models.

<img width="1022" alt="image" src="https://github.gatech.edu/storage/user/58277/files/45de91b6-39bf-4713-adda-44f61ca30612">


This refined visualization allowed us to concentrate on genres that had a more substantial representation in the dataset, contributing to a more accurate and meaningful analysis. We also normalized all the entries in the dataframe using MinMaxScaler which will transform all the entries to values between -1 and 1 based on the min and max values of the column.

Normalizing the values:
<img width="1235" alt="image" src="https://github.gatech.edu/storage/user/58277/files/90db78ce-0a0a-454b-b431-c7cc0433c1bf">


Finally, the exploration of feature relationships was complemented by a heatmap illustrating below.
This heatmap visually conveyed the strength and direction of correlations between different audio features, providing insights into potential multicollinearity issues. We observed that the correlation between the remaining features doesn't seem to be high and so we will move forward with all the features.

Correlation between features

<img width="677" alt="image" src="https://github.gatech.edu/storage/user/58277/files/8359b555-e050-4676-ba5a-3a3a17e09c01">



## Methods 


In this project, we employed a variety of methods, algorithms, and models to address the challenges associated with understanding the dynamics of song popularity, identifying prevalent genres, and classifying songs based on mood using the "Top Hits Spotify from 2000-2019" dataset. The primary focus was on leveraging machine learning techniques to uncover patterns in song data. Below, we outline the key methods employed in different stages of the project:

### **Dimensionality Reduction / Feature Selection**

**Feature Reduction Methods**

The initial dataset underwent a comprehensive preprocessing phase to enhance its suitability for analysis. One crucial step involved the application of Principal Component Analysis (PCA) to the data frame. PCA was employed to reduce dimensionality, eliminate highly correlated features, and preserve essential information within the dataset. We also implemented LDA and t-SNE, which we will show scatter plots for, but for the sake of brevity we'll only go over PCA in detail in this report. 

**Brief: Principal Component Analysis (PCA)**

**Objective**

The primary goal of PCA was to transform the original dataset using an orthogonal transformation. This transformation aimed to create a lower-dimensional linear space with the following objectives:
Maximizing Variance: The transformation sought to maximize the variance of the projected data, ensuring that crucial information was retained.
Minimizing Mean Squared Distance: The process minimized the mean squared distance between data points and their projections in the reduced-dimensional space.

**Procedure**

The application of PCA resulted in the generation of a set of values termed "principal components." These components exhibit low correlation and effectively capture the essential information present in the original dataset.

**Determining the Number of Principal Components**

To identify the appropriate number of principal components, we employed proportion of variance plots, which is a form of Scree plot. This approach involves looking at the cumulative proportion of variance explained by each PC and selecting a number of PCs that can collectively describe 80% of the total variance. This method allows us to capture a substantial amount of information while potentially reducing dimensionality effectively. It is more flexible and shows that we should select 3 principal components. Thus we will move forward with 3 PCs. The results obtained from PCA set the foundation for subsequent analyses, providing a reduced-dimensional representation of the data that retains critical information while mitigating issues associated with multicollinearity.

<img width="585" alt="image" src="https://github.gatech.edu/storage/user/58277/files/527170a1-bb55-4bcb-b94b-e8e65931bd89">

**Plots**


PCA:

![PPCA](../plots/ppca.png)

LDA:

![PLDA](../plots/plda.png)

t-SNE:

![t-SNE](../plots/tsne.png)


### **Learning Model Selection and Evaluation:**

We explored various machine learning models for genre classification, including Random Forest, Naive Bayes, Stochastic Gradient Descent (SGD), and Logistic Regression. The models were evaluated using cross-validation to assess their accuracy, precision, recall, and F1 scores. Random Forest and Logistic Regression emerged as the top-performing models based on the F1 score, demonstrating their effectiveness in classifying songs into different genres.

 ![image](https://github.gatech.edu/storage/user/58277/files/d43cb311-30c5-401a-8f74-4ef8db047137)

All of our models seemed to have very low accuracy with cross validation testing (Logistic Regression performing the best with 59% accuracy with the training data).

![cross val](../plots/accu.png)

### **K-Means Clustering:**

In addition to supervised learning, we explored unsupervised learning through KMeans clustering. The Elbow Method was employed to determine the optimal number of clusters, and the dataset was partitioned into seven clusters for further analysis. We did not find much success with using this approach to classify genres as it makes more sense for us to use the given genre labels rather than ambiguous cluster numbering. 
  
  <img width="481" alt="image" src="https://github.gatech.edu/storage/user/58277/files/a587508a-9130-4632-8b45-4b16de189192">

 <img width="1015" alt="image" src="https://github.gatech.edu/storage/user/58277/files/758f8045-86bb-4039-a076-07194b572293">


## **Result/Discussion:**

  <img width="500" alt="image" src="https://github.gatech.edu/storage/user/58277/files/1b743b05-3be2-4c14-aec0-bf4a19d09863">

We tested with our trained models with our test set and found that random forest performed the best, with an F1 score of 0.57. Please note, we found that the best performance happened when we did not use PCA, LDA, or t-SNE, which may be due to some error in our implementation we will have to debug. Regardless, we believe F1-Score is an appropriate measure here since our dataset is very imbalanced. Looking back at the distribution of genres across the dataset, hip-hop and pop dominate the others leading to genres like Latin and metal being much harder to classify. In addition, we feel as though our dataset is relatively small with <2000 entries, so in the next steps we plan to either move to a larger one or build it ourselves. Overall, we have created our two-step pipeline of feature selection and model training, which can classify genre and we hope to extend it to hit prediction. 


## **Next Steps:**

**Feature Analysis:**

We currently use PCA, LDA, and t-SNE. We plan to further refine our PCA and LDA implementations, and also replace t-SNE with UMAP for its simpler hyperparameter tuning and speed/stability for larger datasets. 

**Model Refinement:**
  
If certain components seem less informative, we could explore the possibility of refining the model by excluding them and then reassess the performance. We will also consider integrating additional datasets to enrich the analysis and potentially uncover more complex relationships within the music features.
  
**Advanced Visualization Techniques:**
  
Experiment with more advanced visualization techniques to enhance the communication of results and obtain a deeper understanding of the data's underlying structure.

The above next steps would allow us to potentially enhance the depth and accuracy of our analysis and provide more useful insights into the musical characteristics we are analyzing. 


### Updated Timeline


[Gannt Chart Timeline](https://docs.google.com/spreadsheets/d/1DY6M5F-5bqtjDR2IK07es8zp60YzUyz3/edit?usp=sharing&ouid=115370411170529593050&rtpof=true&sd=true)


### Contribution Table


| Tasks | Aashna | Akshat | Anika | Devang | Raj|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Midterm Writing | X | X | X | X | X |
| Methods |  | X |  |  | X |
| Data Collection |  | X |  | X | X |
| Midterm Github Page | X | | X | X | X |


### Checkpoints


Our dataset is the "Top Hits Spotify from 2000-2019" dataset containing 2000 instances with 18 variables.  

  - Things we accomplished:
    - Data processing and cleanup complete
    - Song genre prediction architectures implemented with initial tunes
    - Relevant plots generated
    - Data prepared for supervised model training
    - Midterm Report written
    

### References

[1]	M. Mauch, R. M. MacCallum, M. Levy, and A. M. Leroi, "The evolution of popular music: USA 1960–2010," Royal Society Open Science, vol. 2, no. 5, p. 150081, 2015. doi: 10.1098/rsos.150081.

[2]	M. Kovehra, "Top Hits Spotify from 2000-2019," Kaggle, [Online]. Available: https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019/data.

[3]	S. Bruckert, Spotipy, Github.com, [Online]. Available: https://github.com/spotipy-dev/spotipy.

[4]	I. Dimolitsas, S. Kantarelis, and A. Fouka, "SpotHitPy: A Study For ML-Based Song Hit Prediction Using Spotify," arXiv preprint arXiv:2301.07978, 2023.

[5]	J. S. Gulmatico, et al., "SpotiPred: A Machine Learning Approach Prediction of Spotify Music Popularity by Audio Features," in 2022 Second International Conference on Power, Control and Computing Technologies (ICPC2T), Raipur, India, 2022, pp. 1-5, doi: 10.1109/ICPC2T53885.2022.9776765.

