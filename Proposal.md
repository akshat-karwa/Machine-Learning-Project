# Team 20: 7641 Project 

## Proposal



### Introduction / Background


The music industry has experienced significant evolution over the past two decades with the advent of streaming services. With streaming and social media changing consumption patterns, the evolution of culture can be directly pulsed through changes in music preferences [1]. Therefore, for streaming services like Spotify, understanding the dynamics of song popularity, genres, and moods has become essential to curating the customer experience. Our project aims to delve into this musical transformation by analyzing the "Top Hits Spotify from 2000-2019" dataset, which is composed of the 100 chart-topping songs from each year [2].



### Problem Definition


By leveraging machine learning, we aim to uncover patterns that contribute to song popularity, identify prevalent genres, and classify songs based on mood, offering insights into music trends over this period.

#### Supervised Learning:

In the context of supervised learning, our primary goal is to construct a regression model that predicts song popularity based on features extracted from the dataset. This model will help us identify key characteristics shared by top hit songs ultimately leading to the selection of the top 100 songs overall. 

#### Unsupervised Learning:

In the context of unsupervised learning, our focus extends to genre and mood classification. Through unsupervised techniques, we aim to uncover the most prevalent music genres and moods among the top hits, shedding light on the thematic and emotional trends driving the music industry during the time-period.



### Methods 


* Exploratory data analysis to identify patterns and trends in song attributes using feature visualization with help from Spotipy [3].
* Feature information determination by analyzing mutual information and entropy between features. Feature selection using correlation analysis and L1 regularization, rather than standard PCA [4]. 
* Song popularity prediction using expectation maximization, random forest, gradient boosting, and support vector regression models [5]. Regression evaluation using metrics like mean squared error and r-squared.
* Genre and mood classification using k-nearest neighbors, support vector machines, random forest, and CNN models.
* Stratified k-fold cross-validation and hyperparameter tuning using RandomizedSearchCV to enhance model robustness and performance.
* Using expectation maximization, we would iteratively optimize GMM parameters like audio characteristics and relevant music features. After training the Gaussian Mixture Model, we would evaluate the quality of our results using metrics such as the Silhouette score.



### Potential Results and Discussion


#### Metrics:

* Accuracy
* Recall
* Precision
* F1 Score

#### Hypothesis:

We believe that feature selection with correlation analysis and L1 Regularization may provide more accurate prediction capabilities over previous methods that implement standard PCA [4].

#### Discussion:

Our objectives are to create a regression model to effectively predict song popularity, along with unsupervised models to track genre and mood evolution over time. These objectives aim to contribute to a better understanding of music preferences and the changing musical landscape. Our reach goals include creating a recommendation system for personalized music experiences and a visualization tool to explore music trends over time. The reach goals we've set would enhance the user experience of music platforms and help users discover new music aligned with their preferences.



### Proposed Timeline


[Gannt Chart Timeline](https://docs.google.com/spreadsheets/d/1DY6M5F-5bqtjDR2IK07es8zp60YzUyz3/edit?usp=sharing&ouid=115370411170529593050&rtpof=true&sd=true)



### Contribution Table


| Tasks | Aashna | Akshat | Anika | Devang | Raj|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Proposal Writing | X | X | X | X | X |
| Proposal Slides | X | X | X | X | X |
| Proposal Timeline |  |  | X |  |  |
| Proposal Video | X |  |  |  |  |
| Proposal Dataset Selection |  | X |  | X |  |
| Proposal Github Page |  |  |  |  | X |


### Checkpoints


Our dataset is the "Top Hits Spotify from 2000-2019" dataset containing 2000 instances with 18 variables.  

  - Before Midterm → Nov 10:
    - Data processing and cleanup complete
    - Song genre and mood prediction architectures implemented and tuned
    - Relevant plots generated
    - Data prepared fo supervised model training
    - Report written
    
  - Before Final → Dec 5:
    - Song hit prediction architecture implemented and tuned
    - Relevant plots generated
    - Hyperparameters refined
    - Evaluation of genre and mood classifiers complete
    - Final presentation recorded and report written



### References

[1]	M. Mauch, R. M. MacCallum, M. Levy, and A. M. Leroi, "The evolution of popular music: USA 1960–2010," Royal Society Open Science, vol. 2, no. 5, p. 150081, 2015. doi: 10.1098/rsos.150081.

[2]	M. Kovehra, "Top Hits Spotify from 2000-2019," Kaggle, [Online]. Available: https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019/data.

[3]	S. Bruckert, Spotipy, Github.com, [Online]. Available: https://github.com/spotipy-dev/spotipy.

[4]	I. Dimolitsas, S. Kantarelis, and A. Fouka, "SpotHitPy: A Study For ML-Based Song Hit Prediction Using Spotify," arXiv preprint arXiv:2301.07978, 2023.

[5]	J. S. Gulmatico, et al., "SpotiPred: A Machine Learning Approach Prediction of Spotify Music Popularity by Audio Features," in 2022 Second International Conference on Power, Control and Computing Technologies (ICPC2T), Raipur, India, 2022, pp. 1-5, doi: 10.1109/ICPC2T53885.2022.9776765.

