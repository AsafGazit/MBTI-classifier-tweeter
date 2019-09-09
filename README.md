# MBTI-classifier
Performance comparison of a MultiLayer Perceptron (feed-forward neural net) for personality type indicator classification, based on tweeter feed data, when applying oversampling (SMOTE) and undersampling (NearMiss) techniques.

### The comparison includes:
1. The effect of feature engineering, namely counting occurrences, sentiment analysis and part of speech tags (
[as suggested in the this Medium article](https://medium.com/@bian0628/data-science-final-project-myers-briggs-prediction-ecfa203cef8)).

2. Performance of a personality classifier vs. personality dimension classifier (full type indicator compared to 4 classifiers that specializes in each of the MBTI personality dimensions). 

3. The effect of oversampling and undersampling using SMOTE (Synthetic Minority Over-sampling Technique) and Near Miss.
The class inequality of this dataset makes this aspect to be of special interest. As such, models are trained on dataset after applying SMOTE (oversample) and Near Miss (undersampling) to compare those effects.

### The dataset

Myers-Briggs Personality Type Dataset contains XXX records, each containing a personality indicator for a subject and the subjects’ 50 last tweets.

### Classes
The MBTI personality type divides everyone into 16 distinct personality types across 4 axis:
1. Introversion (I) – Extroversion (E)
2. Intuition (N) – Sensing (S)
3. Thinking (T) – Feeling (F)
4. Judging (J) – Perceiving (P)

Classes distribution:

<img src="https://github.com/AsafGazit/MBTI-classifier/blob/master/img/MBTI_classdist.png" alt="MBTI classes distribution" width="90%" height="90%">

And along the axes:

<img src="https://github.com/AsafGazit/MBTI-classifier/blob/master/img/PJ_classdist.png" alt="MBTI classes distribution along the axes" width="70%" height="70%">

### Preprocessing and feature engineering

Two sets of features are composed of:
1. Clean tweet text: no URLs, user tags, stopwords and punctuation.
2. Designed features, which includes:
- counting occurrences: average word count per tweet, URLs, user tags, ellipsis and punctuation.
- Sentiment analysis: The classifiers are detailed here (LINK).
- Part of speech tags: average use of each part of speech (using NLTK classifier).

Preprocessing code can be found [here.](https://github.com/AsafGazit/MBTI-classifier/blob/master/code/preprocess_clean.py)

### Vectorization and standardisation

1. TF-IDF vectorizer is applied on the clean tweet text. The vocabulary is set for 3,000 words.
2. Standard scaler is applied over the designed features.

### MLP model

The model consists of two dense layers containing 64 units with a ReLU activation function and two dropout layers after each dense layer. The optimiser applied is RMSprop. 
The number of epochs chosen is 1000.

This model is inspired by [this this Medium project.](https://medium.com/@bian0628/data-science-final-project-myers-briggs-prediction-ecfa203cef8)

In total, 20 models were trained and tested:
4 classifiers for full MBTI label and 4 classifiers for each of the 4 MBTI dimensions.

Model training code can be found [here.](https://github.com/AsafGazit/MBTI-classifier/blob/master/code/model_training_clean.py)

### Results

<img src="https://github.com/AsafGazit/MBTI-classifier/blob/master/img/all_performance.png" alt="Classifiers performance" width="90%" height="90%">

The best full 16 classes MBTI classifier the SMOTE applied dataset with 57.75% accuracy.  Well above 6.25% of random guessing or 21.32% of “smart” guessing (picking the largest class, as the classes are imbalanced).

Let's examine the classifier’s confusion matrix:

<img src="https://github.com/AsafGazit/MBTI-classifier/blob/master/img/Full_MBTI_classifier_All_features_with_SMOTE.png" alt="Full MBTI with SMOTE CM" width="70%" height="70%">

Notably, out of all the full MBTI classifiers trained, the SMOTE applied is the only one that predicted instances in all classes, even if some of those were wrong. The TF-IDF classifier predicted instances in only 11 of the 16 available MBTI classes and the full features model only predicted 7 out of the 16 classes available.

It seems that for this multiclass task, the application of SMOTE has benefitted the training of the classifier. Although the actual accuracy score of the model is similar to the TF-IDF features model, it is less biased by the size of the large classes. This is what we hoped to achieve by the SMOTE application.

Let’s examine the dimension classifiers.
The classifiers trained on the dataset onwhich SMOTE is applied seem to perform well. Let's examine their confusion matrix: 

<img src="https://github.com/AsafGazit/MBTI-classifier/blob/master/img/dimension_classifier_All_features_smote_CM.jpg" alt="Axes MBTI with SMOTE CM" width="80%" height="80%">

It seems that even with SMOTE applied, some axes are not well predicted by the classifiers.

Now lets try to gather the specialist classifiers to a single MBTI prediction and see how they did:

<img src="https://github.com/AsafGazit/MBTI-classifier/blob/master/img/gathered_classifiers_performance.png" alt="Gathered classifiers performance" width="70%" height="70%">

Generally, the gathered models seem to perform worse than the full MBTI ones. This can imply 
1- The dimensions are not independent and identically distributed (IID) or the training dataset cannot express their independence due to bias. 
This is probably the case for the dataset without SMOTE applied. For a full MBTI classifier, the classes size difference may create conditional probability favorable classes (i.e given some axes are recognised, there is a high probability for a favourable class). This conditional ruling cannot be captured in the gathered axis classifiers.

2- The MLP model remains the same (configuration, size and training) for the full MBTI classifiers and the specialised classifiers. As no early stopping is applied and the complexity of the axis classifiers is assumed to be less than the full MBTI classisers, it is plausible that the training overfitted the classifiers to a higher extent, thus reducing the accuracy of the gathered model tested on the test set.

Lets see the SMOTE applied gathered classifier confusion matrix for more information:

<img src="https://github.com/AsafGazit/MBTI-classifier/blob/master/img/Dimension_classifiers_combined_All_features_with_SMOTE.png" alt="Gathered classifiers performance" width="70%" height="70%">

This shows an interesting result: there are 2 MBTI classes that are not predicted at all by the gathered classifier. This is not the case for the full MBTI classifier when SMOTE was applied.

### Discussion

- SMOTE vs NearMiss
In this comparison, mostly due to the size of the dataset, NearMiss was not really an appropriate method for application and was used as an alternative to SMOTE (oversampling vs undersampling performance). It seems that without the application of SMOTE for such a dataset, the class size variance creates a bias “hurdle” to be overcome for the trained classifier. The most class balanced predictions were made by the full MBTI classifier when SMOTE was applied.

- Specialised axes classifiers VS full MBTI indicator classifiers.
For all the axes, the specialised classifiers showed impressive performance ranging between 765 and 88% accuracy (for SMOTE applied datasets), however, when gathered to predict the full MBTI class this score drops.

- MLP model
The model used tried to, generally speaking, recreate the model described in <MEDART>.
The size of the network (# of layer/units) was not changed for any of the tests and there was no early stopping applied. 
This implies that there is a mismatch in the comparison between the full MBTI classifier and the single dimension axis as those may represent functions of different complexity. 
On the other hand, this aimed for experiment with the effects of variations in the data set on the model performance, a goal that can be considered satisfied even though the models are not fine-tuned.
