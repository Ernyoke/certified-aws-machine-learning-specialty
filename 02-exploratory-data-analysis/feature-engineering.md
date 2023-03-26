# Feature Engineering

- Feature engineering: applying knowledge of the data and knowledge of the used model in order to create better features to train the model with
- We can't just throw in raw data and expect we get good results
- The curse of dimensionality:
    - Too many features can be a problem - leads to sparse data
    - Every feature is a new dimension
    - Much of feature engineering is selecting the feature most relevant to the problem at hand (domain knowledge can be relevant)
    - Unsupervised dimensionality reduction techniques can also be employed to distill many features into fewer features (PCA - principle component analysis, K-Means)

## Imputation of Missing Data

### Mean Replacement

- Replace missing values with the mean value from the rest of the column (not rows). A column represents a single feature; it only makes sense to take the mean from other samples of the same feature
- It is fast and easy, wont affect mean or sample size of overall data set
- Median may be a better choice than mean when outlier are present
- It's generally pretty terrible, because:
    - Only works on column level, misses correlations between features
    - We can't use it on categorical features (we could use the most frequent value)
    - Not very accurate

### Dropping

- Can be helpful if:
    - Not many rows contain missing data 
    - ...and dropping those rows does not bias the data
    - ...and we don't have a lot of time
- It is never going to be the right answer for the "best" approach, in fact almost anything else is better option

## Machine Learning for Imputation

- KNN: find K nearest (most similar) rows and average their values
    - Assumes numerical data, not categorical
    - There are ways to handle categorical data (Hamming distance), but categorical data is probably better served by deep learning
- Deep Learning:
    - Build a machine learning model to impute data for another machine learning model
    - Works well for categorical data, but it is complicated to be built
- Regression:
    - Find linear or non-linear relationships between the missing feature and other features
    - Most advanced technique: MICE (Multiple Imputation by Chained Equations)

## Handling Unbalanced Data

- It is a large discrepancy between positive and negative cases of data. For example: fraud detection, fraud is rare and most rows will be not fraud
- Unbalanced data sets are mainly a problem for neural networks
- Dealing with unbalanced data:
    - Oversampling: duplicate samples from the minority class. Can be done are random
    - Undersampling:
        - Instead of creating more positive samples, remove the negative ones
        - Throwing data away is usually not the right answer

## SMOTE

- Synthetic Minority Over-sampling TEchnique
- Artificially generate new samples of the minority class using nearest neighbors:
    - Run KNN of each sample of the minority class
    - Create a new sample from the KNN result (mean of neighbors)
- Generally better than just oversampling

## Adjusting Thresholds

- When making predictions about classifications, we have some sort of threshold of probability at which point we flag something as the positive
- If we have to many false positives, one way to fix that is to simply increase that threshold

## Binning

- Used to bucket observations together based on ranges of values
- Quantile binning: categorizes data by their place in the data distribution. Ensures even sizes for the bins
- Binning is used to transform numeric data to ordinal data
- Useful when there is uncertainty in the measurements

## Transforming

- Applying some functions to a feature to make it better suited for training
- Feature data with exponential trend may benefit from a logarithmic transform

## Encoding

- Transforming data into some new representation required by the model
- One-hot encoding:
    - Creates buckets for every category
    - The bucket for the current category has a 1, all others have 0 value
    - Vert common in deep learning, where categories are represented by individual output neurons

## Scaling / Normalization

- Some models prefer feature data to be normally distributed around 0 (most neural nets)
- Most models require feature data to be scaled to comparable values, otherwise features with larger magnitudes will have more weight than they should

## Shuffling

- Many algorithms benefit rom shuffling the training data
- Otherwise they may learn from residual signals in the training data resulting from the order in which they were collected