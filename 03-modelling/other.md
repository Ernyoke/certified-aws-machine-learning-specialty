# Other Random Stuff

## t-SNE

- Definition (from: https://www.datacamp.com/tutorial/introduction-t-sne):
    - t-SNE (t-distributed Stochastic Neighbor Embedding) is an unsupervised non-linear dimensionality reduction technique for data exploration and visualizing high-dimensional data
    - Non-linear dimensionality reduction means that the algorithm allows us to separate data that cannot be separated by a straight line
- t-SNE vs PCA:
     - PCA (Principal Component Analysis) is a linear technique that works best with data that has a linear structure
     - t-SNE is a nonlinear technique that focuses on preserving the pairwise similarities between data points in a lower-dimensional space

## Synthetic Minority Oversampling Technique (SMOTE)

- It is an oversampling approach in which the minority class is over-sampled by creating "synthetic" examples rather than by over-sampling with replacement
- The minority class is oversampled by taking each minority class sample and introducing synthetic examples along the line segments joining any/all of the minority class nearest neighbors
- Depending upon the amount of over-sampling required, neighbors from the k nearest neighbors are randomly chosen

## Naive Bayesian and Full Bayesian Networks

- The Naive Bayes model and a full Bayesian network are both probabilistic models used in machine learning and statistics, but they have significant differences in terms of complexity and modeling assumptions
- Naive Bayes is simple and assumes feature independence, making it suitable for basic classification tasks with modest-sized datasets
- Full Bayesian Networks are more complex, allowing for flexible modeling of dependencies among variables. They are used in situations where variables interact in intricate and conditional ways
- More details: [Simple Explanation-Difference between Naive Bayes and Full Bayesian Network Model](https://medium.com/@mansi89mahi/simple-explanation-difference-between-naive-bayes-and-full-bayesian-network-model-505616545503#:~:text=In%20summary%2C%20the%20main%20difference,modeling%20of%20dependencies%20among%20variables.)

## Weight Initialization

- Xavier Weight Initialization: 
    - It is an initialization method is calculated as a random number with a uniform
    probability distribution (U) between the range `-(1/sqrt(n))` and `1/sqrt(n)`, where n is the number of inputs to the node.
- Normalized Xavier Weight Initialization:
    - It is calculated as a random number with a uniform probability distribution (U) between the range `-(sqrt(6)/sqrt(n + m))` and `sqrt(6)/sqrt(n + m)`, where n us the number of inputs to the node (e.g. number of nodes in the previous layer) and m is the number of outputs from the layer (e.g. number of nodes in the current layer)
- He Weight Initialization:
    - The initialization method is calculated as a random number with a Gaussian probability distribution (G) with a mean of 0.0 and a standard deviation of `sqrt(2/n)`, where n is the number of inputs to the node
    - Recommended for networks with ReLU activation function

## Cross-Validation

- See: https://docs.aws.amazon.com/machine-learning/latest/dg/cross-validation.html
- Cross-validation is a technique for evaluating ML models by training several ML models on subsets of the available input data and evaluating them on the complementary subset of the data
- Use cross-validation to detect overfitting, ie, failing to generalize a pattern
- Cross-validation types:
    - *K-fold cross validation*: In Amazon ML, we can use the k-fold cross-validation method to perform cross-validation. In k-fold cross-validation, we split the input data into k subsets of data (also known as folds). We train an ML model on all but one (k-1) of the subsets, and then evaluate the model on the subset that was not used for training. This process is repeated k times, with a different subset reserved for evaluation (and excluded from training) each time
    - *Holdout cross validation*: the model trains on a specific set and is tested on another specific set
    - *Stratified K-fold cross validation*: recommended for unbalanced dataset
