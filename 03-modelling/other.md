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