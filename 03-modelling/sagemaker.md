# Amazon SageMaker

- SageMaker is intended to manage the entire machine learning workflow
- SageMaker allows us to: fetch, clean and prepare data -> train and evaluate model deploy models and evaluate results in production -> repeat
- SageMaker Notebooks:
    - They are jupiter notebook instances running on EC2 machines
    - They have access to S3
    - They have access to libraries such as Scikit-Learn, Spark, Tensorflow
    - They have access to a wide variate of built-in modles
    - They have the ability to spin up training instances and deploy trained models
- Data prep on SageMaker:
    - Data usually comes from S3, the format varies with algorithms, of is RecordIO/Protobuf
    - We can ingest data from Athena, EMR, Redshift, Amazon Keyspaces DB
    - Apache Spark integrates with SageMaker
    - Scikit-Learn, numpy, pandas all at our disposal within the notebook
- SageMaker processing:
    - Processing jobs:
        - Copy data from S3
        - Spin up a processing container
        - Output processed data to S3
- SageMaker training:
    - We crate a training job:
        - We provide an URL to an S3 bucket with the training data
        - We provide an URL to an S3 bucket for the output
        - We provide a path to an ECR container with the training code
        - SageMaker will spin up ML compute resources
    - Training options:
        - Builtin training algorithms
        - Spark MLLib
        - Custom Python Tensorflow/MXNet code
        - PyTorch, Scikit-Learn, RLEstimator (reinforcement learning)
        - XGBoost, Hugging Face, Chainer
        - Our own Docker image
        - Algorithm purchased from AWS marketplace
- Deploying trained models:
    - We can save trained data to S3
    - We can deploy in two ways:
        - Persistent endpoint for making individual predictions on demand
        - SageMaker Batch Transform to get predictions for an entire dataset
    - Other options:
        - Inference Pipelines: used for more complex processing
        - SageMaker Neo: deploy to edge devices
        - Elastic Inference for accelerating deep learning models
        - Automatic scaling: increase the # of endpoints if needed
        - Shadow Testing: evaluates new models against currently deployed models to catch errors

## Linear Learner

- Linear regression:
    - Fit a line to our training data
    - Predictions based on that line
- It can handle both regression (numeric) predictions and classification predictions:
    - For classifications a linear threshold function is used
    - Can do binary or multi-class classification
- Input format:
    - RecordIO-wrapped protobuf (Float32 data only!) - most performant
    - CSV - first column is assumed to be the label
    - File or Pipe mode both supported
- Processing:
    - Training data must be normalized (all features are weighted the same)
    - Linear Learner can do the normalization automatically
    - Input data should be shuffled
- Training:
    - Uses stochastic gradient descent (SGD)
    - We can chose from optimization algorithms: Adam, AdaGrad, SGD, etc.
    - Multiple models can be optimized in parallel
    - We can tune L1, L2 regularization
- Important hyperparameters:
    - Balance_multiclass_weights: gives each class equal importance in loss functions
    - Learning_rate, mini_batch_size
    - L1 regularization
    - Wd - Weight decay (L2 regularization)
- Instance types:
    - Training:
        - Single or multi-machine CPU or GPU
        - I does help to have more than one machine/it does not help to have more than one GPU per machine

## XGBoost

- eXtreme Gradient Boosting
    - Boosted groups of decision trees
    - New trees made to correct the errors of previous trees
    - Uses gradient descent to minimize loss as new trees are added
- Can be used for classifications and for regressions (regression trees)
- Input format:
    - Accepts CSV, libsvm, recordIO-protobuf, Parquet
    - Models are serialized/deserialized with Pickle
    - Can be used as a framework within notebooks
        - Sagemaker.xgboost
    - Or as a built-in SageMaker algorithm
- Important hyperparameters:
    - Subsample: prevent overfitting
    - Eta: step size shrinkage, used to prevent overfitting
    - Gamma: minimum loss reduction to create a partition; larger =  more conservative
    - Alpha: L1 regularization; larger =  more conservative
    - Lambda: L2 regularization; larger =  more conservative
    - eval_metric:
        - Optimize an AUC, error, rmse
        - For example, if we care about false positives more than accuracy, we might use AUC here
    - scale_pos_weight:
        - Adjust balance of positive and negative weights
        - Helpful for unbalanced classes
        - Might set to sum(negative cases)/sum(positive cases)
    - max_depth:
        - Max depth of a tree
        - Too high value can cause overfit
- Instance types:
    - Uses CPU's only for multiple instances, does not support GPU for that
    - Is memory-bound, not compute-bound (M5 is a good choice)
    - As of XGBoost 1.2, single-instance GPU training is available (P2, P3)
        - We must set tree_method hyperparameter to gpu_hist
        - It trains more quickly and can be more cost effective

## Seq2Seq

- Input is a sequence of tokens, output is a sequence of tokens
- Used for:
    - Machine translation
    - Text summarization
    - Speech to text
- It is implemented with RNNs and CNNs with attention
- Input format:
    - Expects RecordIO-Protobuf format: tokens must be integers (this is unusual since most algorithms expect floating point data)
    - We need to provide tokenized text files (we cannot provide a simple Word file, for example)
    - We can convert data to Protobuf format using sample code provided
    - We must provide: training data, validation data and vocabulary files
- How it is used?
    - Training for machine translation can take days
    - Pre-trained models are available
    - Public training datasets are available for specific translation tasks
- Important Hyperparameters:
    - Batch_size
    - Optimizer_type: adam, sgd, rmsprop
    - Learning_rate
    - Num_layers_encoder
    - Num_layers_decoder
    - We can optimize for:
        - Accuracy
        - BLUE score: compare our translation against multiple reference translations
        - Perplexity: cross-entropy metric
- Instance Types:
    - Can only use GPU instance types (P3 for example)
    - Can only use a single machine for training - it can use multi-GPUs on one machine

## DeepAR

- Used for forecasting one-dimensional time series data
- Uses RNNs
- Allows us to train the same model over several related time series
- Can find frequencies and seasonality
- Input format:
    - JSON lines format: can be GZIP of Parquet
    - Each record must contain:
        - Start: Starting timestamp
        - Target: time series values
    - Optionally, each record can contain:
        - Dynamic_feat: dynamic features such as was a promotion applied to a product ina time series, product purchases
        - Cat: categorical feature
- How is it used?
    - We always include entire time series for training, testing and inference
    - We always use the entire dataset as test set, remove last time points for training. We evaluate on withheld values
    - We don't want to use large values for prediction length (>400)
    - Train on many time series and not just one when possible
- Important hyperparameters:
    - Context_length
        - Number of time points the model sees before making a prediction
        - Can be smaller than seasonalities; the model will lag one year anyhow
    - Epochs
    - mini_batch_rate
    - Learning_rate
    - Num_cells
- Instance Types:
    - We can use CPU or GPU machines
    - We can have single or multi-machine clusters
    - Recommendation: start with CPU (ml.c4.2xlarge, ml.c4.4xlarge), move upt to GPU if necessary (large models or large mini-batch sizes >512)
    - For inference only CPU supported
    - May need larger instances for tuning

## BlazingText

- Can be used for coupe of different things:
    - Text classification:
        - Predict labels for a sentence, if we train the system with existing sentences and with the labels associated with them. It's a supervised learning
        - Useful in web searches, information retrieval (intended to be used for sentences, not for entire documents)
    - Word2vec
        - Creates a vector representation of words
        - Semantically similar words are represented by vectors close to each other (word embedding)
        - It is useful for NLP, but is not an NLP algorithm in itself
        - Can be used for machine translation, sentiment analysis
- Input format:
    - For supervised mode (text classification):
        - One sentence per line
        - First "word" in the sentence is the string `__label__` followed by the actual label for the sentence
    - Accepts "augmented manifest text format"
    - For Word2vec just wants a text file with one training sentence per line
- How is it used?
    - Word2vec has multiple modes:
        - Cbow (Continuous Bag of Words)
        - Skip-gram
        - Batch skip-gram
            - Distributed computation over many CPU nodes
- Important hyperparameters:
    - Word2vec:
        - Mode: batch_skipgram, skipgram, cbow
        - Learning_rate
        - Window_size
        - Vector_dim
        - Negative_samples
    - Text classification:
        - Epochs
        - Learning_rate
        - Word_ngrams
        - Vector_dim
- Instance Types:
    - For cbow and skipgram, recommended a single ml.p3.2xlarge (any single CPU or single GPU instance will work)
    - For batch_skipgram, we can use single or multiple CPU instances
    - For text classification, recommended is a c5 node for less than 2GB of training data. For larger datasets, it is recommended to use a single GPU instance (ml.p2.xlarge, ml.p3.2xlarge)

## Object2Vec

- It creates low-dimensional dense embeddings of high-dimensional objects
- It is basically Word2vec, generalized to handle things other than just words
- Can compute nearest neighbors of objects
- Can be used for genre prediction, recommendations, etc.
- Input format:
    - Data must be tokenized into integers
    - Training data consists of pairs of tokens and/or sequences of tokens, for example:
        - Sentence-sentence
        - Labels-sequence (genre description?)
        - Customer-customer
        - Product-product
        - User-item
- How it is used?
    - We train it with 2 input channels, two encoders and a comparator
    - Encoder choices:
        - Average-pooled embeddings
        - CNN's
        - Bidirectional LSTM
    - Comparator is followed by a feed-forward neural network
- Important hyperparameters:
    - Usual deep-learning parameters: dropout, early stopping, epochs, learning rate, batch size, layers, activation function, optimizer, weight decay
    - Enc1_network, enc2_network: we choose the encoder type: hcnn, bilstm, pooled_embedding
- Instance Types:
    - Can only train on a single machine: CPU or GPU instances are supported, multi-GPU is also supported
    - Recommended instance types:
        - CPU: ml.m5.2xlarge, ml.p2.xlarge
        - GPU: P2, P3, G4dn, G5

## Object Detection

- Objectives:
    - Identify all the objects in an image with bounding boxes
    - Detect and classify objects with a single deep neural network
    - Classes are accompanied by confidence scores
    - Algorithms can be trained from scratch, or use pre-trained models based on ImageNet
- There are 2 object detection variants: MXNet and Tensorflow
- In both cases an image as taken as an input, the output is all instances of objects in the image with categories and confidence scores
- MXNet:
    - Uses a CNN with the Single Shot multibox Detector (SSD) algorithm: the base CNN can be VGG-16 or ResNet-50
    - In has both transfer learning mode and incremental training mode: we can use a pre-trained model for the base network weights instead of random initial weights
    - Uses flip, rescale and jitter to avoid overfitting
- Tensorflow:
    - Uses ResNet, EfficientNet, MobileNet models from Tensorflow Model Garden
- Input format:
    - MXNet: RecordIO or image format (jpeg, png)
    - With image format, we have to supply a JSON file for annotation data for each image
- Important Hyperparameters:
    - Mini_batch_size
    - Learning_rate
    - Optimizer: sgd, adam, rmsprop, adadelta
- Instance Types:
    - Use GPU instances for training (multi-GPU and multi-machine is also supported). Recommended instance types: ml.p2.xlarge - ml.p2.16xlarge, ml.p3..., G4dn, G5
    - Use CPU or GPU for inference: M5, P2, P3, G4dn

## Image Classification

- Objectives:
    - Assign one or more labels to an image
    - Does not tell where he objects are, just what objects are in the image
- There are separate versions of algorithms for MXNet and Tensorflow
- MXNet:
    - Full training mode: network is initialized with random weights
    - Transfer learning mode:
        - Network is initialized with pre-trained weights
        - The top fully-connected layer is initialized with random weights
        - Network is fine-tuned with new training data
    - Default image size is 3-channel 224 * 224 (ImageNet's dataset)
- Tensorflow: uses various Tensorflow Hub models (MobileNet, Inception, ResNet, EfficientNet)
    - Top classification layer is available for fine tuning or further training
- Important Hyperparameters:
    - Batch size, learning rate, optimizer
    - Optimizer-specific parameters: weight decay, beta 1, beta 2, eps, gamma
- Instance Types:
    - GPU instances for training: P2, P3, G4dn, G5. Multi-machine and multi-GPU is supported
    - Inference: same CPU or GPU instances are supported

## Semantic Segmentation

- Pixel level object classification
- Different from image classification, it assigns labels to each segment of an image
- It is also different from object detection, which assigns labels to bounding boxes
- Useful for self-driving vehicles, medical imaging diagnostics, robot sensing
- Produces a *segmentation mask*
- Input format:
    - It expects JPG/PNG images with annotations
    - We can use label maps to describe annotations
    - Supports augmented manifest image format for Pipe mode
    - For inference JPG images are accepted
- Under the hood is built on MXNet Gluon and Gluon CV
- It gives us a choice of 3 different algorithms:
    - Fully-Convolutional Network (FCN)
    - Pyramid Scene Parsing (PSP)
    - DeepLabV3
- For the underlying architecture of the neural network we have a few choices:
    - ResNet50
    - ResNet101
    - Both trained on ImageNet database
- Incremental training or training from scratch are supported
- Important Hyperparameters:
    - Epochs, learning rate, batch size
    - Choice of algorithm
    - Backbone Neural Network
- Instance Types:
    - Semantic segmentation: only GPU instances are supported for training: P2, P3, G4dn, G5. Training is limited to a single machine only
    - Inference: CPU (C5 or M5) or GPU (P3 or G4dn) are supported

## Random Cut Forest (RFC)

- Used for anomaly detection
- It works in an unsupervised setting
- It can detect:
    - Unexpected spikes in time series data
    - Breaks in periodicity
    - Unclassifiable data points
- It assigns and anomaly score to each data point
- Based on an algorithm developed by Amazon that they seem to be very proud of!
- Input format:
    - RecordIO-protobuf or CSV
    - Can use File of Pipe mode on either
    - Optional test channel for computing accuracy, precision, recall and F1 on labeled data (anomaly or not)
- How does it work under the hood?
    - Creates a forest of trees where each tree is a partition of the training data; looks at expected change in complexity of the tree as a result of adding a point into it
    - Data is sampled randomly
- RFC shows up in Kinesis Analytics as well; it can work on streaming data too
- Important Hyperparameters:
    - Num_trees: increasing it reduces the noise
    - Num_samples_per_tree: should be chosen such that 1/num_samples_per_tree approximates the ratio of anomalous to normal data
- Instance types:
    - Does not take advantage of GPUs
    - Recommended types are M4, C4 or C5 for training and ml.c5.xl for inference

## Neural Topic Model

- Usage:
    - Organizing documents into topics
    - Classify or summarize documents based on topics
- It's not just TF/IDF
- It is unsupervised algorithm, the underlying algorithm is being called "Neural Variational Inference"
- Input format:
    - Four data channels:
        - `train` (required)
        - `validation`, `test` and `auxiliary`  are optional
    - Accepts recordIO-protobuf or CSV
    - Input words must be tokenized into integers: every document must contain a count for every word in the vocabulary
    - The auxiliary channel is for the vocabulary
    - It can be used in file or pipe mode
- How is it used?
    - We define how many topics we want
    - These topics are latent representation based on top ranking words
    - One of two topic modelling algorithm offered by SageMaker
- Important Hyperparameters:
    - Lowering mini_batch_size and learning_rate can reduce validation loss at expense of training time
    - Num_topics
- Instance types:
    - GPU or CPU:
        - GPU recommended for training
        - CPU can be used for inference

## Latent Dirichlet Allocation (LDA)

- Topic modelling algorithm, it is not based on deep learning based algorithm
- It is an unsupervised algorithm, the topics themselves are unlabeled, they are just groupings of documents with a shared subset of words
- Can be used for things other than words:
    - Cluster customers based on purchases
    - Harmonic analysis in music
- Input format:
    - Takes in a training channel and an optional test channel
    - Input can be recordIO-protobuf or CSV
    - Each document has a counts for every word in vocabulary (in CSV format) - integer how often each word occurs in the document
    - Pipe mode only supported with recordIO
- How is it used?
    - Unsupervised; generates as many topics we specify
    - Optional test channel can be used for scoring results (per word log-likelyhood)
    - Functionally similar to NTM, but CPU-based => cheaper, more efficient
- Important Hyperparameters:
    - Num_topics
    - Alpha0 - Concentration parameter:
        - Smaller values generate sparse topic mixtures
        - Larger values (>1.0) produce uniform mixtures
- Instance types:
    - Single-instance CPU for training

## K-Nearest-Neighbors (KNN)

- Simple classification or regression algorithm
- Used for:
    - Classification: find the K closest points to a sample of points and return the most frequent label
    - Regression: find the K closest points to a sample point and return the average value
- Input format:
    - Train channel contains our data
    - Test channel emits accuracy or MSE
    - recordIO-protobuf or CSV training (first column is the label)
    - Supports file or pipe mode on either
- How is it used?
    - Data is first sampled
    - SageMaker includes a dimensionality reduction stage:
        - Avoid sparse data ("curse of dimensionality") at cost of noise/accuracy
        - `sign` of `fjlt` methods
    - Build an index for looking up neighbors
    - Serialize the model
    - Query the model for a given K
- Important Hyperparameters:
    - K: how many neighbors we look at
    - Sample_size
- Instance types:
    - Training can be done on CPU or GPU: MI.m5.2xlarge, MI.p2.xlarge
    - Inference:
        - Recommended CPU for lower latency
        - GPU for higher throughput on large batches

# K-Means Clustering

- Unsupervised clustering algorithm
- Data is divided into K groups, where each member of a group are as similar as possible to each other. Similarity is measured by Euclidean distance
- SageMaker provides Web-scale K-Means clustering
- Input format:
    - Channels: train channel, optional test channel
        - Training flag: ShardedByS3Key
        - Testing flag: FullyReplicated
    - Supported formats: recordIO-protobuf or CSV
    - Supports both file or pipe mode input
- How it is used?
    - Every observation is mapped to n-dimensional space (n - number of features)
    - The job of K-Means is to optimize the center of K clusters. "Extra cluster centers" may be specified to improve accuracy (will end-up reducing the value of `k` - `K = k*x`)
    - Algorithm:
        - Determine the initial cluster centers:
            - Random or k-means++ approach
            - K-means++ tries to make initial clusters far apart
        - Iterate over training data and calculate cluster centers
        - Reduce clusters from `K` to `k` using Lloyd's method
- Important Hyperparameters:
    - K: number of clusters
        - Choosing the right value for K can be tricky
        - We can use the "elbow method"
    - Mini_batch_size
    - Extra_center_factor (`x`)
    - Init_method
- Instance types:
    - CPU or GPU can be used, recommended is CPU
    - Only one GPU per instance can be used
    - Recommended instance types: 
        - ml.g4dn.xlarge for GPU, other g4dn and g4 are supported
        - p2, p3 for CPU

## PCA - Principal Component Analysis

- It is a dimensionality reduction technique:
    - Projects higher=dimensional data (lots of features) into lower-dimensional (like 2D plot) while minimizing loss of information
    - The reduced dimensions are called components:
        - First component has largest possible variability
        - Second component has the next largest possible variability
        - etc.
- It is unsupervised
- Input format:
    - recordIO-protobuf or CSV
    - File or pipe format is supported for both
- How it is used?
    - Under the hood a covariance matrix is created after which it used an algorithm called singular value decomposition (SVD)
    - Has 2 different mode of operation:
        - *Regular*: for sparse data and moderate number of observations and features
        - *Randomized*: for large number of observations and features, uses and approximation algorithm
- Important Hyperparameters:
    - Algorithm_mode
    - Subtract_mean: has the effect of unbiasing the data
- Instance types:
    - We can use both CPU and GPU

## Factorization Machines

- Specializes in dealing with sparse data, such as:
    - Click predictions
    - Item recommendations
    - Since an individual user doesn't interact with most pages/products the data is sparse
- It is supervised learning algorithm, it can do both classification or regression
- It is limited to pair-wise interactions, example: user-item
- Input format:
    - Must be recordIO-protobuf with Float32
    - CSV is not practical for sparse data
- How it is used?
    - Finds factors we can use to predict a classification (click or not, purchase or not) or value (predicted rating) given a matrix representing some pair of things (users & items)
    - Usually used in the context of recommender systems
- Important Hyperparameters:
    - Initialization methods for bias, factors and linear terms
    - Each one of these can be tuned, they can be uniform, normal or constant
- Instance types:
    - Both CPU and GPU instances can be used
    - CPU instances are recommended
    - GPU only works with dense data

## IP Insights

- Used for finding fishy behavior in weblogs
- It is an unsupervised learning technique for finding IP address usage patterns
- Identifies suspicious behavior from IP addresses:
    - Identity logins from anomalous IP's
    - Identify accounts creating resources from anomalous IP's
- Training data:
    - It can take in user names, account ID's directly; we don't need to pre-process data
    - Training channel is optional, can be used to compute the value for the AUC (area under the curve) score
    - Input has to be CSV data: Entity/IP 
- How it is used?
    - Uses a neural network to learn latent vector representations of entities and IP addresses
    - Entities are hashed and embedded; this requires a sufficiently large hash size
    - Automatically generates negates samples during training by randomly pairing entities and IP's
- Important Hyperparameters:
    - Num_entity_vectors:
        - Hash size
        - Recommended to be set to twice the number of unique entity identifiers
    - Vector_dim
        - Size of embedding vectors
        - Scales model size
        - Too large value can result in overfitting
    - Epochs, learning rate, batch size, etc.
- Instance types:
    - CPU or GPU
    - GPU instances are recommended: MI.p3.2xlarge or higher
    - We can use multiple GPUs for training
    - Size of CPU instances depends on vector_dim and num_entity_vectors

## Reinforcement Learning

### Q-Learning

- A specific implementation of reinforcement learning
- With Q-Learning we have:
    - A set of environmental states `s`
    - A set of possible actions in those state `a`
    - A value of each state/action `Q`
- We start of with `Q` values of 0
- Explore the space
- As bad things happen after a given state/action, reduce its `Q`
- As rewards happen after a given state/action, increase its `Q`

### Exploration Problem

- How do we efficiently explore all of the possible states?
    - Simple approach: always choose the action for a given state with the highest `Q`. If there is a tie, choose a random.
        - This approach is really inefficient and we might miss a lot of paths
    - Better way: introduce an epsilon term (Markov Decision Process - MDP)
        - If a random number is less than epsilon, we don't follow the highest `Q`, but choose at random
        - That way, exploration never totally stops
        - Choose epsilon can be tricky

### Reinforcement Learning in SageMaker

- Uses a deep learning framework with Tensorflow or MXNet
- Supports Intel Coach and Ray Rllib toolkits
- Custom, open-source or commercial environments supported:
    - MATLAB, Simulink
    - EnergyPlus, RoboSchool, PyBullet
    - Amazon Sumerian, AWS RoboMaker

## Automatic Model Tuning

- We define the hyperparameters we care about, the ranges we want to try and the metrics we are optimizing for
- SameMaker spins up a "Hyperparameter Tuning Job" that trains as many combinations as we allow
- The set of hyperparameters producing the best results can then be deployed as a model
- It learns as it goes, so it does not have to try every possible combination
- Best practices:
    - We should not optimize too many parameters at once
    - We should limit ranges to as small as possible
    - We should use logarithmic scale when appropriate
    - We should not run too many training jobs concurrently
    - We should make sure training jobs running on multiple instances report the correct objective metric in the end

## Apache Spark with SageMaker

- Pre-process data as normal with Spark
- Use sagemaker-spark library 
- `SageMakerEstimator` - use this instead of Spark AI library. It exposes algorithms such as: KMeans, PCA, XGBoost
- `SageMakerModel` - make inferences

## SageMaker Debugger

- Saves internal model state at periodical intervals
    - Gradients/tensors over time as a model is trained
    - We can define rules for detecting unwanted conditions while training
    - A debug job is run for each rule we configure
    - Logs and fires a CloudWatch event when the rule is hit
- Integrates with SageMaker Studio using SageMaker Studio Debugger dashboards
- It can automatically generate training reports
- It has several built-in rules:
    - Monitor system bottlenecks
    - Profile model framework operations
    - Debug model parameters
- Supported Frameworks and Algorithms:
    - Tensorflow
    - PyTorch
    - MXNet
    - XGBoost
    - SageMaker generic estimator (for use with custom training containers)
- Debugger API is available in GitHub - SMDebug
- Newer features:
    - SageMaker Debugger Insights Dashboard
    - Debugger ProfileRule:
        - ProfilerReport
        - Hardware system metrics (CPUBottleneck, GPUMemoryIncrease, etc.)
        - Framework Metrics (MaxInitializationTime, OverallFrameworkMetrics, etc.)
    - Built-in actions to receive notifications or stop straining
    - Profiling system resource usage and training

## SageMaker Autopilot

- It is a wrapper around AutoML
- Automates:
    - Algorithm selection
    - Data preprocessing
    - Model tuning
    - All infrastructure
- It does all the trial and error for us
- Workflow:
    - Load data from S3 for training
    - Select the target column for prediction
    - Automatic model creation
    - Model notebook is available for visibility and control
    - Model leaderboards: ranked list of recommended models from which we can pick one
    - Deploy and monitor the model, refine via notebook if needed
- We can add human guidance
- We can use it with or without code in SageMaker Studio or AWS SDKs
- We can use Autopilot on the following problem types:
    - Binary classification
    - Multiclass classification
    - Regression
- When running it on hyperparameter optimization, we can chose from the following algorithm types:
    - Linear Learner
    - XGBoost
    - Deep Learning (MLP's)
    - Ensemble mode
- Input data must be tabular CSV or Parquet
- Autopilot Explainability:
    - Integrates with SageMaker Clarify
    - Gives more transparency on how models arrive at predictions
    - Feature attribution:
        - Uses SHAP Baselines/Shapley values
        - Research from cooperative game theory
        - Assigns each feature an importance value for a given prediction

## SageNaker Model Monitor

- Get alerts via CloudWatch on quality deviations on our deployed models
- Visualizes data drift, for example: loan model starts giving people more credit due to drifting or missing input features
- Detects anomalies and outliers
- Detects new features
- Requires no coding
- Integrates with SageMaker Clarify:
    - Clarify detects potential bias
    - With Model Monitor we can monitor for bias and be alerted to new potential bias via CloudWatch
    - Clarify also helps explain model behavior
- Data is stored in S3 and secured
- Monitoring jobs should be scheduled via Monitor Schedule
- Metrics are emitted to CloudWatch
- Integrates with Tensorboard, QuickSight, Tableau or we can just visualize data within SageMaker Studio
- Monitor types:
    - Drift in data quality:
        - Relative to a baseline we create
        - "Quality" is just statistical properties and features
    - Drift in model quality (accuracy):
        - Can integrate with Ground Truth labels
    - Bias drift
    - Feature attribution drift
        - Based on Normalized Discounted Cumulative Gain (NDCG) score
        - This compares feature ranking of training vs. live data

## Deployment Safeguards

- Deployment Guardrails:
    - Can be deployed to asynchronous or real-time inference endpoints
    - We can control the shifting traffic to new models (Blue/Green deployments)
    - Provides auto-rollbacks
- Shadow Tests:
    - Allows us to compare performance of shadow variant to production
    - We can monitor models in SageMaker console and decide when to promote them
