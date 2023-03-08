# EMR - Elastic MapReduce

- It is a managed Hadoop framework running on EC2 instances
- Includes Spark, HBase, Presto, Flink, Hive and more
- Offers several integration points with other AWS services

## EMR Clusters Node Types

- Master node: manages the cluster, it is a single EC2 instance
    - Every cluster has a master node
    - We can have clusters with only master nodes
- Core node: 
    - Hosts HDFS data and runs tasks
    - Can be scaled up and down, but with some risks
    - Multi-node clusters have at least 1 core node
- Task node: runs tasks, does not host data
    - No risk of data loss when it gets removed
    - Good use of spot instances, they are only used for computation

## EMR Cluster Usage

- There are a couple ways to use an EMR cluster:
    - Transient cluster: are configured to be automatically terminated when all the steps of process are completed
    - Long running cluster: should be manually terminated by head

## AWS Integrations

- EMR uses Amazon EC2 instances for nodes in a cluster
- Nodes are launched in VPC networks
- We can use Amazon S3 to read input data from and to store output (instead of HDFS)
- WE can use AWS Data Pipeline to schedule and start clusters

## EMR Storage

- The default storage on Hadoop is HDFS - it is a distributed scalable filesystem
- HDFS is an ephemeral storage, when the cluster is shut down, the data can be lost
- EMRFS: provides access to data stored on S3 as it was on HDFS

## Apache Spark MLLib

- Offers several machine learning algorithms
- All of these algorithms are implemented in a distributed and scalable way
- Types of algorithms:
    - Classification: logistic regression, naive Bayes
    - Regression
    - Decision trees
    - Recommendation engine (ALS)
    - Clustering (K-Means)
    - Topic Modelling (LDA)
    - ML workflow utilities: pipelines, feature transformation, persistence
    - SVD, PCA, statistics

## EMR Notebook

- It is a similar concept as Zeppelin (a tool for running Spark code interactively in a notebook format)
- Notebooks are backed up by S3
- We can provision EMR clusters form a notebook
- EMR notebooks are hosted in VPCs
- We can only access EMR notebooks from AWS console

## EMR Security

- EMR uses IAM policies
- EMR supports Kerberos authentication