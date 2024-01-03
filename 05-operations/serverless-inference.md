# Serverless Inference

- Introduced in 2022
- We have to specify our container, memory and concurrency requirements
- The underlying capacity is automatically provisioned and scaled
- Good for infrequent or unpredictable traffic
- Will scale down to zero when there are no requests
- Monitoring happens with CloudWatch. Some important metrics:
    - `ModelSetupTime`
    - `Invocations`
    - `MemoryUtilization`

## SageMaker Inference Recommender

- Recommends best instance type and configuration to use for our models
- Automates load testing and model tuning
- Deploys to the optimal inference endpoint
- How it works:
    - We register our model to the model registry
    - It will benchmark different endpoint configurations
    - It will allow use to collect and visualize metrics to decide on instance types
    - Existing models from zoos may have benchmarks already
- Modes of execution:
    - Instance recommendations:
        - Runs load tests on recommended instance types
        - Takes about 45 minutes to complete
    - Endpoint recommendations:
        - Executes custom load tests
        - We specify instances, traffic patterns, latency requirements
        - Takes about 2 hours to complete

## Inference Pipelines

- It is a linear sequence of 2-15 containers
- We can have any combinations of pre-trained built-in algorithms or our own algorithms in Docker containers
- We can combine pre-processing, predictions and post-processing
- Spark ML and scikit-learn containers can be used:
    - Spark ML can be run with Glue or EMR
    - Spark ML containers will be serialized into MLeap format
- Can handle both real-time inference and batch transform