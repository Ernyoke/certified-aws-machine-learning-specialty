# AWS Batch

- Allows us to run batch jobs based on Docker images
- Provisioning of the underlying infrastructure is done by AWS dynamically (EC2 and Spot Instances)
- AWS figures out the optimal quantity and the type of the instances based on the volume of jobs and their requirements
- There is no need to manage any clusters
- We pay for the underlying EC2 instances
- There can be 2 types of jobs:
    - Scheduled Batch jobs using CloudWatch Events
    - Orchestrated Batch Jobs using AWS Step Functions

## AWS Batch vs Glue

- Glue ETL:
    - Runs Apache Spark code using Scala or Python
    - We do not worry about configuring and managing the resources, it is serverless
- Glue Data Catalog:
    - Is used to make data available for Athena or Redshift Spectrum
- AWS Batch:
    - Can be used for any long running computing job regardless of the type (not just for ETL)
    - Resources are created inside our account and they are managed by AWS
    - Recommended for non-ETL workloads