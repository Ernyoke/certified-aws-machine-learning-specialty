# AWS Data Pipeline

- It is a service uses for moving data from one place to another, essentially it is an ETL service
- Destinations used for Data Pipeline include S3, RDS, DynamoDB, Redshift and EMR
- Data Pipeline is used to manage task dependencies (it is an orchestrator)
- The orchestration itself is running on EC2 instances
- It has capabilities for retries and notifications failures
- Data sources may be on-premises
- It is highly available
- Use-cases:
    - Move data from RDS into S3
    - Move data from DynamoDB to S3

## AWS Data Pipeline vs Glue

- Glue ETL:
    - Runs Apache Spark code using Scala or Python
    - We do not worry about configuring and managing the resources, it is serverless
- Glue Data Catalog:
    - Is used to make data available for Athena or Redshift Spectrum
- Data Pipeline:
    - It is an orchestration service
    - Offers more control over the environment, compute resources are running in our account
    - Allows access to EC2 and EMR instances
- Both Glue and Data Pipeline are ETL services, Glue focuses on Apache Spark jobs, Data Pipelines gives us more control about the environment and technology used