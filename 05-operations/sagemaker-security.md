# SageMaker Security

## General AWS Security

- Use Identity and Access Management (IAM) whenever it is possible
- Use MFA whenever possible
- Use SSL/TLS when connecting to anything
- Use CloudTrail to log API and user activity
- Use encryption
- Be careful with PII

## Protecting Data at Rest in SageMaker

- Use AWS KMS whenever possible:
    - KMS keys are accepted by notebooks and all SageMaker jobs: training, tuning, batch transformation and endpoints
    - Notebooks and everything under `/opt/ml` and `/tmp` in the Docker containers can be encrypted with KMS keys
- Secure training with S3:
    - We can use encrypted S3 buckets for training data and hosting models
    - S3 can also use KMS

## Protecting Data in Transit in SageMaker

- All traffic supports TLS/SSL within SageMaker
- We can use IAM roles assigned to SageMaker to give permissions to access resources
- Inter-node training communication may be optionally encrypted. Encrypting data can increase training time and cost within deep learning

## SageMaker + VPC

- Training jobs run in a VPC (Virtual Private Cloud)
- We can use a private VPC for even more security, although we might have to set up VPC endpoints for services such as S3
- Notebooks are Internet-enabled by default. If disabled, our VPC needs an interface endpoint (PrivateLink) or NAT Gateway, and allow outbound connections for training and hosting to work
- Training and Inference Containers are also Internet-enabled by default. Network isolation is an option, but this also prevents S3 access

## SageMaker + IAM

- We can set specific permissions for training, endpoints, notebooks, etc.
- We can use predefined policies, such as:
    - `AmazonSageMakerReadOnly`
    - `AmazonSageMakerFullAccess`
    - `AdministratorAccess`
    - `DataScientist`

## SageMaker Logging and Monitoring

- CloudWatch can log, monitor and alarm on:
    - Invocations and latency of endpoints
    - Health of instance nodes (CPU, memory, etc)
    - Ground Truth (active workers, how much they are doing)
- CloudTrail records actions from users, roles and services within SageMaker