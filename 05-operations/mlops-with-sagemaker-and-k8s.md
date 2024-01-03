# MLOps with SageMaker and Kubernetes

- Integrates SageMaker with Kubernetes-based ML infrastructure
- There are a couple of tools available for this integration:
    - Amazon SageMaker Operators for Kubernetes
    - Components for Kubeflow Pipelines
- Enables hybrid ML workflows (on-prem + cloud)
- Enables integration for existing ML platforms built on Kubernetes/Kubeflow

## SageMaker Projects

- SageMaker Studio's native MLOps solution with CI/CD:
    - Build images
    - Prep data, feature engineering
    - Train models
    - Evaluate models
    - Deploy models
    - Monitor and update models
- Uses code repositories for building and deploying ML solutions
- Uses SageMaker Pipelines defining steps