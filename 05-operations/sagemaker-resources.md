# Managing SageMaker Resources

## Choosing Instance Types

- In general, algorithms that rely on deep learning will benefit from GPU instances (P3, g4dn) for training
- Inference is usually less demanding and we can often get away with compute instances there (C5)
- GPU instances can be really pricey

## Managed Spot Training

- We can use EC2 spot instances for training with which we can save up to 90% over on-demand instances
- Spot instances can be interrupted by AWS, so we should use checkpoints saved on S3
- Using spot instances can increase training time as we need to wait for spot instances to become available

## Elastic Inference

- **Service is deprecated az of April 2023 in favor of "more cost effective" solutions!**
- Accelerates deep learning inference at fraction of cost of actually using a GPU instance for inference
- EL accelerators may be added alongside a CPU instance (ml.eia1.medium/large/xlarge)
- EL accelerators may also be applied to notebooks
- **EL only works with deep learning frameworks!** Works with Tensorflow, PyTorch and MXNet pre-built containers
- Works with custom containers built with EL-enabled Tensorflow, PyTorch or MXNet
- Works with Image Classification and Object Detection built-in algorithms

## Automatic Scaling

- Withing SageMaker, when deploying stuff to production, we can set up a scaling policy to define target metrics, min/max capacity, cooldown periods, etc.
- Works with CloudWatch
- Dynamically adjusts number of instances for a production variant
- Best practice: load test our configuration before using it

## SagerMaker and Availability Zones

- SageMaker will automatically attempt to distribute instances across availability zones
- We need more than one instance if we care about high availability
- VPCs should be configured with at least 2 subnets in different AZs to be able to take advantage of availability zones
