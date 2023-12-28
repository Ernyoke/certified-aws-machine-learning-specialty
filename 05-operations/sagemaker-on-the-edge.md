# SageMaker on the Edge

## SageMaker Neo

- Deploy SageMaker models out to Edge devices
- Edge devices are:
    - ARM, Intel, nVidia processors
    - Embedded in whatever (cars, iot, etc.)
- Can take and code written in Tensorflow, MXNet, PyTorch, ONNX, XGBoost, DarkNet, Keras and optimizes code for specific devices
- Consists of a compiler and a runtime

## Neo + AWS IoT Greengrass

- Neo-compiled models can be deployed to an HTTPS endpoint hosted on C5, M5, M4, P3 or P2 instances (kind of defeats the purpose)
- We can deploy Neo compiled code to IoT Greengrass as well
- Inference happens at the edge with local data, using a model trained in the cloud
- Uses Lambda inference for applications