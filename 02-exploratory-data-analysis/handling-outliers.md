# Handling Outliers

## Variance (sigma squared - σ^2)

- Measures how "spread-out" the data is
- It is the average of the squared differences from the mean
- Example: find the variance of the data set: 1, 4, 5, 4, 8
    - Calculate the mean: 4.4
    - Find the differences from the mean: -3.4, -0.4, 0.6, -0.4, 3.6
    - Find the squared differences: 11.56, 0.14, ...
    - FInd the average of the squared differences: 5.04

## Standard Deviation (sigma - σ)

- It is the square root of the variance
- Standard deviation is usually used to identify outliers. Data points that lie more than one standard deviation from the mean can be considered outliers
- We can talk about how extreme a data point is by talking about "how many sigmas" away from the mean it is

## Dealing with Outliers

- Sometimes it is appropriate to remove outliers from the training data
- We have to do this responsibly, we should understand what we are doing
- AWS's Random Cut Forest algorithm creeps into many of its services (QuickSight, Kinesis Analytics, SageMaker) - it is made for outlier detection