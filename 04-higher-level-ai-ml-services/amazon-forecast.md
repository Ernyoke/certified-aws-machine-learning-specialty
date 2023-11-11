# Amazon Forecast

- It's a time series analysis as-a-service
- It is a fully managed service to deliver highly accurate forecast with ML
- "AutoML" feature let's us choose the best model for our time series data: ARIMA, DeepAR, ETS, NPTS, Prophet
- Works with any time series, examples: price series, promotions, economic performance
- It also can combine the input data with associated data to find relationships
- Use cases: inventory planning, financial planning, resource planning
- Based on "dataset groups", "predictors" and "forecasts"

## Forecast Algorithms

- CNN-QR:
    - Convolutional Neural Network - Quantile Regression
    - Computationally expensive
    - Best for large datasets with hundred of time series
    - Accepts related historical time series data and metadata
- DeepAR+:
    - It is using Recurrent Neural Networks
    - Computationally expensive
    - Best for large datasets
    - Accepts related forward-looking time series and metadata
- Prophet:
    - Additive model with non-linear trends and seasonality
    - Is not as expensive as CNN-QR or DeepAR+
- NPTS:
    - Non-Parametric Time Series
    - Good for sparse data. Has variants for seasonal/climatological forecasts
- ARIMA:
    - Autoregressive Integrated Moving Average
    - Commonly used for simple datasets (<100 time series)
- ETS:
    - Exponential Smoothing
    - Commonly used for simple datasets (<100 time series)