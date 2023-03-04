# Kinesis

## Overview

- Kinesis is managed alternative for Apache Kafka
- Kinesis is great for:
    - Application logs, metrics, IoT, clickstreams
    - "real-time" big data
    - Streaming and processing frameworks (Spark, NiFi, etc.)
- Data is automatically replicated synchronously to 3 AZs


## Kinesis Data Streams

- Streams are divided in ordered Shards (Partitions)
- Data retention is 24 hours by default, can be set to at maximum 365 days
- Data can be reprocessed/replayed
- Multiple applications can consume the same data stream
- Once data in inserted in Kinesis, it can't be deleted (immutability)
- Records in Kinesis Data Streams can be up to 1MB

## Kinesis Data Streams Capacity Modes

- There 2 types of capacity modes:
    - **Provisioned mode**:
        - We choose the number of shards provisioned, we can scale the stream manually or using an API
        - Each shard can get 1MB/s (or 1000 records per second)
        - Out throughput: each shard gets 2MB/s applicable for classic or fan-out consumers
        - We pay per shard per hour
    - **On-demand mode**:
        - We don't need to provision or manage the capacity
        - Default capacity provisioned is 4MB/s ir or 4000 records per second
        - The data stream will scale automatically, based on the observed data during the last 30 days
        - We pay per stream per hour and data in/out per GB
- Data Stream limits:
    - Producer:
        - 1MB/s or 1000 messages at write per shard
        - `ProvisionedThroughputException` is thrown in cases we reach this limit
    - Consumer Classic:
        - 2MB/s read per shard across all consumers
        - 5 API calls per second per shard across all consumers
    - Data Retention:
        - 24 hours data retention by default
        - Can be extended to 365 days

## Kinesis Data Firehose

- Used for storing data into target destiantions
- Firehose is a near-realtime service relying on batch writes
- Firehose destination for data ingestion:
    - Amazon S3
    - Amazon Redshift (copy through S3)
    - Amazon OpenSearch
    - Other third party products (Splunk, DataDog, etc.)
    - Custom destination with a valid HTTP endpoint
- Firehose is a fully managed service, no administration required
- Provides automatic scaling
- Supports many data formats and data conversions (CSV/JSON to Parquet/ORC - only for S3)
- Data transformation is accomplished with AWS Lambda
- Supports compression if the target is S3

## Kinesis Data Streams vs Firehose

- Streams:
    - We can write custom code for producers/costumers
    - It is real time(~200 ms latency for classic, ~70 ms latency for enhanced fan-out)
    - Offers automatic scaling it it used in on-demand mode
    - Data Storage is between 1 to 365 days, replay capability and multi consumers
- Firehose:
    - Fully managed, used to ingest data into S3, ElasticSearch, Splunk, etc.
    - Offers serverless data transformation with AWS Lambda
    - Near real time (lowest buffering time is 1 minute)
    - Scales automatically
    - No data storage, data is discarded after buffering time ellipses

## Kinesis Data Analytics

- Input streams for Kinesis Data Analytics can be either a Kinesis Data Stream or Firehose
- With Kinesis Data Analytics we can analyze incoming data using SQL or Apache Flink
- Use-cases for Data Analytics:
    - Streaming ETL: select columns, make simple transformations on streaming data
    - Continuos metric generation: for example live leader boards
    - Responsive analytics: look for certain criteria and build alerts (filtering)
- Features:
    - We only pay for resources consumed
    - It is serverless, scales automatically
    - It is using IAM permissions to access streaming sources and destinations
    - We can use SQL or Apache Flink for the analytics script
    - It offers schema discovery
    - A Lambda function can be used for preprocessing the data

## Machine Learning with Kinesis Data Analytics

- We can use 2 machine learning algorithms with Kinesis Data Analytics SQL scripts:
    - `RANDOM_CUT_FOREST`: can be used for anomaly detection on numeric columns. It is using recent history to compute the model
    - `HOTSPOT`: locate and return information about relatively dense regions in a dataset

## Kinesis Video Streams

- Video Streams is used for handling video input from different sources such as:
    - Security cameras
    - AWS DeepLens
    - Smartphone camera
    - Audio feeds
    - RADAR data
    - RTSP camera
    - etc.
- Offers video playback capability
- Consumers can be:
    - Custom code (Tensorflow, MXNet, etc.)
    - AWS SageMaker
    - Amazon Rekognition
- We can keep the video data for an interval between 1 hour up to 10 years
- Use-cases:
    - Decode video frames and do ML-based inference with SageMaker
    - Detect people in video streams
    - etc.

# Kinesis Summary

- Kinesis Data Streams: real-time streams for ML applications
- Kinesis Data Firehose: ingest massive data in near-real time
- Kinesis Data Analytics: real-time ETL/ML algorithms on streams
- Kinesis Video Streams: real-time video streams to create ML applications

