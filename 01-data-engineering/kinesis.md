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