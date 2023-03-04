# AWS Glue

## Glue Data Catalog

- It is a metadata repository for all the data tables
- It has automated schema inference and schemas are versioned
- Integrates with Athena or Redshift Spectrum (with Schema and data discovery)
- Glue Crawlers can help build the Glue Data Catalog

## Glue Data Catalog - Crawlers

- Crawlers can go through data and infer schemas and partitions
- Works with JSON, Parquet, CSV, relational store
- Crawlers offer support for: S3, Redshift, Amazon RDS
- Crawlers can run on schedule or on-demand
- We have to create an IAM role for crawlers to be able to access data stores

## Glue and S3 Partitions

- Glue crawlers will extract partitions depending on how the data is organized in S3
- Examples: device sends sensor data every hour, the data is organized based on year, month, day, hour, etc.

## Glue ETL

- ETL: Extract, Transform, Load
- Allows us to transform data, clean data and enrich data
- Glue ETL can generate code in Python or Scala which can be modified by us or it allows us to provide our own Spark or PySpark scripts
- The target of a Glue job can be S3, JDBC or a table in Glue Data Catalog
- Glue ETL is fully managed, cost effective, pay only for the resources consumed
- Jobs are run on a serverless Spark platform
- We can use Glue Schedules to schedule jobs or Glue Triggers to trigger jobs based on events
- Transformations:
    - Bundled Transformations:
        - DropFields, DropNullFields: remove fields
        - Filter: filter records based on a condition
        - Join: join tables
        - Map: add new fields, delete fields, perform lookups
    - Machine Learning Transformations:
        - FindMatches ML: identify duplicates or matching records even when the records do not match exactly
    - Format conversions: CSV, JSON, Avro, Parquet, ORC, XML
    - Apache Spark transformations (ex. K-Means alg.)