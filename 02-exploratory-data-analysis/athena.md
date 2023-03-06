# Amazon Athena

- It is an interactive query service for S3 (SQL), we don't need to load data anywhere, it stays in S3
- It is powered by Presto under the hood
- It is serverless
- Supports many data formats:
    - CSV (human readable)
    - JSON (human readable)
    - ORC (columnar, splittable)
    - Parquet (columnar, splittable)
    - Avro (splittable)
- The data can be unstructured, semi-structured or structured
- Examples of usage:
    - Ad-hoc queries of web logs
    - Analyze CloudTrail logs
    - Integration with Jupyter, Zeppeling, RSTudio
- Integrates with AWS QuickSight
- Integrates via ODBC/JDBC with other visualization tools

## Athena Cost Model

- We pay-per-use:
    - $5 per TB scanned
    - Successful or cancelled queries count, failed don't
    - No charge for DDL
- We can save a lot of money by using columnar formats such as ORC, Parquet

## Athena Security

- Athena uses IAM for bucket access
- Results are encrypted at rest in an S3 staging directory
- Cross-account access is possible with bucket policies
- TLS encryption for in-transit data Athena and S3

## Anti-Patterns

- Highly formatted reports/visualization (use QuickSight)
- ETL (use Glue instead)