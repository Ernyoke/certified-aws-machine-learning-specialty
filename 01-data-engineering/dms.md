# DMS - Database Migration Service

- Offers quick migrations for databases into AWS
- It is resilient and self-healing
- The source database remains available during migration
- Supports:
    - Homogeneous migrations: example Oracle DB to Oracle DB
    - Heterogeneous migrations: example Microsoft SQL Server to Aurora
- Uses continuous data replication with CDC
- We must create an EC2 instance to perform the replication tasks

