# Amazon QuickSight

- It is a fast, easy, cloud-powered business analytics service
- It is an application which lets us create dashboards, graphs, charts and reports based on stored data
- Allows all employees in an org. to:
    - Build visualizations
    - Perform ad-hoc analysis
    - Quickly get business insights from data
- It can be used on multiple devices (browser, mobile)
- It is a serverless application
- It can connect to different data stores such as:
    - Redshift
    - Aurora/RDS
    - Athena
    - EC2 hosted databases
    - Files (S3 or on-premises) including different formats (Excel, CSV, TSV, log format)
- It can work on top of AWS IoT Analytics
- Allows some limited ETL for data preparation

## SPICE

- Amazon QuickSight under the hood is using SPICE:
    - Super-fast, Parallel, In-memory Calculation Engine
    - Uses columnar storage, in-memory, machine code generation
    - Accelerates interactive queries on large datasets
- In-memory calculations are limited to 10GB per user

## QuickSight Use-cases

- Interactive ad-hoc exploration/visualization of data
- Dashboards and KPI reports
- Analyze and visualize data from:
    - Logs from S3
    - On-premises databases
    - AWS (RDS, Redshift, Athena, S3)
    - SaaS applications, such as SalesForce
    - Any JDBC/ODBC data source

## Machine Learning Insights

- It is a feature of QuickSights
- Features:
    - Anomaly detection: under the hood is using Amazon's RandomCutForrest algorithm
    - Forecasting: uses also RandomCutForrest
    - Auto-narratives: build rich dashboards with embedded narratives

## QuickSight Q

- It is an add-on to QuickSight
- It is machine learning powered, it can answer business questions with Natural Language Processing (NLP)
- Personal training is required for usage
- We must set up topics associated with datasets
- Fields and datasets must be NLP-friendly

## QuickSight Paginated Reports

- They are reports designed to be printed
- They can be based on existing Quicksight dashboards

## QuickSight Anti-Patterns

- Highly formatted canned reports (NOT TRUE ANYMORE after Paginated Reports were added!)
- ETL: use Glue instead

## Security

- Offers MFA
- VPC connectivity is possible, private VPC access is also possible
- Row-level security is offered
- Enterprise edition only: column-level security

## QuickSight User Management

- Users are defined via IAM or email signup
- SAML-based single sign-on
- Active Directory integration for enterprise edition only