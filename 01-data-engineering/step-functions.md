# AWS Step Functions

- They are state machines used to design and orchestrate workflows
- They provide easy visualizations
- They provide advanced error handling and retry capabilities
- Offers audit of the history of the workflows
- They have the ability to wait for an arbitrary amount of time
- The maximum execution time of a State Machine is 1 year
- Use cases:
    - Create a state machine used to train a ML model. The workflow can invoke several other AWS services (SageMaker, Lambda, etc.)
    - Orchestrate batch jobs
