# Amazon Personalize

- It is a fully-managed recommendation engine, same as what is used on Amazon websites
- It is primarily used through an API:
    - We feed in data (purchases, ratings, impressions, cart adds, etc.) via S3 or API integration
    - We provide an explicit schema for the data in Avro format
    - After this it can be used via JavaScript or SDK:
        - `GetRecommendations`
        - `GetPersonalizedRanking`
- Provides console and CLI access to

## Features

- Real-time or batch recommendations
- Recommendations for new users and new items (the cold start problem)
- Contextual recommendations: device type, time of day, etc.
- Similar items
- Support for unstructured text input
- Intelligent user segmentation, for marketing campaigns
- Business rules and filters:
    - Filter out recently purchased items
    - Highlight premium content
    - Ensure a certain percentage of results are of the same category
- Promotions:
    - Inject promoted content into recommendations
    - Can find most relevant promoted content
- Trending now
- Personalized Rankings

## Terminology

- Datasets:
    - User data, items, interactions
- Recipes:
    - USER_PERSONALIZATION
    - PERSONALIZED_RANKING
    - RELATED_ITEMS
- Solutions:
    - Trains the model
    - Optimizes for relevance as well as our additional objectives
    - Hyperparameter Optimization (HPO)
- Campaigns:
    - Deploys our "solution version"
    - Deploys capacity for generating real-time recommendations

## Hyperparameter

- For User-Personalization, Personalized-Ranking recipes:
    - `hidden_dimension` - automatically optimized for us
    - `bptt` - back-propagation through time - RNN
    - `recency_mask` - weights recent events
    - `min/max_user_history_lengths_percentile` - filter out robots
    - `exploration_weight` - 0-1, controls relevance of items returned as result
    - `exploration_item_age_cut_off` - how far back in time we go
- For Similar-items recipe:
    - `item_id_hidden_dimension` - automatically optimized
    - `item_metadata_hidden_dimension` - automatically optimized

## Maintaining Relevance

- Keep the datasets current
    - Use incremental data imports
- Use PutEvents operation to feed in real-time user behavior
- Retrain the model:
    - New solution version
    - Updates every 2 hours by default
    - Recommended to do a full retrain (trainingMode=FULL) weekly

## Security

- Data is not shared across accounts
- Data may be encrypted with KMS
- Data may be encrypted at rest in our region
- Data in transit between our account and Amazon's internal system encrypted with TLS
- Access control via IAM
- Data in S3 must have appropriate bucket policy for Amazon Personalize to process
- Monitoring/logging with CloudWatch and CloudTrail

## Pricing

- We pay per ingestion per-GB
- Training: we per per-hour
- Inference: pay per TPS-hour (TPS - Transaction Per Second)
- Batch recommendations: per user or per item
