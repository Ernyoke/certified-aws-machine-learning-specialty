# Amazon Lex

- Billed as the inner workings of Alexa
- It is a Natural-language chatbot engine
- A Bot is build around Intents:
    - Utterances invoke intents ("I want to order a pizza")
    - Lambda functions are invoked to fulfill the intent
    - Slots specify extra information needed by the intent. Example: pizza size, toppings, crust type, when to deliver, etc.
- Can be deployed to AWS Mobile SDK, Facebook Messenger, Slack and Twilio

## Amazon Lex Automated Chatbot Designer

- Used to automate the creation of a chatbot
- We provide existing conversation transcripts
- Lex applies NLP and deep learning, removing overlaps and ambiguity
- Extracts intents, user requests, phrases, values for slots
- Ensures intents are well defined and separated
- Integrates with Amazon Connect transcripts