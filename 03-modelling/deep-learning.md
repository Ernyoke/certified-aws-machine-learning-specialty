# Deep Learning

- Deep Learning frameworks:
    - Tensorflow / Keras
    - Apache MXNet

- Types of neural networks:
    - Feedforward Neural Network
    - Convolutional Neural Network (CNN):
        - Mainly used for 2D data, example image classification
    - Recurrent Neural Network (RNN):
        - Manly used for sequences in time or for things which have an order for them, example stock prediction, understand words of sentences, translation
        - Flavors of RNN: LSTM, GRU

## Activation Functions

- It is a function inside of a given neuron, that sums up the all the inputs and decides what output should be sent to the next layer of neurons
- Types of activation functions:
    - Linear: 
        - It doesn't really "do" anything, it outputs the input data
        - Can't do backpropagation
        - There is no point in having more than one layer of liner activation functions
    - Binary Step function:
        - It's an ON/OFF function
        - It can't handle multiple classification - it is a binary function
        - Vertical slopes don't work well with calculus (derivate is infinite)
    - Non-Linear activation functions:
        - They can create complex mappings between input and outputs
        - They allow backpropagation (they have useful derivatives)
        - They allow to have multiple layers
- Non-linear activation functions:
    - Sigmoid or Logistic / TanH (hyperbolic tangent):
        - They are nice and smooth
        - Sigmoid will scale the input between 0 to 1, TanH will scale the input between -1 to 1. They change slowly for high or low values ("Vanishing Gradient")
        - They are computationally expensive
        - Tanh is generally preferred over sigmoid
    - Rectified Linear Unit (ReLU):
        - Very popular choice for activation function
        - Very easy and fast to compute
        - In case the input are zero or negative, we have a linear function and all of its problems ("Dying ReLU problem")
    - Leaky ReLU:
        - Solves the "Dying ReLU problem" by introducing a negative slope bellow 0
    - Parametric ReLU (PReLU):
        - ReLU, but the slope is the negative part is learned via backpropagation
        - Complicated and computationally intensive
    - Other ReLU variants:
        - Exponential Linear Unit  (ELU)
        - Swish
            - From Google, performs really well
            - Performs well mostly with very deep networks (40+ layers)
        - Maxout
            - Output the max of the inputs
            - Technically ReLU is a special case of maxout
            - Doubles the parameters that need to be trained, not often practical
    - Softmax:
        - Used on the final output layer of a multiple classification problem
        - Basically converts outputs to probabilities of each classification
        - Can't produce more than one label for something (sigmoid can)
- Choosing an activation function:
    - For multiple classification, we should use softmax on the output layer
    - RNNs do well with Tanh
    - For everything else
        - Start with ReLU
        - If we need to do better, we should try Leaky ReLU
        - Last resort: PReLU, Maxout
        - Swish for really deep networks

## Convolutional Neural Networks (CNN)

- They are used mostly for image analysis
- Recommended when we have data that doesn't neatly align into columns. Examples:
    - Images that we want to find features within
    - Machine translation
    - Sentence classification
    - Sentiment analysis
- They can find features that aren't in a specific spot, examples:
    - Stop sing in a picture
    - Words within a sentences
- They are "feature-location invariant"
- How dod they work:
    - Local receptive fields are groups of neurons that only respond to a part of what is seen (subsampling)
    - They over overlap each other to cover the entire visual field  (convolution)
    - They feed into higher layers that identify increasing complex images
    - For color images we can used extra layers for red, green, and blue channels
- Building a CNN with Keras:
    - Source data must be of appropriate dimensions
    - Conv2D layer types does the actual convolution on a 2D image
    - MaxPooling2D layers can be used to reduce a 2D layer down by taking the maximum value in a given block
    - Flatten layers will convert the 2D layer to a 1D layer for passing into a flat hidden layer of neurons
    - Typical usage:
        - Conv2D -> MaxPooling -> Dropout -> Flatten -> Dense -> Dropout -> Softmax
- CNNs are very computationally intensive (CPU, GPU, adn RAM)
- CNNs have a lot of hyperparameters to configure (kernel sizes, layers with different number of units, amount of pooling, etc.)
- They are specialized architectures of CNNs:
    - LeNet-5: handwriting recognition
    - AlexNet: image classification
    - GoogLeNet: deeper than AlexNet, introduces inception modules (groups of convolutional layers)
    - ResNet (Residual Network): even deeper - maintains performance via skip connections

## Recurrent Neural Networks (RNNs)

- They are used for:
    - Time-series data:
        - Predict future behavior based on past behavior
        - Analyze web logs, sensor logs, stock trades
        - Self-driving based on past trajectories
    - Data that consists of sequences of arbitrary lengths:
        - Machine translation
        - Produce image captions
        - Produce machine-generated music
- RNN topologies:
    - Sequence to sequence: predict stock prices based on series of historical data
    - Sequence to vector: words in a sentence to sentiment
    - Vector to sequence: create captions from an image
    - Encodes -> Decoder: machine translation
- Training RNNs:
    - We need to backpropagate not only through the layers of the network but also through time
    - All these time steps adds up fast. To avoid this we can limit the number of time steps (truncated backpropagation through time)
    - The state from an earlier time step gets diluted over time. This can be a problem if the older behavior does not matter less than the newer (example: sentence of words). To counteract this effect we can do the following:
        - LSTM Cell:
            - Long Short-Term Memory Cell
            - Maintains a separate short-term and long-term state
        - GRU Cell:
            - Gated Recurrent Unit
            - Simplified LSTM Cell that performs similarly
- Training an RNN is hard:
    - They are very sensitive to topologies and choice of hyperparameters
    - Training is very resource intensive
    - A wrong choice can lead to RNN that doesn't converge at all

## Modern Natural Language Processing

- Transform deep learning architectures:
    - Adopts mechanism of "self-attention":
        - Weights significance of each part of the input data
        - Processes sequential data but processes the entire input at once
        - The attention mechanism provides context, so no need to process one word at a time
    - Models: BERT, RoBERTa, T5, GPT-2/3/4, DistilBERT
        - DistilBERT: uses knowledge distillation to reduce model size by 40%
        - BERT: Bi-directional Encoder Representation from Transformers
        - GPT: Generative Pre-Trained Transformer
- Transfer Learning - take pre-trained models and use them for own purposes
    - NLP models (and others) are too big and complex to build from scratch and re-train every time
    - Model zoos such as Hugging Face offer pre-trained models to start from
        - Hugging has an integration with Sagemaker via Hugging Face Deep Learning Containers
    - We can fine-tune these models for our own use cases
    - Transfer Learning approaches:
        - Continue training a pre-trained model - fine-tuning: use a low learning rate to ensure we are just incrementally improving the model
        - Add new trainable layers to the top of a frozen model: turn old features into predictions of new data
        - Retrain from scratch - in case we have a large amount of data and compute capacity
        - Use it as-is

## Deep Learning on EC2/EMR

- EMR supports Apache MXNet and GPU instance types
- Appropriate instance types for deep learning:
    - P3: 8 Tesla V100 GPUs
    - P2: 16 K80 GPUs (less expensive)
    - G3: 4 M60 GPUs (all Nvidia chips)
    - G5g: AWS Graviton 2 processors / Nvidia T4G Tensor GPUs
    - P4d: A100 "UltraClusters" for supercomputing
- Deep Learning AMIs

## Tuning Neural Networks

- Neural networks are trained by gradient descent (or similar means)
- We start at some random point and sample different solutions (weights) seeking to minimize some cost function over many epochs
- Learning rate: how far apart the samples are
- Effects of learning rate:
    - If the learning rate is too high, we can overshoot the optimal solution
    - If the learning rate is too small, it will take too long to find the optimal solution
    - Learning rate is an example of hyperparameter
- Batch size - how many training samples are used within each batch of each epoch
- Somewhat counter-intuitively:
    - Smaller batch sizes can work their way out of a "local minima" more easily
    - Batch sizes that are too large can end up getting stuck in the wrong solution
    - Random shuffling at each epoch can make this look like very inconsistent results from run to run