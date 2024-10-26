# FTCB (FileTrainedChatBot)

**FTCB** is a chatbot trained on structured data in `intents.json`. Although it does not read or interpret general file content directly, it leverages training data to respond accurately. Using `nltk` for natural language processing, `numpy` for numerical operations, and `pytorch` for model training, FTCB is built for efficient and targeted interactions.

## Installation

To install FTCB, you’ll need the following libraries:
- `torch`
- `nltk`
- `numpy`

Install these packages with the following commands:

```bash
pip3 install torch
pip3 install nltk
pip3 install numpy
```

After installation, configure the following files:

- `intents.json`: This file holds the training data that FTCB uses for learning.
- `profile.json`: Contains configuration settings for the chatbot’s profile.

### Training the Chatbot

In the /build directory, run the training script:

```bash
> python3 train.py
```

Monitor the training process to ensure the loss is below 0.01. If it isn’t, rerun the training to improve accuracy.

### Running the Chatbot

To start the chatbot, execute the following command in the /ui directory:

```bash
> python3 chat.py
```

FTCB is now ready to respond based on the training data provided in intents.json.

