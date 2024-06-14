# FTCB (FileTrainedChatBot)
FTCB is a chatbot that is trained on a file. <br>
It is a simple chatbot that can be trained on any file and can answer questions based on the content of the file. <br>It uses the `nltk` library for natural language processing and `numpy` for numerical operations. Using `pytorch` for training the model. <br>

## Installation
```bash
pip3 install torch nltk numpy
> configure intents.json and profile.json
> run train.py
```

Wait until training is finished. <br>
Check if training loss is less than ``0.01``. <br>

Run the chatbot using the following command:
```bash
> run chat.py
```
