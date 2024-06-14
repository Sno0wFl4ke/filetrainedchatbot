# FTCB (FileTrainedChatBot)
FTCB is a chatbot that is trained on a file. <br>
It is a simple chatbot that can be trained on any file and can answer questions based on the content of the file. <br>It uses the `nltk` library for natural language processing and `numpy` for numerical operations. Using `pytorch` for training the model. <br>

## Installation

To make this project work, you need to install the following libraries:
- `torch`
- `nltk`
- `numpy`

You can install these libraries using the following command:
```bash
pip3 install torch
pip3 install nltk
pip3 install numpy
```

After you have installed all libraries, you need to configure the `intents.json` and `profile.json` files.
- The `intents.json` file contains the training data for the chatbot. 
- The `profile.json` file contains the profile of the chatbot. 

## Training the chatbot
Run the train.py file in the /build directory using the following command:
```bash
> python3 train.py
```
Wait until training is finished. <br>
Check if training loss is less than ``0.01``. If not rerun! <br>
All set. 


Run the chatbot using the following command in /ui directory:
```bash
> python3 chat.py
```
