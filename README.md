# LSTM - audio

This repository handles training and exporting the weights of a LSTM model which emulates distorion effects and amplifier circuits. <br>

### Neural Network

The network architecture is based on ‘Real-time black-box modelling with recurrent neural networks’ by A. Wright, E.-P. Damskägg, and V. Välimäki. <br>
The used LSTM has 32 hidden layers.

### Dataset

To train the neural network we need an input and output dataset. <br>
Input dataset: clean audio signal. <br>
Output Dataset: the same input signal processed through the amplifier we want to emulate. <br>
You can find a sample dataset in the `dataset/input` and `dataset/output` folders. <br>
Feel free to experiment with your own audio files.

### References

The main reference for this project is the book "Build AI-Enhanced Audio Plugins with C++" by Matthew John Yee-King.

Book: http://www.yeeking.net/book/

Code: https://github.com/yeeking/ai-enhanced-audio-book
