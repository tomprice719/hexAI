# hexAI

An AI that plays [the game of hex](https://en.wikipedia.org/wiki/Hex_(board_game)). It uses 2-ply minimax with a heuristic function computed by a deep convolutional neural network.

To play a game:

pip3 install requirements.txt  
python3 -m hexAI.minimax_player

Most of the code used for training the model is also included. There is no single function that will replicate the training process used for the included model, since it was trained using an ad-hoc script that I fiddled with during training, which called functions in training_utils. However, it's pretty simple to make a training script in terms of those functions.
