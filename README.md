# hexAI

An AI that plays [the game of hex](https://en.wikipedia.org/wiki/Hex_(board_game)).

To play a game:

pip3 install requirements.txt  
python3 -m hexAI.minimax_player

The code used for training the model is also included. There is no single function that will create another model similar to the default one, as this was trained using an ad-hoc script that I fiddled with during training. However, it's pretty simple to express a training script in terms of the utilities in training_utils.
