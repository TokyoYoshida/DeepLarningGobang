## DeeplarningGobang

DeepLarningGobang is an algorithm of Gobang using deep learning and Monte Carlo tree search.

## Run for Training

To Training, Change as follows part of the board.py.
```
TENSORFLOW_MODE = TENSORFLOW_TRAINMODE
```

And run the following command.
```
python board.py
```
The results of the training will be saved in the model.ckpt.

NOTE:Board is a 3x3 in the initial state. If you want to change, please change the following values of board.py.
```
BOARD_ROW_SIZE = <number of row>
BOARD_COL_SIZE = <number of column>
```

## Run for play the game

To play game, Change as follows part of the board.py.
```
TENSORFLOW_MODE = TENSORFLOW_FIHGTMODE
```
And run the following command.
```
python board.py
```

In Fight mode, board.py acts as a Web server. Therefore, when your turn, you must be a request to the Web server. (You are always first move)

To access the Web server, use a script "put_to_board".

```
put_to_board <number on board>
```
<number on board> is using the serial number from the upper left.(see below)

|0  |1  |2  |
|---|---|---|
|3  |4  |5  |
|6  |7  |8  |

## Contribution

Contribution is welcome!

If there is contact, please write to the Issues. Or, please mail.
yoshidaforpublic@gmail.com

## License

 MIT
