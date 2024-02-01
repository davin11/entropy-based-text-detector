Hi!

Below you can find a outline of how to reproduce my solution for the ''LLM - Detect AI Generated Text'' competition.
If you run into any trouble with the setup/code or have any questions please contact me at davide.cozzolino@unina.it

# ARCHIVE CONTENTS
- train.py   : code to train the one-class svm.
- predict.py : code to generate predictions.
- LICENSE    : license.

# HARDWARE:
- Ubuntu 24GB
- NVIDIA Tesla P100

# SOFTWARE
- Python 3.10
- nvidia drivers 535
- CUDA 12.2
- python packages are detailed in `requirements.txt`

# CODE
Download the files of competition `train_essays.csv` and `test_essays.csv` in this folder.

Run this script for training:
```
python train.py train_essays.csv
```

Run this script to generate predictions:
```
python predict.py test_essays.csv
```

