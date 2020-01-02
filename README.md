# Temporal Differences Learning

## Introduction
The goal of this project is to reproduce the Figure 3, 4, 5 in Richard Sutton’s 1988 paper [***Learning to Predict by the Methods of Temporal Differences***](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf).

## Directory
+ **./img** - to save the output
+ **main.py** - to reproduce the experiments and generate figures directly
+ **main.ipynb** - to view the procudure step by step
```
Temporal-Differences-Learning/
├── README.md
├── img
├── main.ipynb
├── main.py
├── reference
└── requirements.txt
```

## Dependencies
+ python >= 3.7.2
+ jupyter >= 1.0.0
+ numpy >= 1.16.2
+ matplotlib >= 3.1.1

## Setup
Please ensure the following packages are already installed. A virtual environment is recommended.
+ Python (for .py)
+ Jupyter Notebook (for .ipynb)

```
$ cd Temporal-Differences-Learning/
$ pip3 install pip --upgrade
$ pip3 install -r requirements.txt
```

## Run
To view the note book:
```
$ jupyter notebook
```
To run the script:
```
$ python3 main.py
```
To run the script using data with no duplicates:
```
$ python3 main.py --unique True
```
To run the script as you like:
```
$ python3 main.py --random_seed <int> --seq_len <int> --batch_num <int> --seq_num <int> --unique <bool>
```

## Output
If everything goes well, you may see the similar results shown as below.
```
Start!
Reproduce Figure 3 in Sutton (1988)
Train of Lambda 0
Train of Lambda 0.1
Train of Lambda 0.3
Train of Lambda 0.5
Train of Lambda 0.7
Train of Lambda 0.9
Train of Lambda 1
Saving Figure 3 to img/figure_3.png

Reproduce Figure 4 in Sutton (1988)
Train of Lambda 0.0
Train of Lambda 0.3
Train of Lambda 0.8
Train of Lambda 1.0
Saving Figure 4 to img/figure_4.png

Reproduce Figure 4 in Sutton (1988)
Find Best Alpha for Each Lambda
Train of Lambda 0.0
Train of Lambda 0.1
Train of Lambda 0.2
Train of Lambda 0.3
Train of Lambda 0.4
Train of Lambda 0.5
Train of Lambda 0.6
Train of Lambda 0.7
Train of Lambda 0.8
Train of Lambda 0.9
Train of Lambda 1.0
Best Alpha 0.2 for Lambda 0.0
Best Alpha 0.2 for Lambda 0.1
Best Alpha 0.2 for Lambda 0.2
Best Alpha 0.2 for Lambda 0.3
Best Alpha 0.2 for Lambda 0.4
Best Alpha 0.15 for Lambda 0.5
Best Alpha 0.15 for Lambda 0.6
Best Alpha 0.15 for Lambda 0.7
Best Alpha 0.1 for Lambda 0.8
Best Alpha 0.1 for Lambda 0.9
Best Alpha 0.05 for Lambda 1.0
Re-Train Using Best Alpha for Each Lambda
Train of Lambda 0.0 Alpha 0.2
Train of Lambda 0.1 Alpha 0.2
Train of Lambda 0.2 Alpha 0.2
Train of Lambda 0.3 Alpha 0.2
Train of Lambda 0.4 Alpha 0.2
Train of Lambda 0.5 Alpha 0.15
Train of Lambda 0.6 Alpha 0.15
Train of Lambda 0.7 Alpha 0.15
Train of Lambda 0.8 Alpha 0.1
Train of Lambda 0.9 Alpha 0.1
Train of Lambda 1.0 Alpha 0.05
Saving Figure 5 to img/figure_5.png

Done!
```
Please find output under ***./img***.

## Authors
* **[Ning Shi](https://mrshininnnnn.github.io/)** - MrShininnnnn@gmail.com

## Reference
1. Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine learning, 3(1), 9-44.