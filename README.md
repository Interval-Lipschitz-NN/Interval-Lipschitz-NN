# Interval Lipschitz NN - UAI 2021

## Environment
```Ubuntu 20.04 LTS```

## Dependency
- art (adversarial-robustness-toolbox)
- MPFI (boost library in cpp)
- matplotlib 3.3.3
- numpy 1.19.4
- pybind11 2.6.1
- Python 3.8.5
- pytorch 1.7.0

## Running Code
``` python3 main.py```

## Experiments Corresponding to the Tables and Graphs
**All the details are in the main() function which is in main.py**
- Experimental results in the paper's tables

| Table | Experiment Index | Dataset | Activation Function | Radius |          Comments          |
|:-----:|:----------------:|:-------:|:-------------------:|:------:|:--------------------------:|
|   1   |         0        |   IRIS  |       Sigmoid       |  0.001 |                            |
|       |         1        |   IRIS  |       Sigmoid       |  0.002 |                            |
|       |         2        |     Balance-scale    |       Sigmoid       |  0.001 |                            |
|       |         3        |     Balance-scale    |       Sigmoid       |  0.002 |                            |
|   2   |         4        |   IRIS  |         ReLU        |  0.001 |                            |
|       |         5        |   IRIS  |         ReLU        |  0.002 |                            |
|       |         6        |  MNIST  |         ReLU        |  0.001 |                            |
|       |         7        |  MNIST  |         ReLU        |  0.002 |                            |
|   3   |         8        |   NULL  |         ReLU        |  0.001 | Inputs near the boundaries |
|       |         9        |   NULL  |         ReLU        |  0.002 | Inputs near the boundaries |

- PDE experiments

| Graph Index | Experiment Index | Center Point | Radius |     Inputs Domain     |
|:-----------:|:----------------:|:------------:|:------:|:---------------------:|
|      1      |        10        |  (0.5, 0.5)  |   0.5  |     [0, 1]X[0, 1]     |
|      2      |        11        |  (0.5, 0.5)  |   0.1  | [0.4, 0.6]X[0.4, 0.6] |
|      3      |        12        |  (0.3, 0.3)  |   0.1  | [0.2, 0.4]X[0.2, 0.4] |
|      4      |        13        |  (0.7, 0.7)  |   0.1  | [0.6, 0.8]X[0.6, 0.8] |
|      5      |        14        |  (0.1, 0.1)  |   0.1  |   [0, 0.2]X[0, 0.2]   |
|      6      |        15        |  (0.9, 0.9)  |   0.1  |   [0.8, 1]X[0.8, 1]   |

## Code Structure
```
INTERVAL-LIPSCHITZ-NN
|
|--- CLEVER_python
|   |
|   |--- CLEVER_Lipschitz.py
|   |--- CLEVER.py
|
|--- data
|   |
|   |--- MNIST
|
|--- Interval_cpp
|   |
|   |--- myInterval_PDE.cpp
|   |--- myInterval_ReLU.cpp
|   |--- myInterval_Sigmoid.cpp
|   |--- myUtility.cpp
|   |--- muUtility.h
|
|--- models
|   |
|   |--- IRIS
|   |--- MNIST
|
|--- parameters
|
|--- PDE_IsolatedBoxes
|
|--- tmp
|
|--- utilities
|   |
|   |--- main_utility.py
|
|--- compile_interval_cpp.sh
|
|--- IntervalCPP_PDE.cpython-38-x86_64-linux-gnu.so
|
|--- IntervalCPP_ReLU.cpython-38-x86_64-linux-gnu.so
|
|--- IntervalCPP_Sigmoid.cpython-38-x86_64-linux-gnu.so
|
|--- main.pu
|
|--- Nets.py

```