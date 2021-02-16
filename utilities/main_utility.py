import os
import torch
import torch.nn as nn
from torchvision import datasets
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from art.estimators.classification import PyTorchClassifier

from Nets import Net_ReLU, Net_Sigmoid

# Function 01: loadData
def loadData(index, isOwnDataset, randomSeed):
    # IRIS dataset -- inputs: 4d; outputs: 3d
    if index==0:
        print("Using IRIS dataset.")
        iris = load_iris()
        X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, shuffle=True, random_state=randomSeed)
        inputNum = 4
        outputNum = 3

    # Digits dataset -- inputs: 64d; outputs: 10d
    elif index == 1:
        print("Using 64d digits dataset.")
        digits = load_digits()
        X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.2, shuffle=True)
        inputNum = 64
        outputNum = 10

    # MNIST dataset -- inputs: 28*28 = 784d; outputs: 10d
    elif index == 2:
        print("Using MNIST dataset.")
        data_train = datasets.MNIST(root = "./data/",
                            train = True,
                            download = True)

        data_test = datasets.MNIST(root="./data/",
                            train = False)
        
        X_train = data_train.data.numpy()
        X_train = X_train.reshape(X_train.shape[0],-1)

        X_test = data_test.data.numpy()
        X_test = X_test.reshape(X_test.shape[0],-1)

        Y_train = data_train.targets.numpy()
        Y_test = data_test.targets.numpy()

        inputNum = 28*28
        outputNum = 10

    # Diabetes -- inputs: 8d; outputs: 2d
    elif index == 3:
        data=pd.read_csv('./data/diabetes.csv')
        X=data.drop(['Outcome'], axis=1)
        y=data['Outcome']
        X_train, X_test, Y_train, Y_test= train_test_split(X,y, test_size=0.2, random_state=10)
        X_train = X_train.values.tolist()
        X_test = X_test.values.tolist()
        Y_train = Y_train.values.tolist()
        Y_test = Y_test.values.tolist()

        inputNum = 8
        outputNum = 2

    # Balance-scale -- inputs: 4d; outputs: 3d
    elif index == 4:
        data=pd.read_csv('./data/balance-scale.csv')
        X=data.drop(['Class'], axis=1)
        Y=data['Class']
        y = []
        for i in Y:
            if i == 'L':
                y.append(0)
            elif i == 'B':
                y.append(1)
            else:
                y.append(2)
        X_train, X_test, Y_train, Y_test= train_test_split(X,y, test_size=0.2, random_state=10)
        X_train = X_train.values.tolist()
        X_test = X_test.values.tolist()

        inputNum = 4
        outputNum = 3

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    Y_train = torch.IntTensor(Y_train)
    Y_test = torch.IntTensor(Y_test )
    return X_train, X_test, Y_train, Y_test, inputNum, outputNum


# Function 02: initNN
def initNN(actFunction, trainedNN, modelName, batchNum, epochNum, X_train, X_test, Y_train, Y_test, inputNum, hiddenNum, outputNum):

    # Init
    if actFunction == "ReLU":
        model = Net_ReLU(inputNum, hiddenNum, outputNum)
    else:
        model = Net_Sigmoid(inputNum, hiddenNum, outputNum)

    if trainedNN:
        path = "models/" + modelName
        model.load_state_dict(torch.load(path))

    # Create classifier by using art library
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, inputNum),
        nb_classes=outputNum,
    )

    # Train the NN
    if not trainedNN:
        classifier.fit(X_train, Y_train, batch_size=batchNum, nb_epochs=epochNum)

    # Get predictions
    predictions = classifier.predict(X_test)
    myPred = []
    for i in range(len(predictions)):
        myPred.append(predictions[i].argmax())

    # Calculate accuracy
    df = pd.DataFrame({'Y': Y_test, 'YHat': myPred})
    df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
    print("The accuracy is: ")
    print(df['Correct'].sum() / len(df))

    return classifier, myPred, model


# Function 03: storeParameters
def storeParameters(model, lipschitz, clever_x_test, clever_y_test, index):
    # Convert parameters to numpy
    W1 = model.fc1.weight.data.numpy()
    b1 = model.fc1.bias.data.numpy()
    W2 = model.out.weight.data.numpy()

    # Saving as csv
    path = "parameters/E" + str(index);
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(path+"/X0.csv", clever_x_test, delimiter=",")
    np.savetxt(path+"/Pred_X0.csv", np.array([clever_y_test]), delimiter=",")
    np.savetxt(path+"/W1.csv", W1, delimiter=",")
    np.savetxt(path+"/W2.csv", W2, delimiter=",")
    np.savetxt(path+"/b1.csv", [b1], delimiter=",")
    np.savetxt(path+"/CLEVER.csv", np.array([lipschitz]), delimiter=",")

# Function 04: plot isolated boxes
def plotIsolatedBoxes(file_name, c0, radius, experIndex):

    # Read the data from the file
    with open(file_name, 'r') as boxes:
        data = boxes.readlines()

        X = []
        Y = []
        for line in data:
            tmp = line.split("'")
            tmp0 = tmp[1].split('(')[1].split(')')
            tmp0 = tmp0[0].split(',')
            tmp01 = float(tmp0[0])
            tmp02 = float(tmp0[1])
            tmp00 = (tmp01 + tmp02)/2

            tmp1 = tmp[3].split('(')[1].split(')')
            tmp1 = tmp1[0].split(',')
            tmp11 = float(tmp1[0])
            tmp12 = float(tmp1[1])
            tmp10 = (tmp11 + tmp12)/2

            X.append(tmp00)
            Y.append(tmp10)
    
    # Draw the box lines
    X0 = np.linspace(0, 1, 1000)
    Y0 = np.linspace(0, 0, 1000)
    Y1 = np.linspace(1, 1, 1000)

    Y2 = np.linspace(0, 1, 1000)
    X1 = np.linspace(0, 0, 1000)
    X2 = np.linspace(1, 1, 1000)

    # Draw the domain lines
    c1 = c0[0]
    c2 = c0[1]
    r = radius
    X01 = np.linspace(c1-r, c1+r, 1000)
    Y01 = np.linspace(c2-r, c2-r, 1000)
    Y11 = np.linspace(c2+r, c2+r, 1000)
    Y21 = np.linspace(c2-r, c2+r, 1000)
    X11 = np.linspace(c1-r, c1-r, 1000)
    X21 = np.linspace(c1+r, c1+r, 1000)

    # Plot the graph
    plt.gca().set_aspect('equal')
    p1 = plt.scatter(X0, Y0, color = 'black',s = 0.1)
    p1 = plt.scatter(X0, Y1, color = 'black',s = 0.1)
    p1 = plt.scatter(X1, Y2, color = 'black',s = 0.1)
    p1 = plt.scatter(X2, Y2, color = 'black',s = 0.1)
    p1 = plt.scatter(X01, Y01, color = 'dodgerblue',s = 0.1)
    p1 = plt.scatter(X01, Y11, color = 'dodgerblue',s = 0.1)
    p1 = plt.scatter(X11, Y21, color = 'dodgerblue',s = 0.1)
    p1 = plt.scatter(X21, Y21, color = 'dodgerblue',s = 0.1)
    p1 = plt.scatter([0.5], [0.5], marker = 'o', color = 'black', s = 10)
    p1 = plt.scatter([c1], [c2], marker = 'o', color = 'limegreen', s = 10)
    p1 = plt.scatter(X, Y, marker = 'x', color = 'r', s = 0.1)

    # Save praph
    path = "PDE_IsolatedBoxes/E" + str(experIndex) + "/" + "E" + str(experIndex) + ".png"
    plt.savefig(path)