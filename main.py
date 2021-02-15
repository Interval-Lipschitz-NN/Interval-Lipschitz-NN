import numpy as np
import os
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier

from utilities.main_utility import loadData, initNN, storeParameters, plotIsolatedBoxes
from CLEVER_python.CLEVER import clever_PDE
from CLEVER_python.CLEVER_Lipschitz import CLEVER_Lipschitz_Default_Dataset, CLEVER_Lipschitz_Special
import IntervalCPP_ReLU, IntervalCPP_Sigmoid, IntervalCPP_PDE
from Nets import Net_PDE


def experiment(experIndex, datasetIndex, actFunction, trainedNN, modelName, batchNum, epochNum, hiddenNum, comparedToCLEVER, saveModel, saveModelName, randomSeed, linf_radius):
    """
    :param experIndex: the index of experiment,
    :param datasetIndex: the index of dataset,
    :param actFunction: the chosen activation function,
    :param trainedNN: whether uses the trained NN,
    :param modelName: the modleName of trained NN,
    :param batchNum: the batch number when trains the NN,
    :param epochNum: the epoch number when trains the NN,
    :param hiddenNum: the neuro number in hidden layer,
    :param comparedToCLEVER: whether compared to the CLEVER result,
    :param saveModel: whether save model,
    :param saveModelName: if save model, the name of the model,
    :param randomSeed: the random seed used in seprating the dataset,
    :param linf_radius: the radius of perturbation.
    """
    # Load Dataset
    isOwnDataset = 10
    X_train, X_test, Y_train, Y_test, inputNum, outputNum = loadData(datasetIndex, isOwnDataset, randomSeed)
    
    # Init NN
    classifier, myPred, model = initNN(actFunction, trainedNN, modelName, batchNum, epochNum, X_train, X_test, Y_train, Y_test, inputNum, hiddenNum, outputNum)
    
    # Save Model
    if saveModel:
        if datasetIndex == 0:
            categ = "IRIS"
        torch.save(model.state_dict(), 'models/' + categ + "/" + saveModelName)

    # Set Conditions
    maxIterationNum = 20
    minGap = 0.0001
    maxBoxes = 3000

    # Get CLEVER Lipschitz Constant
    Nb = 10
    Ns = 10
    pool_size = 2
    lipschitz, clever_x_test, clever_y_test = CLEVER_Lipschitz_Default_Dataset(X_test, myPred, classifier, Nb, Ns, linf_radius, pool_size)

    # Store Parameters
    storeParameters(model, lipschitz, clever_x_test, clever_y_test, experIndex)

    # Get Interval Lipschitz Constant
    if actFunction == "ReLU":
        IntervalCPP_ReLU.get_interval_Lipschitz_CPP(str(experIndex), linf_radius, inputNum, hiddenNum, outputNum, comparedToCLEVER, maxIterationNum, minGap, maxBoxes)
    elif actFunction == "Sigmoid":
        IntervalCPP_Sigmoid.get_interval_Lipschitz_CPP(str(experIndex), linf_radius, inputNum, hiddenNum, outputNum, comparedToCLEVER, maxIterationNum, minGap, maxBoxes)



def special_experiment_ReLU(experIndex, trainedNN, modelName, batchNum, epochNum, hiddenNum, comparedToCLEVER, saveModel, saveModelName, randomSeed, linf_radius):
    """
    :param experIndex: the index of experiment,
    :param trainedNN: whether uses the trained NN,
    :param modelName: the modleName of trained NN,
    :param batchNum: the batch number when trains the NN,
    :param epochNum: the epoch number when trains the NN,
    :param hiddenNum: the neuro number in hidden layer,
    :param comparedToCLEVER: whether compared to the CLEVER result,
    :param saveModel: whether save model,
    :param saveModelName: if save model, the name of the model,
    :param randomSeed: the random seed used in seprating the dataset,
    :param linf_radius: the radius of perturbation.
    """
    # Load Dataset
    isOwnDataset = 10
    datasetIndex = 0
    X_train, X_test, Y_train, Y_test, inputNum, outputNum = loadData(datasetIndex, isOwnDataset, randomSeed)
    
    # Init NN
    actFunction = "ReLU"
    classifier, myPred, model = initNN(actFunction, trainedNN, modelName, batchNum, epochNum, X_train, X_test, Y_train, Y_test, inputNum, hiddenNum, outputNum)
    
    # Save Model
    if saveModel:
        if datasetIndex == 0:
            categ = "IRIS"
        torch.save(model.state_dict(), 'models/' + categ + "/" + saveModelName)

    # Set Conditions
    maxIterationNum = 10
    minGap = 0.0001
    maxBoxes = 3000

    # Get Special Inputs X
    weight = model.fc1.weight.detach().numpy()
    bias = model.fc1.bias.detach().numpy()
    len = 4
    A = np.zeros((len, len))
    B = np.zeros(len)
    bias = -1 * bias
    
    for i in range(len):
        B[i] = bias[i]
        for j in range(len):
            A[i][j] = weight[i][j]
    
    x = np.linalg.solve(A, B)
    x_special = np.zeros(4)

    for i in range(x.shape[0]):
        x_special[i] = x[i]

    # Get CLEVER Lipschitz Constant
    Nb = 10
    Ns = 10
    pool_size = 2
    lipschitz, clever_x_test, clever_y_test = CLEVER_Lipschitz_Special(x_special, myPred, classifier, Nb, Ns, linf_radius, pool_size)

    # Store Parameters
    storeParameters(model, lipschitz, clever_x_test, clever_y_test, experIndex)

    # Get Interval Lipschitz Constant
    IntervalCPP_ReLU.get_interval_Lipschitz_CPP(str(experIndex), linf_radius, inputNum, hiddenNum, outputNum, comparedToCLEVER, maxIterationNum, minGap, maxBoxes)
    


def experiment_PDE(experIndex, x0, linf_radius, comparedToCLEVER, file_name, max_boxes, maxIterationNum):

    # Get Model
    model = Net_PDE()
    x1 = torch.tensor(x0)
    y1 = model(x1)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    classifier = PyTorchClassifier(
            model=model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=(1, 2),
            nb_classes=1,
        )


    # Get CLEVER Lipschitz Constant
    Nb = 10
    Ns = 10
    pool_size = 2
    x2 = np.array(x0)
    lipschitz = clever_PDE(classifier, x2, Nb, Ns, linf_radius, norm=np.inf, pool_factor=pool_size)
    print("The Lipschitz constant in CLEVER is " + str(lipschitz))


    # Store Parameters 
    storeParameters(model, lipschitz, [x2], 0, experIndex)
    inputs_size = 2 
    hidden_neural_num = 20 
    output_size = 1
    minGap = 0.0001
    tuple_result = IntervalCPP_PDE.get_interval_Lipschitz_CPP(str(experIndex), linf_radius, inputs_size, hidden_neural_num, output_size, comparedToCLEVER, maxIterationNum, minGap, max_boxes)

    # Isolated boxes
    boxes = tuple_result[0]
    path = "PDE_IsolatedBoxes/E" + str(experIndex)
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + "/" + file_name
    file= open(path,"w+")
    for i in range(len(boxes)):
        file.writelines(str(boxes[i])+"\n")
    file.close()

    plotIsolatedBoxes(path, x0, linf_radius, experIndex)



def main():
    
    ################################################
    # Table 1
    ################################################
    """
    Table 1 | Experiment 00
    Dataset: IRIS | Activation Function: Sigmoid
    """
    # experIndex = 0
    # datasetIndex = 0
    # actFunction = "Sigmoid"
    # trainedNN = True
    # modelName = "IRIS/IRIS00.pt"
    # batchNum = 30
    # epochNum = 100
    # hiddenNum = 10
    # comparedToCLEVER = False
    # saveModel = False
    # saveModelName = "IRIS00.pt"
    # randomSeed = 1
    # linf_radius = 0.001
    # experiment(experIndex, datasetIndex, actFunction, trainedNN, modelName, batchNum, epochNum, 
    #            hiddenNum, comparedToCLEVER, saveModel, saveModelName, randomSeed, linf_radius)

    
    """
    Table 1 | Experiment 01
    Dataset: IRIS | Activation Function: Sigmoid
    """
    # experIndex = 1
    # datasetIndex = 0
    # actFunction = "Sigmoid"
    # trainedNN = True
    # modelName = "IRIS/IRIS01.pt"
    # batchNum = 30
    # epochNum = 100
    # hiddenNum = 10
    # comparedToCLEVER = False
    # saveModel = False
    # saveModelName = "IRIS01.pt"
    # randomSeed = 1
    # linf_radius = 0.002
    # experiment(experIndex, datasetIndex, actFunction, trainedNN, modelName, batchNum, epochNum, 
    #            hiddenNum, comparedToCLEVER, saveModel, saveModelName, randomSeed, linf_radius)


    """
    Table 1 | Experiment 02
    Dataset:  | Activation Function: Sigmoid
    """

    """
    Table 1 | Experiment 03
    Dataset:  | Activation Function: Sigmoid
    """

    ################################################
    # Table 2
    ################################################
    """
    Table 2 | Experiment 04
    Dataset: IRIS | Activation Function: ReLU
    """
    # experIndex = 4
    # datasetIndex = 0
    # actFunction = "ReLU"
    # trainedNN = True
    # modelName = "IRIS/IRIS02.pt"
    # batchNum = 30
    # epochNum = 100
    # hiddenNum = 10
    # comparedToCLEVER = False
    # saveModel = False
    # saveModelName = "IRIS02.pt"
    # randomSeed = 1
    # linf_radius = 0.001
    # experiment(experIndex, datasetIndex, actFunction, trainedNN, modelName, batchNum, epochNum, 
    #            hiddenNum, comparedToCLEVER, saveModel, saveModelName, randomSeed, linf_radius)

    
    """
    Table 2 | Experiment 05
    Dataset: IRIS | Activation Function: ReLU
    """
    # experIndex = 5
    # datasetIndex = 0
    # actFunction = "ReLU"
    # trainedNN = True
    # modelName = "IRIS/IRIS03.pt"
    # batchNum = 30
    # epochNum = 100
    # hiddenNum = 10
    # comparedToCLEVER = False
    # saveModel = False
    # saveModelName = "IRIS03.pt"
    # randomSeed = 1
    # linf_radius = 0.002
    # experiment(experIndex, datasetIndex, actFunction, trainedNN, modelName, batchNum, epochNum, 
    #            hiddenNum, comparedToCLEVER, saveModel, saveModelName, randomSeed, linf_radius)

    
    """
    Table 2 | Experiment 06
    Dataset: MNIST | Activation Function: ReLU
    """
    # experIndex = 6
    # datasetIndex = 2
    # actFunction = "ReLU"
    # trainedNN = True
    # modelName = "MNIST/MNIST.pt"
    # batchNum = 30
    # epochNum = 100
    # hiddenNum = 30
    # comparedToCLEVER = False
    # saveModel = False
    # saveModelName = "MNIST.pt"
    # randomSeed = 0
    # linf_radius = 0.001
    # experiment(experIndex, datasetIndex, actFunction, trainedNN, modelName, batchNum, epochNum, 
    #            hiddenNum, comparedToCLEVER, saveModel, saveModelName, randomSeed, linf_radius)


    """
    Table 2 | Experiment 07
    Dataset: MNIST | Activation Function: ReLU
    """
    # experIndex = 7
    # datasetIndex = 2
    # actFunction = "ReLU"
    # trainedNN = True
    # modelName = "MNIST/MNIST.pt"
    # batchNum = 30
    # epochNum = 100
    # hiddenNum = 30
    # comparedToCLEVER = False
    # saveModel = False
    # saveModelName = "MNIST.pt"
    # randomSeed = 0
    # linf_radius = 0.002
    # experiment(experIndex, datasetIndex, actFunction, trainedNN, modelName, batchNum, epochNum, 
    #            hiddenNum, comparedToCLEVER, saveModel, saveModelName, randomSeed, linf_radius)


    ################################################
    # Table 3
    ################################################
    """
    Table 3 | Experiment 08
    Dataset: IRIS | Activation Function: ReLU
    """
    # experIndex = 8
    # trainedNN = True
    # modelName = "IRIS/IRIS04.pt"
    # batchNum = 30
    # epochNum = 100
    # hiddenNum = 30
    # comparedToCLEVER = False
    # saveModel = False
    # saveModelName = "IRIS04.pt"
    # randomSeed = 1
    # linf_radius = 0.001
    # special_experiment_ReLU(experIndex, trainedNN, modelName, batchNum, epochNum, hiddenNum, 
    #                         comparedToCLEVER, saveModel, saveModelName, randomSeed, linf_radius)

    
    """
    Table 3 | Experiment 09
    Dataset: IRIS | Activation Function: ReLU
    """
    # experIndex = 9
    # trainedNN = True
    # modelName = "IRIS/IRIS05.pt"
    # batchNum = 30
    # epochNum = 100
    # hiddenNum = 30
    # comparedToCLEVER = False
    # saveModel = False
    # saveModelName = "IRIS05.pt"
    # randomSeed = 1
    # linf_radius = 0.002
    # special_experiment_ReLU(experIndex, trainedNN, modelName, batchNum, epochNum, hiddenNum, 
    #                         comparedToCLEVER, saveModel, saveModelName, randomSeed, linf_radius)


    ################################################
    # PDE
    ################################################
    """
    Graph 01 | Experiment 10
    Center Point: [0.5, 0.5] | Radius: 0.5
    """
    # experIndex = 10
    # x0 = [0.5, 0.5]
    # linf_radius = 0.5
    # comparedToCLEVER = False
    # file_name = "Isolated_Boxes.txt"
    # max_boxes = 30000
    # maxIterationNum = 10
    # experiment_PDE(experIndex, x0, linf_radius, comparedToCLEVER, file_name, max_boxes, maxIterationNum)

    """
    Graph 02 | Experiment 11
    Center Point: [0.5, 0.5] | Radius: 0.1
    """
    # experIndex = 11
    # x0 = [0.5, 0.5]
    # linf_radius = 0.1
    # comparedToCLEVER = False
    # file_name = "Isolated_Boxes.txt"
    # max_boxes = 30000
    # maxIterationNum = 10
    # experiment_PDE(experIndex, x0, linf_radius, comparedToCLEVER, file_name, max_boxes, maxIterationNum)

    """
    Graph 03 | Experiment 12
    Center Point: [0.3, 0.3] | Radius: 0.1
    """
    experIndex = 12
    x0 = [0.3, 0.3]
    linf_radius = 0.1
    comparedToCLEVER = False
    file_name = "Isolated_Boxes.txt"
    max_boxes = 30000
    maxIterationNum = 20
    experiment_PDE(experIndex, x0, linf_radius, comparedToCLEVER, file_name, max_boxes, maxIterationNum)


if __name__ == '__main__':
    main()
    