import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_ReLU(nn.Module):
    def __init__(self, inputNum, hiddenNum, outputNum):
        super().__init__()
        self.fc1 = nn.Linear(in_features=inputNum, out_features=hiddenNum)
        self.out = nn.Linear(in_features=hiddenNum, out_features=outputNum)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        return self.out(x)

class Net_Sigmoid(nn.Module):
    def __init__(self, inputNum, hiddenNum, outputNum):
        super().__init__()
        self.fc1 = nn.Linear(in_features=inputNum, out_features=hiddenNum)
        self.out = nn.Linear(in_features=hiddenNum, out_features=outputNum)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x.float()))
        return self.out(x)

class Net_PDE(nn.Module):
    def __init__(self, inputNum=2, hiddenNum=20, outputNum=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features=inputNum, out_features=hiddenNum)
        self.out = nn.Linear(in_features=hiddenNum, out_features=outputNum)
        w_hidden = torch.tensor([   [-0.791636392274, 0.405410650556], [2.104037571710, -1.817878593885], [0.055500361520, 2.794980642297], [0.144256943918, -0.994731363003], [2.266245472703, -1.708296224438], 
                                    [1.896290365660, 1.710785134293], [0.166597680850, 1.137834783101], [-0.200003130639, -0.318788021016], [2.097059284415, 1.884763678425], [-1.957432365036, -0.073871941684],
                                    [-1.555312333591, 0.059218632455], [-0.182528434323, -1.891391398822], [-0.029359672863, -0.110670390187], [1.068817713914, 0.007719234695], [0.381601690513, 0.658322241235],
                                    [1.387332987922, 0.920062523632], [0.052434142832, -1.232722510619], [0.392710825626, 1.474891543585], [0.038110089712, -0.799215386015], [-2.074021465439, 0.104573034195]])
        w_output = torch.tensor([[-0.225601187717, -1.273670107566, 0.527405768474, -0.476809749441, 1.280144720456, -1.766647941691, 1.542681844839, -1.181405598440, 1.227534443335, -1.652318673178, 
                                 0.578430515632, 2.158010829121, -0.664516434970, 1.072369935945, 0.512433028906, 0.312958846522, 1.589953403409, -1.236633262437, -1.180208819396, 2.021434197206]])
        bias = torch.tensor([0.860345365133, -1.862383963561, 1.183216285688, 0.555739674189, 1.461160134132, -3.408190429734, -1.599040868163, 0.255319786598, -0.182093264211, 2.408751516229, 
                            -0.623591098159, -0.331333584479, -1.327780994325, -1.187648291875, -0.970405217782, -0.443909583168, -1.805230662087, 1.537074706816, 1.122540156639, -0.389472303950])

        self.fc1.weight.data = w_hidden
        self.fc1.bias.data = bias
        self.out.weight.data = w_output
        self.out.bias.data = torch.tensor([0.0])

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x.float()))
        return self.out(x)