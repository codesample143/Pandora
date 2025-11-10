from torch import nn
import torch.nn.functional as F
import torch
class Evaluation(nn.Module):
    # [(layersize, evalf) ...]
    def __init__(self, layers): # Prev implementation: input_size, hidden_size1, hidden_size2, hidden_size3, output_size,
        super(Evaluation, self).__init__()
        self.layers = layers
        self.fcs = nn.ModuleList()
        for layer_pair_index in range(1, len(layers)):
            last_size, _ = layers[layer_pair_index - 1]
            current_size, _ = layers[layer_pair_index]
            self.fcs.append(nn.Linear(last_size, current_size))
        # self.fc1 = nn.Linear(input_size, hidden_size1)
        # self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        # self.fc4 = nn.Linear(hidden_size3, output_size)
    def forward(self, x):
        def apply(s):
            if s == "relu":
                return F.relu
            elif s == 'sigmoid':
                return F.sigmoid
        _, initf = self.layers[0]
        x = apply(initf)(self.fcs[0](x))

        for i in range(1, len(self.fcs)):
            _, cf = self.layers[i]
            x = apply(cf)(self.fcs[i](x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.sigmoid(self.fc4(x)) #make sure we get value between 0 and 1
        return x
