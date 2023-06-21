import torch.nn as nn
import torch
from torch.nn import functional as F

class GatedAttention(nn.Module):
    def __init__(self,class_num):
        super(GatedAttention, self).__init__()
        self.L = 1536
        self.D = 128
        self.K = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, class_num),
        )

    def forward(self, x):
        x = x.squeeze(0)

        x= torch.reshape(x,(-1,1536))
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, x)  # KxL

        Y_prob = self.classifier(M)


        return Y_prob, M,A
