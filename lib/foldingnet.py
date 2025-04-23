import torch
import torch.nn as nn
import numpy as np

from lib.pointnet import PointNetGlobalMax, get_MLP_layers, PointNetVanilla, PointwiseMLP
from lib.utils import make_box, make_sphere, make_cylinder

class ChamfersDistance3(nn.Module):
    '''
    Extensively search to compute the Chamfersdistance. No reference to external implementation Incomplete
    '''
    def forward(self, input1, input2):
        # input1, input2: BxNxK, BxMxK, K = 3
        B, N, K = input1.shape
        _, M, _ = input2.shape

        # Repeat (x,y,z) M times in a row
        input11 = input1.unsqueeze(2)           # BxNx1xK
        input11 = input11.expand(B, N, M, K)    # BxNxMxK
        # Repeat (x,y,z) N times in a column
        input22 = input2.unsqueeze(1)           # Bx1xMxK
        input22 = input22.expand(B, N, M, K)    # BxNxMxK
        # compute the distance matrix
        D = input11 - input22                   # BxNxMxK
        D = torch.norm( D, p=2, dim=3 )         # BxNxM

        dist0, _ = torch.min( D, dim=1 )        # BxM
        dist1, _ = torch.min( D, dim=2 )        # BxN

        loss = torch.mean(dist0, 1) + torch.mean(dist1, 1)  # B
        loss = torch.mean(loss)                             # 1
        return loss


class FoldingNetSingle(nn.Module):
    def __init__(self, dims):
        super(FoldingNetSingle, self).__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, X):
        return self.mlp.forward(X)


class FoldingNetVanilla(nn.Module):             # PointNetVanilla or nn.Sequential
    def __init__(self, MLP_dims, FC_dims, grid_dims, Folding1_dims,
                 Folding2_dims, MLP_doLastRelu=False):
        assert(MLP_dims[-1]==FC_dims[0])
        super(FoldingNetVanilla, self).__init__()
        # Encoder
        #   PointNet
        self.PointNet = PointNetVanilla(MLP_dims, FC_dims, MLP_doLastRelu)

        # Decoder
        #   Folding
        #     2D grid: (grid_dims(0) * grid_dims(1)) x 2
        # TODO: normalize the grid to align with the input data
        self.N = grid_dims[0] * grid_dims[1]
        u = (torch.arange(0, grid_dims[0]) / grid_dims[0] - 0.5).repeat(grid_dims[1])
        v = (torch.arange(0, grid_dims[1]) / grid_dims[1] - 0.5).expand(grid_dims[0], -1).t().reshape(-1)
        self.grid = torch.stack((u, v), 1)      # Nx2

        #     1st folding
        self.Fold1 = FoldingNetSingle(Folding1_dims)
        #     2nd folding
        self.Fold2 = FoldingNetSingle(Folding2_dims)


    def forward(self, X):
        # encoding
        f = self.PointNet.forward(X)            # BxK
        f = f.unsqueeze(1)                      # Bx1xK
        codeword = f.expand(-1, self.N, -1)     # BxNxK

        # cat 2d grid and feature
        B = codeword.shape[0]                   # extract batch size
        if not X.is_cuda:
            tmpGrid = self.grid                 # Nx2
        else:
            tmpGrid = self.grid.cuda()          # Nx2
        tmpGrid = tmpGrid.unsqueeze(0)
        tmpGrid = tmpGrid.expand(B, -1, -1)     # BxNx2

        # 1st folding
        f = torch.cat((tmpGrid, codeword), 2 )  # BxNx(K+2)
        f = self.Fold1.forward(f)               # BxNx3

        # 2nd folding
        f = torch.cat((f, codeword), 2 )        # BxNx(K+3)
        f = self.Fold2.forward(f)               # BxNx3
        return f


class FoldingNetShapes(nn.Module):
    ## add 3 shapes to choose and a learnable layer
    def __init__(self, MLP_dims, FC_dims, Folding1_dims,
                 Folding2_dims, MLP_doLastRelu=False):
        assert(MLP_dims[-1]==FC_dims[0])
        super(FoldingNetShapes, self).__init__()
        # Encoder
        #   PointNet
        self.PointNet = PointNetVanilla(MLP_dims, FC_dims, MLP_doLastRelu)

        # Decoder
        #   Folding
        self.box = make_box()  # 18 * 18 * 6 points
        self.cylinder = make_cylinder()  # same as 1944
        self.sphere = make_sphere()  # 1944 points
        self.grid = torch.Tensor(np.hstack((self.box, self.cylinder, self.sphere)))

        #     1st folding
        self.Fold1 = FoldingNetSingle(Folding1_dims)
        #     2nd folding
        self.Fold2 = FoldingNetSingle(Folding2_dims)
        self.N = 1944  # number of points needed to replicate codeword later; also points in Grid
        self.fc = nn.Linear(9, 9, True)  # geometric transformation


    def forward(self, X):
        # encoding
        f = self.PointNet.forward(X)            # BxK
        f = f.unsqueeze(1)                      # Bx1xK
        codeword = f.expand(-1, self.N, -1)     # BxNxK

        # cat 2d grid and feature
        B = codeword.shape[0]                   # extract batch size
        if not X.is_cuda:
            tmpGrid = self.grid                 # Nx9
        else:
            tmpGrid = self.grid.cuda()          # Nx9
        tmpGrid = tmpGrid.unsqueeze(0)
        tmpGrid = tmpGrid.expand(B, -1, -1)     # BxNx9
        tmpGrid = self.fc(tmpGrid)              # transform


        # 1st folding
        f = torch.cat((tmpGrid, codeword), 2)  # BxNx(K+9)
        f = self.Fold1.forward(f)               # BxNx3

        # 2nd folding
        f = torch.cat((f, codeword), 2 )        # BxNx(K+3)
        f = self.Fold2.forward(f)               # BxNx3
        return f

    
class Recon(nn.Module):
    def __init__(self, Folding1_dims, Folding2_dims):
        super(Recon, self).__init__()
        # Decoder
        #   Folding
        self.box = make_box()  # 18 * 18 * 6 points
        self.cylinder = make_cylinder()  # same as 1944
        self.sphere = make_sphere()  # 1944 points
        self.grid = torch.Tensor(np.hstack((self.box, self.cylinder, self.sphere)))

        #     1st folding
        self.Fold1 = FoldingNetSingle(Folding1_dims)
        #     2nd folding
        self.Fold2 = FoldingNetSingle(Folding2_dims)
        self.N = 1944  # number of points needed to replicate codeword later; also points in Grid
        self.fc = nn.Linear(9, 9, True)  # geometric transformation


    def forward(self, codeword):
        # cat 2d grid and feature
        codeword = codeword.transpose(1, 2)
        B = codeword.shape[0]                   # extract batch size
        if not codeword.is_cuda:
            tmpGrid = self.grid                 # Nx2
        else:
            tmpGrid = self.grid.cuda()          # Nx2
        tmpGrid = tmpGrid.unsqueeze(0)
        tmpGrid = tmpGrid.expand(B, -1, -1)     # BxNx2

        # 1st folding
        f = torch.cat((tmpGrid, codeword), 2 )  # BxNx(K+2)
        f = self.Fold1.forward(f)               # BxNx3

        # 2nd folding
        f = torch.cat((f, codeword), 2 )        # BxNx(K+3)
        f = self.Fold2.forward(f)               # BxNx3
        return f