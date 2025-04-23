import torch
import torch.nn as nn
import torch.nn.functional as Functional


def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    # init weights/bias
    nn.init.xavier_uniform_(
        li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li


def get_MLP_layers(dims, doLastRelu):
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i-1], dims[i]))
        if i == len(dims)-1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers


class PointwiseMLP(nn.Sequential):
    '''Nxdin ->Nxd1->Nxd2->...-> Nxdout'''

    def __init__(self, dims, doLastRelu=False):
        layers = get_MLP_layers(dims, doLastRelu)
        super(PointwiseMLP, self).__init__(*layers)


class GlobalPool(nn.Module):
    '''BxNxK -> BxK'''

    def __init__(self, pool_layer):
        super(GlobalPool, self).__init__()
        self.Pool = pool_layer

    def forward(self, X):
        X = X.unsqueeze(-3)  # Bx1xNxK
        X = self.Pool(X)
        X = X.squeeze(-2)
        X = X.squeeze(-2)  # BxK
        return X


class PointNetGlobalMax(nn.Sequential):
    '''BxNxdims[0] -> Bxdims[-1]'''

    def __init__(self, dims, doLastRelu=False):
        layers = [
            PointwiseMLP(dims, doLastRelu=doLastRelu),  # BxNxK
            GlobalPool(nn.AdaptiveMaxPool2d((1, dims[-1]))),  # BxK
        ]
        super(PointNetGlobalMax, self).__init__(*layers)


class PointNetGlobalAvg(nn.Sequential):
    '''BxNxdims[0] -> Bxdims[-1]'''

    def __init__(self, dims, doLastRelu=True):
        layers = [
            PointwiseMLP(dims, doLastRelu=doLastRelu),  # BxNxK
            GlobalPool(nn.AdaptiveAvgPool2d((1, dims[-1]))),  # BxK
        ]
        super(PointNetGlobalAvg, self).__init__(*layers)


class PointNetVanilla(nn.Sequential):

    def __init__(self, MLP_dims, FC_dims, MLP_doLastRelu=False):
        assert(MLP_dims[-1] == FC_dims[0])
        layers = [
            PointNetGlobalMax(MLP_dims, doLastRelu=MLP_doLastRelu),  # BxK
        ]
        layers.extend(get_MLP_layers(FC_dims, False))
        super(PointNetVanilla, self).__init__(*layers)


class PointNetTplMatch(nn.Module):
    '''this can learn, but no better than PointNetVanilla'''

    def __init__(self, MLP_dims, C_tpls, M_points):
        super(PointNetTplMatch, self).__init__()
        self.P = nn.Parameter(torch.rand(
            C_tpls, M_points, MLP_dims[0])*2-1.0)  # CxMx3
        self.G = PointNetGlobalMax(MLP_dims)

    def forward(self, X):
        Fx = self.G.forward(X)  # BxNx3 -> BxK
        Fp = self.G.forward(self.P)  # CxMx3 -> CxK
        S = torch.mm(Fx, Fp.t())  # BxC
        return S


class PairwiseDistanceMatrix(nn.Module):

    def __init__(self):
        super(PairwiseDistanceMatrix, self).__init__()

    def forward(self, X, Y):
        X2 = (X**2).sum(1).view(-1, 1)
        Y2 = (Y**2).sum(1).view(1, -1)
        D = X2 + Y2 - 2.0*torch.mm(X, Y.t())
        return D


class PointNetAttentionPool(nn.Module):

    def __init__(self, MLP_dims, Attention_dims, FC_dims, MLP_doLastRelu=False):
        assert(MLP_dims[-1]*Attention_dims[-1] == FC_dims[0])
        # assert(Attention_dims[-1]==1)
        super(PointNetAttentionPool, self).__init__()
        self.add_module(
            'F',
            PointwiseMLP(MLP_dims, doLastRelu=MLP_doLastRelu),  # BxNxK
        )
        self.S = nn.Sequential(
            PointwiseMLP(Attention_dims, doLastRelu=False),  # BxNxM
            nn.Softmax(dim=-2)  # BxNxM
        )
        self.L = nn.Sequential(*get_MLP_layers(FC_dims, False))

    def forward(self, X):
        F = self.F.forward(X)  # BxNxK
        S = self.S.forward(X)  # BxNxM
        S = torch.transpose(S, -1, -2)  # BxMxN
        G = torch.bmm(S, F)  # BxMxK
        sz = G.size()
        G = G.view(-1, sz[-1]*sz[-2])  # BxMK
        Y = self.L.forward(G)  # BxFC_dims[-1]
        return Y


class PointNetBilinearPool(nn.Module):

    def __init__(self, MLP1_dims, FC1_dims, MLP2_dims, FC2_dims, FC_dims):
        assert(MLP1_dims[-1] == FC1_dims[0])
        assert(MLP2_dims[-1] == FC2_dims[0])
        super(PointNetBilinearPool, self).__init__()
        self.F1 = nn.Sequential(
            PointNetGlobalMax(MLP1_dims),
            *get_MLP_layers(FC1_dims, False)
        )  # BxFC1_dims[-1]
        self.F2 = nn.Sequential(
            PointNetGlobalMax(MLP2_dims),
            *get_MLP_layers(FC2_dims, False)
        )  # BxFC2_dims[-1]
        self.L = nn.Sequential(*get_MLP_layers(FC_dims, False))

    def forward(self, X):
        F1 = self.F1.forward(X)  # BxK1
        F2 = self.F2.forward(X)  # BxK2
        F1 = F1.unsqueeze(-1)  # BxK1x1
        F2 = F2.unsqueeze(-2)  # Bx1xK2
        G = torch.bmm(F1, F2)  # BxK1xK2

        sz = G.size()
        G = G.view(-1, sz[-1]*sz[-2])
        Y = self.L.forward(G)
        return Y


class PointPairNet(nn.Module):

    def __init__(self, dims, FC_dims):
        assert(dims[-1] == FC_dims[0])
        super(PointPairNet, self).__init__()
        self.L = nn.Sequential(*get_MLP_layers(dims, False))
        self.Pool = nn.AdaptiveMaxPool2d((1, 1))
        self.F = nn.Sequential(*get_MLP_layers(FC_dims, False))

    def forward(self, X):
        sz = X.size()  # BxNx3
        Xr = X.view(sz[0], 1, sz[1], sz[2]).expand(
            sz[0], sz[1], sz[1], sz[2])  # BxNxNx3
        Xrc = torch.cat((Xr, Xr.transpose(1, 2)), dim=-1)  # BxNxNx6
        G = self.L.forward(Xrc).transpose(1, -1)  # BxKxNxN

        P = self.Pool.forward(G).squeeze(-1).squeeze(-1)  # BxK
        Y = self.F.forward(P)
        return Y


class BoostedPointPairNet(PointPairNet):

    def __init__(self, d, dims, FC_dims, max_pool=True):
        super(BoostedPointPairNet, self).__init__(dims, FC_dims)
        self.d = d
        self.add_module(
            'BoostPool',
            nn.AdaptiveMaxPool1d(1) if max_pool else nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, X):
        n = X.size()[1]
        X = X.transpose(0, 1)  # NxBx3
        # rid = torch.randperm(n)
        # X = X[rid,...]
        Xs = torch.chunk(X, self.d, dim=0)
        Ys = []
        for Xi in Xs:
            Xi = Xi.transpose(0, 1).contiguous()  # Bxmx3
            Yi = super(BoostedPointPairNet, self).forward(Xi)  # BxC
            Ys.append(Yi.unsqueeze(-1))
        Y = torch.cat(Ys, dim=-1)  # BxCxd
        Y = self.BoostPool.forward(Y).squeeze(-1)  # BxC
        return Y


class BoostedPointPairNet2(nn.Module):
    ''' More efficiently implemented than BoostedPointPairNet '''

    def __init__(self, boost_factor, dims, FC_dims, sym_pool_max=True, boost_pool_max=True):
        assert(dims[-1] == FC_dims[0])
        super(BoostedPointPairNet2, self).__init__()
        self.boost_factor = boost_factor
        self.L = nn.Sequential(*get_MLP_layers(dims, False))
        self.SymPool = nn.AdaptiveMaxPool3d((1, 1, dims[-1])) if sym_pool_max\
            else nn.AdaptiveAvgPool3d((1, 1, dims[-1]))
        self.F = nn.Sequential(*get_MLP_layers(FC_dims, False))
        self.BoostPool = nn.AdaptiveMaxPool2d((1, FC_dims[-1])) if boost_pool_max\
            else nn.AdaptiveAvgPool2d((1, FC_dims[-1]))

    def forward(self, X):
        b, n, din = X.size()
        d = self.boost_factor
        m = n/d
        assert(m*d == n)
        Xr = X.view(b, d, 1, m, din).expand(b, d, m, m, din)
        Xrc = torch.cat((Xr, Xr.transpose(2, 3)), dim=-1)  # bxdxmxmx6
        G = self.L.forward(Xrc)  # bxdxmxmxK
        P = self.SymPool.forward(G).squeeze(-2).squeeze(-2)  # bxdxK
        Y = self.F.forward(P)  # bxdxC
        Y = self.BoostPool.forward(Y).squeeze(-2)  # bxC
        return Y


class BoostedPointPairNetSuccessivePool(nn.Module):
    ''' Change SymPool to successive pool '''

    def __init__(self, boost_factor, dims, FC_dims, sym_pool_max=True, boost_pool_max=True):
        assert(dims[-1] == FC_dims[0])
        super(BoostedPointPairNetSuccessivePool, self).__init__()
        self.boost_factor = boost_factor
        self.L = nn.Sequential(*get_MLP_layers(dims, False))
        self.dims = dims
        self.sym_pool_max = sym_pool_max
        self.F = nn.Sequential(*get_MLP_layers(FC_dims, False))
        self.BoostPool = nn.AdaptiveMaxPool2d((1, FC_dims[-1])) if boost_pool_max\
            else nn.AdaptiveAvgPool2d((1, FC_dims[-1]))

    def forward(self, X):
        b, n, din = X.size()
        d = self.boost_factor
        m = n/d
        assert(m*d == n)
        Xr = X.view(b, d, 1, m, din).expand(b, d, m, m, din)
        Xrc = torch.cat((Xr, Xr.transpose(2, 3)), dim=-1)  # bxdxmxmx6
        G = self.L.forward(Xrc)  # bxdxmxmxK
        if self.sym_pool_max:  # average each point, then max across all points
            Pr = Functional.adaptive_avg_pool3d(
                G, (m, 1, self.dims[-1])).squeeze(-2)  # bxdxmxK
            P = Functional.adaptive_max_pool2d(
                Pr, (1, self.dims[-1])).squeeze(-2)  # bxdxK
        else:  # max each point, then average over all points
            Pr = Functional.adaptive_max_pool3d(
                G, (m, 1, self.dims[-1])).squeeze(-2)  # bxdxmxK
            P = Functional.adaptive_avg_pool2d(
                Pr, (1, self.dims[-1])).squeeze(-2)  # bxdxK
        Y = self.F.forward(P)  # bxdxC
        Y = self.BoostPool.forward(Y).squeeze(-2)  # bxC
        return Y


class BoostedPointNetVanilla(nn.Module):

    def __init__(self, boost_factor, dims, FC_dims, boost_pool_max=True):
        assert(dims[-1] == FC_dims[0])
        super(BoostedPointNetVanilla, self).__init__()
        self.boost_factor = boost_factor
        self.L = nn.Sequential(*get_MLP_layers(dims, False))
        self.Pool = nn.AdaptiveMaxPool2d((1, dims[-1]))
        self.F = nn.Sequential(*get_MLP_layers(FC_dims, False))
        self.BoostPool = nn.AdaptiveMaxPool2d((1, FC_dims[-1])) if boost_pool_max\
            else nn.AdaptiveAvgPool2d((1, FC_dims[-1]))

    def forward(self, X):
        b, n, din = X.size()
        d = self.boost_factor
        m = n/d
        assert(m*d == n)
        Xr = X.view(b, d, m, din)  # bxdxmx3
        F = self.L.forward(Xr)  # bxdxmxK
        Fp = self.Pool.forward(F).squeeze(-2)  # bxdxK
        Yp = self.F.forward(Fp).unsqueeze(0)  # 1xbxdxC
        Y = self.BoostPool.forward(Yp).squeeze(0).squeeze(-2)  # bxC
        return Y