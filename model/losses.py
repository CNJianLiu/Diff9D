import torch

def SmoothL1Dis(p1, p2, threshold=0.1):
    '''
    p1: b*n*3
    p2: b*n*3
    '''
    
    diff = torch.abs(p1 - p2)
    less = torch.pow(diff, 2) / (2.0 * threshold)
    higher = diff - threshold / 2.0
    dis = torch.where(diff > threshold, higher, less)
    # print("dis:", dis, "dis shape:", dis.shape)
    # assert False
    # print("torch.sum(dis, dim=1):", torch.sum(dis, dim=1))
    a = torch.sum(dis, dim=1)
    # print("a mean:", a.mean())
    # assert False
    dis = torch.mean(torch.sum(dis, dim=2 if len(p1.shape)==3 else 1))
    # dis = torch.sum(dis, dim=1).mean()
    # assert False
    return dis


def ChamferDis(p1, p2):
    '''
    p1: b*n1*3
    p2: b*n2*3
    '''
    dis = torch.norm(p1.unsqueeze(2) - p2.unsqueeze(1), dim=3)
    dis1 = torch.min(dis, 2)[0]
    dis2 = torch.min(dis, 1)[0]
    dis = 0.5*dis1.mean(1) + 0.5*dis2.mean(1)
    return dis.mean()


def PoseDis(s0, t0, R0, s1, t1, R1):
    '''
    r1, r2: b*6
    t1, t2: b*3
    s1, s2: b*3
    '''
    dis_r = torch.mean(torch.norm(R0 - R1, dim=1))
    dis_t = torch.mean(torch.norm(t0 - t1, dim=1))
    dis_s = torch.mean(torch.norm(s0 - s1, dim=1))

    return dis_r + dis_t + dis_s
