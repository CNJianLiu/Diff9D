import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from lib.pspnet import PSPNet
from lib.pspnet_new import PSPNet as PSPNet_2
from lib import gcn3d
from lib.utils import normalize_to_box, sample_farthest_points, load_obj
import lib.pytorch_utils as pt_utils

import pdb

psp_models = {
    'resnet18': lambda: PSPNet_2(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet_2(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet_2(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
}

class PositionalEncoder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                       log_sampling=True, include_input=True,
                       periodic_fns=(torch.sin, )):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


class DenseFusion(nn.Module):
    def __init__(self, num_points):
        super(DenseFusion, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(64, 256, 1)
        self.conv2_cld = torch.nn.Conv1d(64, 256, 1)

        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, rgb_emb, cld_emb):
        bs, _, n_pts = cld_emb.size()
        feat_1 = torch.cat((rgb_emb, cld_emb), dim=1)
        rgb = F.relu(self.conv2_rgb(rgb_emb))
        cld = F.relu(self.conv2_cld(cld_emb))

        feat_2 = torch.cat((rgb, cld), dim=1)

        rgbd = F.relu(self.conv3(feat_1))
        rgbd = F.relu(self.conv4(rgbd))

        ap_x = self.ap1(rgbd)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([feat_1, feat_2, ap_x], 1) # 128 + 512 + 1024 = 1664




class PoseHeadNetV2(nn.Module):
    def __init__(self, num_points, num_obj, in_dim=1152, with_nocs=False, max_point=False):
        super(PoseHeadNetV2, self).__init__()

        self.num_points = num_points
        self.num_obj = num_obj
        self.with_nocs = with_nocs
        self.max_point = max_point
        if self.with_nocs:
            self.nocs_conv = nn.Sequential(
                                nn.Conv1d(3, 64, 1),
                                nn.ReLU(),
                                nn.Conv1d(64, 64, 1),
                                nn.ReLU(),
                                nn.Conv1d(64, 128, 1),
                                nn.ReLU()
                        )

            self.conv1_r = torch.nn.Conv1d(in_dim+128, 512, 1)
            self.conv1_t = torch.nn.Conv1d(in_dim+128, 512, 1)
            self.conv1_c = torch.nn.Conv1d(in_dim+128, 512, 1)
        else:
            self.conv1_r = torch.nn.Conv1d(in_dim, 512, 1)
            self.conv1_t = torch.nn.Conv1d(in_dim, 512, 1)
            self.conv1_c = torch.nn.Conv1d(in_dim, 512, 1)

        self.conv2_r = torch.nn.Conv1d(512, 256, 1)
        self.conv2_t = torch.nn.Conv1d(512, 256, 1)
        self.conv2_c = torch.nn.Conv1d(512, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj*4, 1)  # quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1)  # translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj*1, 1)  # scale

        self.ap1 = torch.nn.AvgPool1d(num_points)


    def forward(self, psp_feat, point_feat, choose, cat_id, nocs=None):
        if self.num_obj == 1:
            cat_id = torch.zeros_like(cat_id, device=cat_id.device)

        bs = point_feat.size()[0]
        di = point_feat.size()[1]

        if self.with_nocs:
            assert nocs is not None
            nocs = self.nocs_conv(nocs)
            pose_feat = torch.cat([point_feat, nocs], dim=1)
        else:
            pose_feat = point_feat.clone()

        rx = F.relu(self.conv1_r(pose_feat))
        tx = F.relu(self.conv1_t(pose_feat))
        cx = F.relu(self.conv1_c(pose_feat))


        
        if self.max_point:
            rx = self.conv2_r(rx)
            tx = self.conv2_t(tx)
            cx = self.conv2_c(cx)

            rx = torch.max(rx, dim=-1, keepdim=True)[0]
            tx = torch.max(tx, dim=-1, keepdim=True)[0]
            cx = torch.max(cx, dim=-1, keepdim=True)[0]
            rx = F.relu(self.conv3_r(rx))
            tx = F.relu(self.conv3_t(tx))
            cx = F.relu(self.conv3_c(cx))

            rx = self.conv4_r(rx).view(bs, self.num_obj, 4)
            tx = self.conv4_t(tx).view(bs, self.num_obj, 3)
            cx = self.conv4_c(cx).view(bs, self.num_obj, 1)
            
            indices = torch.arange(bs).cuda()
            out_rx = rx[indices, cat_id]
            out_tx = tx[indices, cat_id]
            out_cx = cx[indices, cat_id]

            out_rx = out_rx.contiguous()
            out_tx = out_tx.contiguous()
            out_cx = out_cx.contiguous()

            out_rx = F.normalize(out_rx, dim=-1)
            out_cx = F.relu(out_cx)

        else:
            rx = F.relu(self.conv2_r(rx))
            tx = F.relu(self.conv2_t(tx))
            cx = F.relu(self.conv2_c(cx))

            rx = F.relu(self.conv3_r(rx))
            tx = F.relu(self.conv3_t(tx))
            cx = F.relu(self.conv3_c(cx))


            rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
            tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
            cx = self.conv4_c(cx).view(bs, self.num_obj, 1, self.num_points)
            
            indices = torch.arange(bs).cuda()
            out_rx = rx[indices, cat_id]
            out_tx = tx[indices, cat_id]
            out_cx = cx[indices, cat_id]

            out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
            out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
            out_cx = out_cx.contiguous().transpose(2, 1).contiguous()

            out_rx = F.normalize(out_rx, dim=-1)
            out_cx = F.relu(out_cx)

            out_rx = out_rx.mean(dim=1)
            out_tx = out_tx.mean(dim=1)
            out_cx = out_cx.mean(dim=1)
        return out_cx, out_tx, out_rx




class PointNetfeat(nn.Module):
    def __init__(self, npoint = 2500, nlatent = 512):
        """Encoder"""

        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin = nn.Linear(nlatent, nlatent)

        # self.bn1 = torch.nn.BatchNorm1d(64)
        # self.bn2 = torch.nn.BatchNorm1d(128)
        # self.bn3 = torch.nn.BatchNorm1d(nlatent)
        # self.bn4 = torch.nn.BatchNorm1d(nlatent)

        self.npoint = npoint
        self.nlatent = nlatent

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x,_ = torch.max(x, 2)
        x = x.view(-1, self.nlatent)
        x = F.relu(self.lin(x))
        return x.unsqueeze(-1)






class PoseNetV3(nn.Module):
    def __init__(self, opts):
        super(PoseNetV3, self).__init__()
        self.opts = opts
        self.use_nocs_map = opts.use_nocs_map
        num_obj = 6 if opts.select_class == 'all' else 1
        self.num_obj = num_obj
        self.encoder = PSPNet(bins=(1, 2, 3, 6), backend='resnet18')
        self.deform_head = DeformNetV4(n_cat=num_obj, imp=opts.implicit)
        self.pose_head = PoseHeadNetV2(opts.n_pts, num_obj, in_dim=1792, 
                with_nocs=opts.use_nocs_map)

    
    def forward(self, points, img, choose, cat_id, prior):
        if self.num_obj == 1:
            cat_id = torch.zeros_like(cat_id, device=cat_id.device)
        outputs = {}
        bs = img.shape[0]
        img_feat, psp_feat = self.encoder(img) # B,512,8,8
        nocs, deltas, point_feat = self.deform_head(points, psp_feat, choose, cat_id, prior)
        if self.use_nocs_map:
            coords = nocs.detach().clone()
            coords = coords.permute(0, 2, 1).contiguous()
            outputs['pose'] = self.pose_head(psp_feat, point_feat, choose, cat_id, coords)
        else:
            outputs['pose'] = self.pose_head(psp_feat, point_feat, choose, cat_id)
        
        outputs['assign_mat'] = nocs
        outputs['deltas'] = deltas
        di = psp_feat.size(1)
        emb = psp_feat.view(bs, di, -1)
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        outputs['feat_map'] = psp_feat.reshape(bs, di, -1)
        outputs['feat_pix'] = emb
       
        return outputs

class DeformNetV4(nn.Module):
    def __init__(self, n_cat=6, nv_prior=1024, imp=False, depth_input=True,  max_point=False):
        super(DeformNetV4, self).__init__()
        self.n_cat = n_cat
        self.depth_input = depth_input
        self.max_point = max_point
        in_dim = 3 if self.depth_input else 2
        self.imp = imp
        self.instance_color = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
        )
        self.instance_geometry = GCN3D_segR()
            
        self.ap = nn.AdaptiveAvgPool1d(1)

        self.category_local = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.category_global = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
        )
        if self.imp:
            self.deformation = nn.Sequential(
                nn.Linear(2816+3, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 3),
            )
            self.nocs_prediction = nn.Sequential(
                nn.Linear(2816+3, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 3),
            )
        else:
            self.deformation = nn.Sequential(
                nn.Conv1d(2816, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 256, 1),
                nn.ReLU(),
                nn.Conv1d(256, n_cat*3, 1),
            )
            self.nocs_prediction = nn.Sequential(
                nn.Conv1d(2816, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 256, 1),
                nn.ReLU(),
                nn.Conv1d(256, n_cat*3, 1),
            )

        # Initialize weights to be small so initial deformations aren't so big
        self.deformation[4].weight.data.normal_(0, 0.0001)

    def forward(self, points, out_img, choose, cat_id, prior):
        """
        Args:
            points: bs x n_pts x 3
            img: bs x 3 x H x W
            choose: bs x n_pts
            cat_id: bs
            prior: bs x nv x 3

        Returns:
            assign_mat: bs x n_pts x nv
            inst_shape: bs x nv x 3
            deltas: bs x nv x 3
            log_assign: bs x n_pts x nv, for numerical stability

        """
        if self.n_cat == 1:
            cat_id = torch.zeros_like(cat_id, device=cat_id.device)
        bs, n_pts = points.size()[:2]
        nv = prior.size()[1]
        # instance-specific features
        di = out_img.size()[1]
        emb = out_img.view(bs, di, -1)
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        inst_local, inst_global_p = self.instance_geometry(points, emb)
        inst_global = torch.max(inst_global_p, dim=-1, keepdim=True)[0]
        # category-specific features
        cat_prior = prior.permute(0, 2, 1)
        cat_local = self.category_local(cat_prior)    # bs x 64 x n_pts
        cat_global_p = self.category_global(cat_local)  # bs x 1024 x 1
        cat_global = torch.max(cat_global_p, dim=-1, keepdim=True)[0]
        # assignemnt matrix
        assign_feat = torch.cat((inst_global_p, cat_global.repeat(1, 1, n_pts)), dim=1)     # bs x 2176 x n_pts
        # deformation field
        deform_feat = torch.cat((cat_global_p, inst_global.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
        index = cat_id + torch.arange(bs, dtype=torch.long, device=cat_id.device) * self.n_cat
        
        # cat_prior = self.pos_encoding_sin_wave(cat_prior)
        # points = self.pos_encoding_sin_wave(points)
        if self.imp:
            deform_feat = torch.cat([deform_feat, cat_prior], dim=1)
            deform_feat = deform_feat.permute(0, 2, 1).contiguous()
            deform_feat = deform_feat.reshape(-1, deform_feat.shape[-1])
            deltas = self.deformation(deform_feat)
            deltas = deltas.reshape(bs, -1, 3).contiguous() # bs, nv, 3
            assign_feat = torch.cat([assign_feat, points.permute(0,2,1)], dim=1)
            assign_feat = assign_feat.permute(0, 2, 1).contiguous()
            assign_feat = assign_feat.reshape(-1, assign_feat.shape[-1])
            nocs = self.nocs_prediction(assign_feat)
            nocs = nocs.reshape(bs, -1, 3).contiguous() # bs, npt, 3
        else:
            deltas = self.deformation(deform_feat)
            deltas = deltas.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
            deltas = torch.index_select(deltas, 0, index)   # bs x 3 x nv
            deltas = deltas.permute(0, 2, 1).contiguous()   # bs x nv x 3
            nocs = self.nocs_prediction(assign_feat)
            nocs = nocs.view(-1, 3, n_pts).contiguous()   # bs, nc*3, n_pts -> bs*nc, nv, n_pts
            nocs = torch.index_select(nocs, 0, index)   # bs x 3 x n_pts
            nocs = nocs.permute(0, 2, 1).contiguous()    # bs x n_pts x 3
        
        return nocs, deltas, inst_global_p

class GCN3D_segR(nn.Module):
    def __init__(self, support_num= 7, neighbor_num= 10):
        super(GCN3D_segR, self).__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 64, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(128, 128, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.conv_3 = gcn3d.Conv_layer(256, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = gcn3d.Conv_layer(256, 512, support_num= support_num)
        
        self.rgb_conv_1 = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.rgb_conv_2 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        dim_fuse = sum([128, 128, 256, 256, 512, 512, 16])

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)", 
                rgb_f):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        # ss = time.time()
        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace= True)

        rgb_f = self.rgb_conv_1(rgb_f)
        fm_0 = torch.cat([fm_0, rgb_f.permute(0, 2, 1)], dim=-1)

        fm_1 = F.relu(self.bn1(self.conv_1(neighbor_index, vertices, fm_0).transpose(1,2)).transpose(1,2), inplace= True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_1,
                                                  min(self.neighbor_num, v_pool_1.shape[1] // 8))
        fm_2 = F.relu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1).transpose(1,2)).transpose(1,2), inplace= True)
        fm_3 = F.relu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2).transpose(1,2)).transpose(1,2), inplace= True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                     v_pool_2.shape[1] // 8))
        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        f_global = fm_4.max(1)[0] #(bs, f)

        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d.indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d.indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d.indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)
        f_global = f_global.unsqueeze(1).repeat(1, vertice_num, 1)

        feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim= 2)
        fm_fuse = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, f_global], dim= 2)
        return feat.permute(0, 2, 1), fm_fuse.permute(0, 2, 1)