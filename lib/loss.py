import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_distance.chamfer_loss import ChamferLoss
from lib import geom_utils
from lib.smr import SoftRenderer
import pdb


def neg_iou_loss(predict, target, avg = True):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    if(avg):
        return 1. - (intersect / union).sum() / intersect.nelement()
    else:
        return 1. - (intersect / union)


def transformtation_loss(points, rot_pred, trans_pred, scale_pred, pose_pred, pose_gt, trans_gt, is_symmetric):
    """
    points: 3d model points BxNx3
    pose_gt: ground truth pose Bx8 (scale, translation, rotation)
    pose_pred: predicted pose Bx8 (scale, translation, roatation)
    """
    points_gt = geom_utils.quat_rotate(points, pose_gt[:, 4:])
    points_pred = geom_utils.quat_rotate(points, rot_pred)

    # if is_symmetric:
    dists_squared = (points_gt.unsqueeze(1) - points_pred.unsqueeze(2)) ** 2 # BxNxN
    dists = dists_squared.clone()
    dists_norm_squared = dists_squared.sum(dim=-1)
    assign = dists_norm_squared.argmin(dim=1)
    ids_row = torch.arange(dists.shape[0]).unsqueeze(1).repeat(1, dists.shape[1])
    ids_col = torch.arange(dists.shape[1]).unsqueeze(0).repeat(dists.shape[0], 1)
    rot_loss = dists_squared[ids_row, assign, ids_col].mean()
    # else:
    #     dists_squared = (points_gt - points_pred) ** 2
    #     dists_squared_sum = dists_squared.sum(dim=-1)
    #     rot_loss = dists_squared_sum.mean()
    # points_gt = geom_utils.orthographic_proj_withz(points, pose_gt, pose_gt[:, 3:4])
    # points_pred = geom_utils.orthographic_proj_withz(points, pose_pred, pose_pred[:, 3:4])
    # rot_loss
    
    # is_asymmetric = is_symmetric.logical_not()
    # points_gt_sym = points_gt[is_symmetric]
    # points_gt_asym = points_gt[is_asymmetric]
    # points_pred_sym = points_pred[is_symmetric]
    # points_pred_asym = points_pred[is_asymmetric]
    # dists_norm_asym = torch.norm(points_pred_asym - points_gt_asym, dim=-1)# BxN
    # dists_norm_sym = torch.norm(points_pred_sym.unsqueeze(2) - points_gt_sym.unsqueeze(1), dim=-1) # BxNxN
    # dists_norm_asym = dists_norm_asym.mean(dim=-1)
    # if dists_norm_sym.size(0) > 0:
    #     dists_norm_sym = torch.mean(torch.min(dists_norm_sym, dim=-1)[0], dim=-1)
    #     dists_norm = torch.cat([dists_norm_sym, dists_norm_asym])
    # else:
    #     dists_norm = dists_norm_asym.clone()
    # rot_loss = dists_norm.mean()
    scale_loss = torch.nn.functional.l1_loss(scale_pred, pose_gt[:, 0:1])
    center_loss = torch.nn.functional.l1_loss(trans_pred[:, :2], trans_gt[:, :2])
    z_loss = torch.nn.functional.l1_loss(trans_pred[:, 2:], trans_gt[:, 2:])
    losses = rot_loss + center_loss + z_loss + scale_loss
    return losses


class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            if laplacian[i, i]!=0: laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.matmul(self.laplacian, x)
        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).sum(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x

def camera_loss(cam_pred, cam_gt, margin):
    """
    cam_* are B x 7, [sc, tx, ty, quat]
    Losses are in similar magnitude so one margin is ok.
    """
    rot_pred = cam_pred[:, -4:]
    rot_gt = cam_gt[:, -4:]

    rot_loss = hinge_loss(quat_loss_geodesic(rot_pred, rot_gt), margin)
    # Scale and trans.
    # st_loss = (cam_pred[:, :-4] - cam_gt[:, :-4])**2
    # st_loss = hinge_loss(st_loss.view(-1), margin)

    return rot_loss.mean()

def hinge_loss(loss, margin):
    # Only penalize if loss > margin
    zeros = torch.autograd.Variable(torch.zeros(1).cuda(), requires_grad=False)
    return torch.max(loss - margin, zeros)


def quat_loss_geodesic(q1, q2):
    '''
    Geodesic rotation loss.
    
    Args:
        q1: N X 4
        q2: N X 4
    Returns:
        loss : N x 1
    '''
    q1 = torch.unsqueeze(q1, 1)
    q2 = torch.unsqueeze(q2, 1)
    q2_conj = torch.cat([ q2[:, :, [0]] , -1*q2[:, :, 1:4] ], dim=-1)
    q_rel = geom_utils.hamilton_product(q1, q2_conj)
    q_loss = 1 - torch.abs(q_rel[:, :, 0])
    # we can also return q_loss*q_loss
    return q_loss
    

def quat_loss(q1, q2):
    '''
    Anti-podal squared L2 loss.
    
    Args:
        q1: N X 4
        q2: N X 4
    Returns:
        loss : N x 1
    '''
    q_diff_loss = (q1-q2).pow(2).sum(1)
    q_sum_loss = (q1+q2).pow(2).sum(1)
    q_loss, _ = torch.stack((q_diff_loss, q_sum_loss), dim=1).min(1)
    return q_loss


class Loss_old(nn.Module):
    """ Loss for training DeformNet.
        Use NOCS coords to supervise training.
    """
    def __init__(self, corr_wt, cd_wt, entropy_wt, deform_wt):
        super(Loss_old, self).__init__()
        self.threshold = 0.1
        self.chamferloss = ChamferLoss()
        self.corr_wt = corr_wt
        self.cd_wt = cd_wt
        self.entropy_wt = entropy_wt
        self.deform_wt = deform_wt

    def forward(self, assign_mat, deltas, prior, nocs, model):
        """
        Args:
            assign_mat: bs x n_pts x nv
            deltas: bs x nv x 3
            prior: bs x nv x 3
        """
        inst_shape = prior + deltas
        # smooth L1 loss for correspondences
        soft_assign = F.softmax(assign_mat, dim=2)
        coords = torch.bmm(soft_assign, inst_shape)  # bs x n_pts x 3
        diff = torch.abs(coords - nocs)
        less = torch.pow(diff, 2) / (2.0 * self.threshold)
        higher = diff - self.threshold / 2.0
        corr_loss = torch.where(diff > self.threshold, higher, less)
        corr_loss = torch.mean(torch.sum(corr_loss, dim=2))
        corr_loss = self.corr_wt * corr_loss
        # entropy loss to encourage peaked distribution
        log_assign = F.log_softmax(assign_mat, dim=2)
        entropy_loss = torch.mean(-torch.sum(soft_assign * log_assign, 2))
        entropy_loss = self.entropy_wt * entropy_loss
        # cd-loss for instance reconstruction
        cd_loss, _, _ = self.chamferloss(inst_shape, model)
        cd_loss = self.cd_wt * cd_loss
        # L2 regularizations on deformation
        deform_loss = torch.norm(deltas, p=2, dim=2).mean()
        deform_loss = self.deform_wt * deform_loss
        # total loss
        total_loss = corr_loss + entropy_loss + cd_loss + deform_loss
        return total_loss, corr_loss, cd_loss, entropy_loss, deform_loss


class Loss(nn.Module):
    """ Loss for training DeformNet.
        Use NOCS coords to supervise training.
    """
    def __init__(self, opts, use_gt_pose=False, use_gt_model=False):
        super(Loss, self).__init__()
        self.opts = opts
        self.threshold = 0.1
        self.chamferloss = ChamferLoss()
        self.kpt_loss = torch.nn.MSELoss()
        self.renderer = SoftRenderer(opts.img_size, opts.renderer_type)
        self.corr_wt = opts.corr_wt
        self.cd_wt = opts.cd_wt
        self.entropy_wt = opts.entropy_wt
        self.deform_wt = opts.deform_wt
        self.mask_wt = opts.mask_wt
        if opts.no_pose_loss:
            self.pose_wt = 0
            self.pose_param_wt = 0
        else:
            self.pose_wt = opts.pose_wt
            self.pose_param_wt = opts.pose_param_wt
        self.project_wt = opts.project_wt
        self.semi = opts.semi
        self.use_gt_pose = use_gt_pose
        self.use_gt_model = use_gt_model
        self.init_epoch = opts.deform_epoch
        self.direct_nocs = opts.version == 'v3' or opts.version == 'v4'
        self.with_recon = opts.with_recon
        self.recon_wt = opts.recon_wt
        self.feat_align = opts.feat_align
        self.align_wt = opts.align_wt
        
        self.sym = True if opts.select_class in ['bottle', 'can', 'bowl', 'mug'] else False
        

    def forward(self, outputs, gt_trans_local, pred_sTR, gt_sTR, gt_mask, 
        prior, nocs, model, face, cam, is_real, is_symmetric, epoch, points):
        """
        Args:
            assign_mat: bs x n_pts x nv
            deltas: bs x nv x 3
            prior: bs x nv x 3
        """
        bs = model.size(0)
        is_cam = is_real.logical_not()
        losses = dict()
        assign_mat = outputs['assign_mat']
        deltas = outputs['deltas']
        inst_shape = prior + deltas
        if 'recon' in outputs.keys():
            recon = outputs['recon']
            recon_loss_1 = torch.nn.functional.mse_loss(recon, points) 
            recon_loss_2, _, _ = self.chamferloss(recon, points)
            recon_loss = 10*recon_loss_1 + recon_loss_2


        # render mask loss
        if self.use_gt_pose:
            proj_pose = gt_sTR
        else:
            proj_pose = pred_sTR
    
        if self.feat_align:
            feat_map = outputs['feat_map'].detach()
            di = feat_map.size(1)
            feat_pix = outputs['feat_pix'].detach()
            pix_loc = geom_utils.projection(assign_mat, proj_pose.detach(), cam, self.opts.img_size)
            pix_loc = pix_loc.long()
            pix_loc = pix_loc.unsqueeze(1).repeat(1, di, 1)
            feat_pix_proj = torch.gather(feat_map, 2, pix_loc).contiguous()
            feat_align_loss = nn.functional.mse_loss(feat_pix_proj[bs:], feat_pix[bs:])
            feat_align_loss = self.align_wt * (feat_align_loss)
            cd_loss_real, _, _ = self.chamferloss(assign_mat[bs:], inst_shape[bs:].detach())
            feat_align_loss = feat_align_loss + 0.01 * cd_loss_real
            
        pred_seen, _, _ = self.renderer(inst_shape, face, cam, proj_pose, \
            image_size=[self.opts.img_size, self.opts.img_size])
        mask_pred_seen = pred_seen[:,3,:,:]
        mask_loss = neg_iou_loss(mask_pred_seen, gt_mask)
        mask_loss = self.mask_wt * mask_loss
        if self.semi:
            if len(torch.nonzero(is_cam)) == 0:
                total_loss = mask_loss
                losses['mask_loss'] = mask_loss
                losses['total_loss'] = total_loss
                return total_loss, losses
            # pose transformation loss
            pred_scales, pred_trans, pred_rots = outputs['pose']
            pose_loss = transformtation_loss(model[is_cam], pred_rots[:bs][is_cam], \
                pred_trans[:bs][is_cam], pred_scales[:bs][is_cam], pred_sTR[:bs][is_cam], \
                gt_sTR[:bs][is_cam], gt_trans_local[is_cam], self.sym)
            pose_loss = self.pose_wt * pose_loss
            # pose param loss
            pose_param_loss = camera_loss(pred_sTR[:bs][is_cam], gt_sTR[is_cam], 0)
            pose_param_loss = self.pose_param_wt * pose_param_loss
            if not self.direct_nocs:
                # smooth L1 loss for correspondences
                soft_assign = F.softmax(assign_mat, dim=2)
                coords = torch.bmm(soft_assign, inst_shape)  # bs x n_pts x 3
                diff = torch.abs(coords[:bs][is_cam] - nocs[is_cam])
                less = torch.pow(diff, 2) / (2.0 * self.threshold)
                higher = diff - self.threshold / 2.0
                corr_loss = torch.where(diff > self.threshold, higher, less)
                corr_loss = torch.mean(torch.sum(corr_loss, dim=2))
                corr_loss = self.corr_wt * corr_loss
                # # back project loss
                backproj_loss = torch.tensor(0.0, device=coords.device)
                # entropy loss to encourage peaked distribution
                log_assign = F.log_softmax(assign_mat, dim=2)
                entropy_loss = torch.mean(-torch.sum(soft_assign * log_assign, 2))
                entropy_loss = self.entropy_wt * entropy_loss
            else:
                diff = torch.abs(assign_mat[:bs][is_cam] - nocs[is_cam])
                less = torch.pow(diff, 2) / (2.0 * self.threshold)
                higher = diff - self.threshold / 2.0
                corr_loss = torch.where(diff > self.threshold, higher, less)
                corr_loss = torch.mean(torch.sum(corr_loss, dim=2))
                # corr_loss = nn.functional.smooth_l1_loss(assign_mat[:bs][is_cam], \
                #         nocs[is_cam], beta=self.threshold)
                # corr_loss_1 = nn.functional.mse_loss(assign_mat[:bs][is_cam], nocs[is_cam])
                if self.sym:
                    corr_loss_2, _, _ = self.chamferloss(assign_mat[:bs][is_cam], nocs[is_cam])
                    corr_loss = self.corr_wt * (corr_loss + corr_loss_2)
                else:
                    corr_loss = self.corr_wt * corr_loss
                backproj_loss = torch.tensor(0.0, device=assign_mat.device)
                entropy_loss = torch.tensor(0.0, device=assign_mat.device)

            # cd-loss for instance reconstruction
            cd_loss, _, _ = self.chamferloss(inst_shape[:bs][is_cam], model[is_cam])
            cd_loss = self.cd_wt * cd_loss
            # L2 regularizations on deformation
            deform_loss = torch.norm(deltas, p=2, dim=2).mean()
            deform_loss = self.deform_wt * deform_loss
        else:
            # pose transformation loss
            pred_scales, pred_trans, pred_rots = outputs['pose']
            pose_loss = transformtation_loss(model, pred_rots, pred_trans, pred_scales, \
                        pred_sTR, gt_sTR, gt_trans_local, self.sym)
            pose_loss = self.pose_wt * pose_loss
            # pose param loss
            pose_param_loss = camera_loss(pred_sTR, gt_sTR, 0)
            pose_param_loss = self.pose_param_wt * pose_param_loss
            if not self.direct_nocs:
                # smooth L1 loss for correspondences
                soft_assign = F.softmax(assign_mat, dim=2)
                coords = torch.bmm(soft_assign, inst_shape)  # bs x n_pts x 3
                diff = torch.abs(coords - nocs)
                less = torch.pow(diff, 2) / (2.0 * self.threshold)
                higher = diff - self.threshold / 2.0
                corr_loss = torch.where(diff > self.threshold, higher, less)
                corr_loss = torch.mean(torch.sum(corr_loss, dim=2))
                corr_loss = self.corr_wt * corr_loss
                # entropy loss to encourage peaked distribution
                log_assign = F.log_softmax(assign_mat, dim=2)
                entropy_loss = torch.mean(-torch.sum(soft_assign * log_assign, 2))
                entropy_loss = self.entropy_wt * entropy_loss
            else:
                diff = torch.abs(assign_mat - nocs)
                less = torch.pow(diff, 2) / (2.0 * self.threshold)
                higher = diff - self.threshold / 2.0
                corr_loss = torch.where(diff > self.threshold, higher, less)
                corr_loss = torch.mean(torch.sum(corr_loss, dim=2))
                # corr_loss = nn.functional.smooth_l1_loss(assign_mat[:bs][is_cam], \
                #         nocs[is_cam], beta=self.threshold)
                # corr_loss_1 = nn.functional.mse_loss(assign_mat[:bs][is_cam], nocs[is_cam])
                if self.sym:
                    corr_loss_2, _, _ = self.chamferloss(assign_mat, nocs)
                    corr_loss = self.corr_wt * (corr_loss + corr_loss_2)
                else:
                    corr_loss = self.corr_wt * corr_loss
                backproj_loss = torch.tensor(0.0, device=assign_mat.device)
                entropy_loss = torch.tensor(0.0, device=assign_mat.device)
            # cd-loss for instance reconstruction
            cd_loss, _, _ = self.chamferloss(inst_shape, model)
            cd_loss = self.cd_wt * cd_loss
            # L2 regularizations on deformation
            deform_loss = torch.norm(deltas, p=2, dim=2).mean()
            deform_loss = self.deform_wt * deform_loss
        # total loss
        if epoch < self.init_epoch:
            total_loss = corr_loss + entropy_loss + cd_loss + deform_loss
        else:
            total_loss = mask_loss + pose_loss + pose_param_loss + \
                corr_loss + entropy_loss + cd_loss + deform_loss
        if self.feat_align:
            losses['feat_align_loss'] = feat_align_loss
            total_loss = total_loss + feat_align_loss
        losses['mask_loss'] = mask_loss
        losses['pose_loss'] = pose_loss
        losses['pose_param_loss'] = pose_param_loss
        losses['corr_loss'] = corr_loss
        losses['entropy_loss'] = entropy_loss
        losses['cd_loss'] = cd_loss
        losses['deform_loss'] = deform_loss
        losses['backproj_loss'] = backproj_loss
        losses['total_loss'] = total_loss
        return total_loss, losses



def project(vertices, cams):
    vertices = vertices.bmm(cams.transpose(2, 1))
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
    x_ = x / (z + 1e-5)
    y_ = y / (z + 1e-5)

    # # we use x_ for x' and x__ for x'' etc.
    # r = torch.sqrt(x_ ** 2 + y_ ** 2)
    # x__ = 2 * (x_ - orig_size[0] / 2.) / orig_size[0]
    # y__ = 2 * (y_ - orig_size[1] / 2.) / orig_size[1]
    vertices = torch.stack([y_, x_], dim=-1)
    return vertices