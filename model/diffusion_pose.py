import torch
import torch.nn as nn
from rotation_utils import Ortho6d2Mat
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
from tools.visual_points import visual_points
from model.network import time_cond, ShapeEstimator, shape_mlp
from model.network import pose_s_mlp, pose_s_condition_FCAM1, pose_s_condition_FCAM2, pose_s_condition_FCAM3, pose_s_condition_FCAM4, pose_s_condition_FCAM5, pose_s_condition_FCAM6, pose_s_condition_FCAM7, pose_s_condition_concat0, pose_s_condition_concat1, pose_s_condition_concat2, pose_s_decoder
from model.network import pose_t_mlp, pose_t_condition_FCAM1, pose_t_condition_FCAM2, pose_t_condition_FCAM3, pose_t_condition_FCAM4, pose_t_condition_FCAM5, pose_t_condition_FCAM6, pose_t_condition_FCAM7, pose_t_condition_concat0, pose_t_condition_concat1, pose_t_condition_concat2, pose_t_decoder
from model.network import pose_R_mlp, pose_R_condition_FCAM1, pose_R_condition_FCAM2, pose_R_condition_FCAM3, pose_R_condition_FCAM4, pose_R_condition_FCAM5, pose_R_condition_FCAM6, pose_R_condition_FCAM7, pose_R_condition_concat0, pose_R_condition_concat1, pose_R_condition_concat2, pose_R_decoder
from pointnet import PointNetfeat
from losses import SmoothL1Dis, ChamferDis, PoseDis
from diffusers import DDPMScheduler, DDIMScheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
ddim_scheduler = DDIMScheduler(num_train_timesteps=1000)
ddim_scheduler.set_timesteps(num_inference_steps=3)

def pose_guassian(b):
    noise_s = torch.randn(b, 3).cuda()
    noise_t = torch.randn(b, 3).cuda()
    noise_R = torch.randn(b, 3, 3).cuda()
    # noise_R = Ortho6d2Mat(noise_R[:, :3].contiguous(), noise_R[:, 3:].contiguous()).view(-1,3,3) # bs*3*3
    return noise_s, noise_t, noise_R

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al.(http://arxiv.org/abs/1812.07035)
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)
    Returns:
        6D rotation representation, of size (*, 6)
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def pose_forward_diffusion(delta_s, gt_s, delta_t, gt_t, delta_R, gt_R, diffusion_steps):
    diff_s = noise_scheduler.add_noise(gt_s, delta_s, diffusion_steps)  
    diff_t = noise_scheduler.add_noise(gt_t, delta_t, diffusion_steps)
    diff_R = noise_scheduler.add_noise(gt_R, delta_R, diffusion_steps)
    
    return diff_s, diff_t, diff_R
 
class DIFFUSION(nn.Module):
    def __init__(self):
        super(DIFFUSION, self).__init__()
        self.rgb_cam_extractor = models.resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT)
        self.rgb_cam_extractor.fc = nn.Identity()
        self.pts_mlp = PointNetfeat()
        self.time_cond = time_cond() 
        self.ShapeEstimator = ShapeEstimator()
        self.shape_mlp = shape_mlp()

        self.pose_s_mlp = pose_s_mlp()
        self.denoise_s_net_FCAM1 = pose_s_condition_FCAM1()
        self.denoise_s_net_FCAM2 = pose_s_condition_FCAM2()
        self.denoise_s_net_FCAM3 = pose_s_condition_FCAM3()
        self.denoise_s_net_FCAM4 = pose_s_condition_FCAM4()
        self.denoise_s_net_FCAM5 = pose_s_condition_FCAM5()
        self.denoise_s_net_FCAM6 = pose_s_condition_FCAM6()
        self.denoise_s_net_FCAM7 = pose_s_condition_FCAM7()
        self.pose_s_condition_concat2 = pose_s_condition_concat2()
        self.pose_s_condition_concat1 = pose_s_condition_concat1()
        self.pose_s_condition_concat0 = pose_s_condition_concat0()
        self.pose_s_decoder = pose_s_decoder()
        
        self.pose_t_mlp = pose_t_mlp()
        self.denoise_t_net_FCAM1 = pose_t_condition_FCAM1()
        self.denoise_t_net_FCAM2 = pose_t_condition_FCAM2()
        self.denoise_t_net_FCAM3 = pose_t_condition_FCAM3()
        self.denoise_t_net_FCAM4 = pose_t_condition_FCAM4()
        self.denoise_t_net_FCAM5 = pose_t_condition_FCAM5()
        self.denoise_t_net_FCAM6 = pose_t_condition_FCAM6()
        self.denoise_t_net_FCAM7 = pose_t_condition_FCAM7()
        self.pose_t_condition_concat2 = pose_t_condition_concat2()
        self.pose_t_condition_concat1 = pose_t_condition_concat1()
        self.pose_t_condition_concat0 = pose_t_condition_concat0()
        self.pose_t_decoder = pose_t_decoder()
        
        self.pose_R_mlp = pose_R_mlp()
        self.denoise_R_net_FCAM1 = pose_R_condition_FCAM1()
        self.denoise_R_net_FCAM2 = pose_R_condition_FCAM2()
        self.denoise_R_net_FCAM3 = pose_R_condition_FCAM3()
        self.denoise_R_net_FCAM4 = pose_R_condition_FCAM4()
        self.denoise_R_net_FCAM5 = pose_R_condition_FCAM5()
        self.denoise_R_net_FCAM6 = pose_R_condition_FCAM6()
        self.denoise_R_net_FCAM7 = pose_R_condition_FCAM7()
        self.pose_R_condition_concat2 = pose_R_condition_concat2()
        self.pose_R_condition_concat1 = pose_R_condition_concat1()
        self.pose_R_condition_concat0 = pose_R_condition_concat0()
        self.pose_R_decoder = pose_R_decoder()

        self.max_step_size = 1000

    def forward(self, inputs):
        end_points = {}
        rgb = inputs['rgb']
        pts = inputs['pts'] # b*1024*3
        b = rgb.size(0)
        rgb_global = self.rgb_cam_extractor(rgb) # b*512
        c = torch.mean(pts, 1, keepdim=True)
        pts = pts - c
        pts_global, pts_global_local = self.pts_mlp(pts.permute(0, 2, 1)) # b*512, b*(512+64)*1024
        shape, nocs_shape = self.ShapeEstimator(rgb_global, pts_global_local) # bs*3*1024

        # plt.imshow(rgb[0])
        # # plt.colorbar()
        # # plt.title('Gaussian Noise2')
        # plt.show()
        # input()
        
        # a1 = shape.permute(0, 2, 1)
        # a2 = nocs_shape.permute(0, 2, 1)

        # a11 = a1[0]
        # shape1 = a11.cpu().detach().numpy()
        # visual_points(points=shape1)
        # input()

        # a21 = a2[0]
        # nocs_shape1 = a21.cpu().detach().numpy()
        # visual_points(points=nocs_shape1)
        # input()

        shape_global, nocs_shape_global = self.shape_mlp(shape, nocs_shape) # b*192
        
        if self.training:
            gt_s = inputs['gt_s']
            gt_t = inputs['gt_t']
            gt_t = gt_t - c.squeeze(1)
            gt_R = inputs['gt_R']

            # gt_R = matrix_to_rotation_6d(gt_R.permute(0, 2, 1)).reshape(gt_R.shape[0], -1)
            # gt_Rx = gt_R[:, :3]
            # gt_Ry = gt_R[:, 3:]
            # R = Ortho6d2Mat(gt_Rx, gt_Ry).view(-1,3,3) # bs*3*3

            delta_s = torch.randn(b, 3).cuda()
            delta_t = torch.randn(b, 3).cuda()
            delta_R = torch.randn(b, 3, 3).cuda()
            
            diffusion_steps = torch.randint(0, self.max_step_size, (b,), device='cuda').long() #[0, 999]
            time_cond = self.time_cond(diffusion_steps.float()) # b*128     
            diff_s, diff_t, diff_R = pose_forward_diffusion(delta_s, gt_s, delta_t, gt_t, delta_R, gt_R, diffusion_steps)
            condition_emb_init = torch.cat([time_cond, rgb_global, pts_global, shape_global, nocs_shape_global], dim=-1)    # (B,128+512+512+192+192)
            condition_emb_init = condition_emb_init.unsqueeze(1)   # (B,1,128+512+512+192+192)

            pose_s_emb_init = self.pose_s_mlp(diff_s)

            pose_s_emb_1 = self.denoise_s_net_FCAM1(torch.cat([condition_emb_init, pose_s_emb_init], dim=-1)) # (B,1,256+128+512+512+192+192)
            pose_s_emb_2 = self.denoise_s_net_FCAM2(pose_s_emb_1)
            pose_s_emb_3 = self.denoise_s_net_FCAM3(pose_s_emb_2)
            pose_s_emb_4 = self.denoise_s_net_FCAM4(pose_s_emb_3)
            pose_s_34 = self.pose_s_condition_concat2(pose_s_emb_3, pose_s_emb_4)
            pose_s_emb_5 = self.denoise_s_net_FCAM5(pose_s_34)
            pose_s_25 = self.pose_s_condition_concat1(pose_s_emb_2, pose_s_emb_5)
            pose_s_emb_6 = self.denoise_s_net_FCAM6(pose_s_25) 
            pose_s_16 = self.pose_s_condition_concat0(pose_s_emb_1, pose_s_emb_6)
            pose_s_emb_7 = self.denoise_s_net_FCAM7(pose_s_16) 
            pred_delta_s0 = self.pose_s_decoder(pose_s_emb_7)

            pose_t_emb_init = self.pose_t_mlp(diff_t)

            pose_t_emb_1 = self.denoise_t_net_FCAM1(torch.cat([condition_emb_init, pose_t_emb_init], dim=-1)) # (B,1,128+128+512+512)
            pose_t_emb_2 = self.denoise_t_net_FCAM2(pose_t_emb_1)
            pose_t_emb_3 = self.denoise_t_net_FCAM3(pose_t_emb_2)
            pose_t_emb_4 = self.denoise_t_net_FCAM4(pose_t_emb_3)
            pose_t_34 = self.pose_t_condition_concat2(pose_t_emb_3, pose_t_emb_4)
            pose_t_emb_5 = self.denoise_t_net_FCAM5(pose_t_34)
            pose_t_25 = self.pose_t_condition_concat1(pose_t_emb_2, pose_t_emb_5)
            pose_t_emb_6 = self.denoise_t_net_FCAM6(pose_t_25) 
            pose_t_16 = self.pose_t_condition_concat0(pose_t_emb_1, pose_t_emb_6)
            pose_t_emb_7 = self.denoise_t_net_FCAM7(pose_t_16) 
            pred_delta_t0 = self.pose_t_decoder(pose_t_emb_7)

            pose_R_emb_init = self.pose_R_mlp(diff_R.reshape(b, 9))
            
            pose_R_emb_1 = self.denoise_R_net_FCAM1(torch.cat([condition_emb_init, pose_R_emb_init], dim=-1)) # (B,1,128+128+512+512)
            pose_R_emb_2 = self.denoise_R_net_FCAM2(pose_R_emb_1)
            pose_R_emb_3 = self.denoise_R_net_FCAM3(pose_R_emb_2)
            pose_R_emb_4 = self.denoise_R_net_FCAM4(pose_R_emb_3)
            pose_R_34 = self.pose_R_condition_concat2(pose_R_emb_3, pose_R_emb_4)
            pose_R_emb_5 = self.denoise_R_net_FCAM5(pose_R_34)
            pose_R_25 = self.pose_R_condition_concat1(pose_R_emb_2, pose_R_emb_5)
            pose_R_emb_6 = self.denoise_R_net_FCAM6(pose_R_25) 
            pose_R_16 = self.pose_R_condition_concat0(pose_R_emb_1, pose_R_emb_6)
            pose_R_emb_7 = self.denoise_R_net_FCAM7(pose_R_16) 
            pred_delta_R0 = self.pose_R_decoder(pose_R_emb_7)
            pred_delta_R0 =  pred_delta_R0.reshape(b, 3, 3)
            # pred_delta_R0 = Ortho6d2Mat(pred_delta_Rx0, pred_delta_Ry0).view(-1,3,3) # bs*3*3

            end_points['pred_shape'] = shape.permute(0, 2, 1) # bs*1024*3
            end_points['pred_nocs_shape'] = nocs_shape.permute(0, 2, 1) # bs*1024*3

            end_points['pred_size0'] = pred_delta_s0
            end_points['pred_translation0'] = pred_delta_t0
            end_points['pred_rotation0'] = pred_delta_R0

            end_points['delta_size0'] = delta_s
            end_points['delta_translation0'] = delta_t
            end_points['delta_rotation0'] = delta_R
        else:
            sample_s, sample_t, sample_R = pose_guassian(b)
            # sample_s, sample_t, sample_R = s_init, t_init, R_init
            for i, t in enumerate(ddim_scheduler.timesteps):
            # for i, t in enumerate(noise_scheduler.timesteps):
                # if t > 300:
                #     continue
                time_cond = torch.full((b,), t, dtype=torch.float).cuda() #.cuda()
                # Get model pred
                with torch.no_grad():
                    time_cond = self.time_cond(time_cond)
                    condition_emb_init = torch.cat([time_cond, rgb_global, pts_global, shape_global, nocs_shape_global], dim=-1)    # (B,1792)
                    condition_emb_init = condition_emb_init.unsqueeze(1)   # (B,1,1792)

                    pose_s_emb_init = self.pose_s_mlp(sample_s)
                    pose_s_emb_1 = self.denoise_s_net_FCAM1(torch.cat([condition_emb_init, pose_s_emb_init], dim=-1))
                    pose_s_emb_2 = self.denoise_s_net_FCAM2(pose_s_emb_1)
                    pose_s_emb_3 = self.denoise_s_net_FCAM3(pose_s_emb_2)
                    pose_s_emb_4 = self.denoise_s_net_FCAM4(pose_s_emb_3)
                    pose_s_34 = self.pose_s_condition_concat2(pose_s_emb_3, pose_s_emb_4)
                    pose_s_emb_5 = self.denoise_s_net_FCAM5(pose_s_34)
                    pose_s_25 = self.pose_s_condition_concat1(pose_s_emb_2, pose_s_emb_5)
                    pose_s_emb_6 = self.denoise_s_net_FCAM6(pose_s_25) 
                    pose_s_16 = self.pose_s_condition_concat0(pose_s_emb_1, pose_s_emb_6)
                    pose_s_emb_7 = self.denoise_s_net_FCAM7(pose_s_16) 
                    pred_delta_s = self.pose_s_decoder(pose_s_emb_7)

                    pose_t_emb_init = self.pose_t_mlp(sample_t)
                    pose_t_emb_1 = self.denoise_t_net_FCAM1(torch.cat([condition_emb_init, pose_t_emb_init], dim=-1))
                    pose_t_emb_2 = self.denoise_t_net_FCAM2(pose_t_emb_1)
                    pose_t_emb_3 = self.denoise_t_net_FCAM3(pose_t_emb_2)
                    pose_t_emb_4 = self.denoise_t_net_FCAM4(pose_t_emb_3)
                    pose_t_34 = self.pose_t_condition_concat2(pose_t_emb_3, pose_t_emb_4)
                    pose_t_emb_5 = self.denoise_t_net_FCAM5(pose_t_34)
                    pose_t_25 = self.pose_t_condition_concat1(pose_t_emb_2, pose_t_emb_5)
                    pose_t_emb_6 = self.denoise_t_net_FCAM6(pose_t_25) 
                    pose_t_16 = self.pose_t_condition_concat0(pose_t_emb_1, pose_t_emb_6)
                    pose_t_emb_7 = self.denoise_t_net_FCAM7(pose_t_16) 
                    pred_delta_t = self.pose_t_decoder(pose_t_emb_7)

                    pose_R_emb_init = self.pose_R_mlp(sample_R.reshape(b, 9))
                    pose_R_emb_1 = self.denoise_R_net_FCAM1(torch.cat([condition_emb_init, pose_R_emb_init], dim=-1))
                    pose_R_emb_2 = self.denoise_R_net_FCAM2(pose_R_emb_1)
                    pose_R_emb_3 = self.denoise_R_net_FCAM3(pose_R_emb_2)
                    pose_R_emb_4 = self.denoise_R_net_FCAM4(pose_R_emb_3)
                    pose_R_34 = self.pose_R_condition_concat2(pose_R_emb_3, pose_R_emb_4)
                    pose_R_emb_5 = self.denoise_R_net_FCAM5(pose_R_34)
                    pose_R_25 = self.pose_R_condition_concat1(pose_R_emb_2, pose_R_emb_5)
                    pose_R_emb_6 = self.denoise_R_net_FCAM6(pose_R_25) 
                    pose_R_16 = self.pose_R_condition_concat0(pose_R_emb_1, pose_R_emb_6)
                    pose_R_emb_7 = self.denoise_R_net_FCAM7(pose_R_16) 
                    pred_delta_R = self.pose_R_decoder(pose_R_emb_7)
                    pred_delta_R =  pred_delta_R.reshape(b, 3, 3)
                    # pred_delta_R = Ortho6d2Mat(pred_delta_Rx, pred_delta_Ry).view(-1,3,3) # bs*3*3

                # Update sample with step
                # sample_s = noise_scheduler.step(pred_delta_s, t, sample_s).prev_sample
                # sample_t = noise_scheduler.step(pred_delta_t, t, sample_t).prev_sample
                # sample_R = noise_scheduler.step(pred_delta_R, t, sample_R).prev_sample
                sample_s = ddim_scheduler.step(pred_delta_s, t, sample_s).prev_sample
                sample_t = ddim_scheduler.step(pred_delta_t, t, sample_t).prev_sample
                sample_R = ddim_scheduler.step(pred_delta_R, t, sample_R).prev_sample
            
            end_points['pred_size'] = sample_s
            end_points['pred_translation'] = sample_t+c.squeeze(1)
            end_points['pred_rotation'] = sample_R
            # end_points['pred_rotation'] = Ortho6d2Mat(sample_R[:, :3].contiguous(), sample_R[:, 3:].contiguous()).view(-1,3,3) # bs*3*3

        return end_points


class SupervisedLoss(nn.Module):
    def __init__(self, cfg):
        super(SupervisedLoss, self).__init__()
        self.cfg=cfg.loss
        self.freeze_world_enhancer=cfg.freeze_world_enhancer

    def forward(self, end_points):
        
        shape_pred = end_points['pred_shape']
        shape_truth = end_points['model']

        nocs_shape_pred = end_points['pred_nocs_shape']
        nocs_shape_truth = end_points['qo']

        s0 = end_points['pred_size0']
        t0 = end_points['pred_translation0']
        R0 = end_points['pred_rotation0']

        s = end_points['delta_size0']
        t = end_points['delta_translation0']
        R = end_points['delta_rotation0']

        loss_pose = self._get_loss(s0, t0, R0, s, t, R)
        loss_shape = self._get_loss_shape(shape_pred, shape_truth)
        loss_nocs_shape = self._get_loss_nocs_shape(nocs_shape_pred, nocs_shape_truth)

        return loss_pose + loss_shape + loss_nocs_shape
    
    def _get_loss(self, s0, t0, R0, s1, t1, R1):
        loss_pose = PoseDis(s0, t0, R0, s1, t1, R1)
        return loss_pose

    def _get_loss_shape(self, shape_pred, shape_truth):
        loss_shape = ChamferDis(shape_pred, shape_truth)
        return loss_shape
    
    def _get_loss_nocs_shape(self, nocs_shape_pred, nocs_shape_truth):
        loss_nocs_shape = SmoothL1Dis(nocs_shape_pred, nocs_shape_truth)
        return loss_nocs_shape
