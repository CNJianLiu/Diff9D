"""
Utils related to geometry like projection,,
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import pdb


def sample_textures(texture_flow, images):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    images: B x 3 x N x N

    output: B x F x T x T x 3
    """
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 3 x F x T*T
    samples = torch.nn.functional.grid_sample(images, flow_grid, align_corners=True)
    # B x 3 x F x T x T
    samples = samples.view(-1, 3, F, T, T)
    # B x F x T x T x 3
    return samples.permute(0, 2, 3, 4, 1)

def orthographic_proj(X, cam):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    return scale * X_rot[:, :, :2] + trans

def orthographic_proj_withz(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Orth preserving the z.
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    proj = scale * X_rot

    proj_xy = proj[:, :, :2] + trans
    proj_z = proj[:, :, 2] + offset_z

    return torch.cat((proj_xy, proj_z.unsqueeze(2)), 2)

def projection(xyz, proj_pose, cam, img_size):
    shape_posed = orthographic_proj_withz(xyz, proj_pose, proj_pose[:, 3:4]) # B, N, 3
    xyz_h = torch.cat([shape_posed, torch.ones_like(shape_posed[..., :1])], dim=-1)
    proj_pts = cam.bmm(xyz_h.permute(0, 2, 1)) # B, 4, N
    proj_pts = proj_pts.permute(0, 2, 1)
    pixel_locations = proj_pts[..., :2] / torch.clamp(proj_pts[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
    pixel_locations = torch.clamp(pixel_locations, min=0, max=img_size-1)
    pixel_locations = pixel_locations[:, :, 0]  + pixel_locations[:, :, 1] * img_size
    return pixel_locations

# def orthographic_proj_withz(X, cam, offset_z=0.):
#     """
#     X: B x N x 3
#     cam: B x 7: [sc, tx, ty, quaternions]
#     Orth preserving the z.
#     """
#     quat = cam[:, -4:]
#     X_rot = quat_rotate(X, quat)

#     scale = cam[:, 0].contiguous().view(-1, 1, 1)
#     trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

#     proj = scale * X_rot

#     proj_xy = proj[:, :, :2] + trans
#     proj_z = proj[:, :, 2, None] + offset_z

#     return torch.cat((proj_xy, proj_z), 2)

def cross_product(qa, qb):
    """Cross product of va by vb.

    Args:
        qa: B X N X 3 vectors
        qb: B X N X 3 vectors
    Returns:
        q_mult: B X N X 3 vectors
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    
    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]

    # See https://en.wikipedia.org/wiki/Cross_product
    q_mult_0 = qa_1*qb_2 - qa_2*qb_1
    q_mult_1 = qa_2*qb_0 - qa_0*qb_2
    q_mult_2 = qa_0*qb_1 - qa_1*qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2], dim=-1)


def hamilton_product(qa, qb):
    """Multiply qa by qb.

    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]
    
    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]
    
    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0*qb_0 - qa_1*qb_1 - qa_2*qb_2 - qa_3*qb_3
    q_mult_1 = qa_0*qb_1 + qa_1*qb_0 + qa_2*qb_3 - qa_3*qb_2
    q_mult_2 = qa_0*qb_2 - qa_1*qb_3 + qa_2*qb_0 + qa_3*qb_1
    q_mult_3 = qa_0*qb_3 + qa_1*qb_2 - qa_2*qb_1 + qa_3*qb_0
    
    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)

    
def quat_rotate(X, q):
    """Rotate points by quaternions.

    Args:
        X: B X N X 3 points
        q: B X 4 quaternions

    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    ones_x = X[[0], :, :][:, :, [0]]*0 + 1
    q = torch.unsqueeze(q, 1)*ones_x

    q_conj = torch.cat([ q[:, :, [0]] , -1*q[:, :, 1:4] ], dim=-1)
    X = torch.cat([ X[:, :, [0]]*0, X ], dim=-1)
    
    X_rot = hamilton_product(q, hamilton_product(X, q_conj))
    return X_rot[:, :, 1:4]

def get_K_crop_resize(K, boxes, orig_size, crop_resize):
    """
    Adapted from https://github.com/BerkeleyAutomation/perception/blob/master/perception/camera_intrinsics.py
    Skew is not handled !
    """
    K = K.float()
    boxes = boxes.float()
    new_K = K.clone()

    final_width, final_height = max(crop_resize), min(crop_resize)
    crop_width = boxes[:, 1] - boxes[:, 0]
    crop_height = boxes[:, 3] - boxes[:, 2]
    crop_cj = (boxes[:, 0] + boxes[:, 1]) / 2
    crop_ci = (boxes[:, 2] + boxes[:, 3]) / 2

    # Crop
    cx = K[:, 0, 2] + (crop_width - 1) / 2 - crop_ci
    cy = K[:, 1, 2] + (crop_height - 1) / 2 - crop_cj

    # Resize (upsample)
    center_x = (crop_width - 1) / 2
    center_y = (crop_height - 1) / 2
    orig_cx_diff = cx - center_x
    orig_cy_diff = cy - center_y
    scale_x = final_width / crop_width
    scale_y = final_height / crop_height
    scaled_center_x = (final_width - 1) / 2
    scaled_center_y = (final_height - 1) / 2
    fx = scale_x * K[:, 0, 0]
    fy = scale_y * K[:, 1, 1]
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    new_K[:, 0, 0] = fx
    new_K[:, 1, 1] = fy
    new_K[:, 0, 2] = cx
    new_K[:, 1, 2] = cy
    return new_K