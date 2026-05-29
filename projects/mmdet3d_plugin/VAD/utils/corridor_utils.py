import torch

corridor_w_range = 18
corridor_l_range = 32
# vecotr_map_corridor.py line 461:
# corridor_range=[7.5, -7.5, 15, -15])

def normalize_2d_bbox_corridor(bboxes, pc_range):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    theta = bboxes[..., 2:3]
    w = bboxes[..., 3:4]
    l = bboxes[..., 4:5]

    # patch_x = pc_range[3]-pc_range[0]
    # patch_y = pc_range[4]-pc_range[1]
    # normalized_cx = (cx - pc_range[0]) / patch_x
    # normalized_cy = (cy - pc_range[1]) / patch_y

    # normalized_w = w / corridor_w_range
    # normalized_l = l / corridor_l_range

    normalized_bboxes = torch.cat(
            # (patch_x, patch_y, theta.sin(), theta.cos(), normalized_w, normalized_l), dim=-1
            (cx, cy, theta.sin(), theta.cos(), w, l), dim=-1
        )
    
    return normalized_bboxes


def denormalize_2d_bbox_corridor(bboxes, pc_range):
    cx = (bboxes[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    cy = (bboxes[..., 1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    
    # rotation 
    rot_sine = bboxes[..., 2:3]
    rot_cosine = bboxes[..., 3:4]

    w = bboxes[..., 4:5] * corridor_w_range
    l = bboxes[..., 5:6] * corridor_l_range

    denormalized_bboxes = torch.cat(
            (cx, cy, rot_sine, rot_cosine, w, l), dim=-1
        )

    return denormalized_bboxes

def to_real_corridor(bboxes, pc_range):
    cx = (bboxes[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    cy = (bboxes[..., 1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    
    # rotation 
    rot_sine = bboxes[..., 2:3]
    rot_cosine = bboxes[..., 3:4]
    rot = torch.atan2(rot_sine, rot_cosine)

    w = bboxes[..., 4:5] * corridor_w_range
    l = bboxes[..., 5:6] * corridor_l_range

    denormalized_bboxes = torch.cat(
            (cx, cy, rot, w, l), dim=-1
        )

    return denormalized_bboxes