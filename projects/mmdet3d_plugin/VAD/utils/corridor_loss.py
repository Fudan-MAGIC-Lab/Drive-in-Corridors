import math
import mmcv
import torch
from torch import nn as nn
from mmdet.models import weighted_loss
from mmdet.models.builder import LOSSES

import matplotlib.pyplot as plt
import numpy as np

@LOSSES.register_module()
class CorridorMapBoundLoss(nn.Module):
    """Planning constraint to push ego vehicle away from the lane boundary.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        map_thresh (float, optional): confidence threshold to filter map predictions.
        lane_bound_cls_idx (float, optional): lane_boundary class index.
        dis_thresh (float, optional): distance threshold between ego vehicle and lane bound.
        point_cloud_range (list, optional): point cloud range.
    """

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
        map_thresh=0.5,
        lane_bound_cls_idx=2,
        dis_thresh=1.0,
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        perception_detach=False
    ):
        super(CorridorMapBoundLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.map_thresh = map_thresh
        self.lane_bound_cls_idx = lane_bound_cls_idx
        self.dis_thresh = dis_thresh
        self.pc_range = point_cloud_range
        self.perception_detach = perception_detach

    def forward(self,
                ego_fut_preds,
                lane_preds,
                lane_score_preds,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            ego_fut_preds (Tensor): [B, fut_ts, 5]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # if self.perception_detach:
        lane_preds = lane_preds.detach()
        lane_score_preds = lane_score_preds.detach()

        # filter lane element according to confidence score and class
        not_lane_bound_mask = lane_score_preds[..., self.lane_bound_cls_idx] < self.map_thresh
        # denormalize map pts
        lane_bound_preds = lane_preds.clone()
        lane_bound_preds[...,0:1] = (lane_bound_preds[..., 0:1] * (self.pc_range[3] -
                                self.pc_range[0]) + self.pc_range[0])
        lane_bound_preds[...,1:2] = (lane_bound_preds[..., 1:2] * (self.pc_range[4] -
                                self.pc_range[1]) + self.pc_range[1])
        # pad not-lane-boundary cls and low confidence preds
        lane_bound_preds[not_lane_bound_mask] = 1e6

        loss_bbox = self.loss_weight * plan_map_bound_loss(ego_fut_preds, lane_bound_preds,
                                                           weight=weight, dis_thresh=self.dis_thresh,
                                                           reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def plan_map_bound_loss(pred, target, dis_thresh=1.0):
    """Planning map bound constraint (L1 distance).

    Args:
        pred (torch.Tensor): ego_fut_preds, [B, fut_ts, 2].
        target (torch.Tensor): lane_bound_preds, [B, num_vec, num_pts, 2].
        weight (torch.Tensor): [B, fut_ts]

    Returns:
        torch.Tensor: Calculated loss [B, fut_ts]
    """
    B, T, _ = pred.size()
    _, V, P, _ = target.size()
    ego_pred_expanded = pred.unsqueeze(2).unsqueeze(3)  # [B, T, 1, 1, 5]
    maps_expanded = target.unsqueeze(1)  # [B, 1, M, P, 2]
    dist = calculate_distance_to_corridor_boundary(ego_pred_expanded, maps_expanded)  # [B, T, M, P]
    # with torch.no_grad():
    #     plot_corridor_and_map_points(ego_pred_expanded, maps_expanded, dist)
    max_idxs = torch.argmax(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
    max_dist = dist[batch_idxs, ts_idxs, max_idxs]
    loss = max_dist

    return loss

def calculate_distance_to_corridor_boundary(corridor, map_points):
    """
    Calculate distance from map points to the nearest boundary of the corridor.

    Parameters:
        corridor (torch.Tensor): Tensor of shape [B, T, 1, 1, 5] with each entry as (x, y, theta, l, w).
        map_points (torch.Tensor): Tensor of shape [B, 1, M, P, 2] with each entry as (px, py).

    Returns:
        torch.Tensor: Distance tensor of shape [B, T, M, P] representing distance to nearest boundary.
    """
    # Extract parameters
    corridor_center = corridor[..., :2]  # [B, T, 1, 1, 2] - (x, y)
    theta = corridor[..., 2]             # [B, T, 1, 1] - rotation angle
    length = corridor[..., 3] / 2        # [B, T, 1, 1] - half-length
    width = corridor[..., 4] / 2         # [B, T, 1, 1] - half-width

    # Translate map points to corridor center
    translated_points = map_points - corridor_center  # [B, T, M, P, 2] due to broadcasting

    # Rotation matrix (cos, sin) for each corridor
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    R = torch.stack([torch.stack([cos_theta, sin_theta], dim=-1),
                     torch.stack([-sin_theta, cos_theta], dim=-1)], dim=-2)  # [B, T, 1, 1, 2, 2]

    # Rotate map points into corridor's local coordinate system
    local_points = torch.matmul(R, translated_points.unsqueeze(-1)).squeeze(-1)  # [B, T, M, P, 2]

    # Calculate distances to the corridor boundary in the local coordinate system
    dx = torch.abs(local_points[..., 0]) - length  # [B, T, M, P]
    dy = torch.abs(local_points[..., 1]) - width   # [B, T, M, P]

    # Calculate the distance to the nearest boundary (only for points inside the corridor)
    distance_to_boundary = torch.where(
        (dx <= 0) & (dy <= 0),  # Check if the point is inside the corridor
        -torch.maximum(torch.clamp_max(dx, 0), torch.clamp_max(dy, 0)),  # Distance to nearest boundary
        torch.tensor(0.0, device=corridor.device)  # Outside corridor has zero loss
    )
    distance_to_boundary = distance_to_boundary.view(distance_to_boundary.shape[0], distance_to_boundary.shape[1], -1)

    return distance_to_boundary  # Shape [B, T, M*P]

def plot_corridor_and_map_points(corridor, map_points, distance_to_boundary, sample_idx=(0, 0)):
    """
    Plot the corridor, map points, and distances for debugging.
    
    Parameters:
        corridor (torch.Tensor): Tensor of shape [B, T, 1, 1, 5] representing the corridor.
        map_points (torch.Tensor): Tensor of shape [1, 1, M, P, 2] representing map points.
        distance_to_boundary (torch.Tensor): Distance tensor of shape [B, T, M*P].
        sample_idx (tuple): Tuple of (B_idx, T_idx) to select a specific example to plot.
    """
    B_idx, T_idx = sample_idx
    
    # Extract specific sample
    corridor_sample = corridor[B_idx, T_idx, 0, 0]  # [5]
    map_points_sample = map_points[0, 0]            # [M, P, 2]
    distance_sample = distance_to_boundary[B_idx, T_idx]  # [M, P]

    # Extract corridor parameters
    x, y, theta, l, w = corridor_sample.detach().cpu().numpy()
    l, w = l / 2, w / 2  # Convert to half-length and half-width

    # Calculate the rectangle vertices in corridor's local coordinate
    corners = np.array([[-l, -w], [l, -w], [l, w], [-l, w], [-l, -w]])
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta)], 
                                [-np.sin(theta), np.cos(theta)]])
    rotated_corners = np.dot(corners, rotation_matrix) + np.array([x, y])

    # Prepare the plot
    plt.figure(figsize=(10, 10))
    plt.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'b-', linewidth=2, label="Corridor Boundary")

    # Plot map points and annotate distances
    map_points_np = map_points_sample.detach().cpu().numpy().reshape(-1, 2)  # Flattened to [M*P, 2]
    distances_np = distance_sample.detach().cpu().numpy().flatten()  # Flatten to [M*P]

    for (px, py), dist in zip(map_points_np, distances_np):
        if abs(px) > 50 or abs(py) > 15:
            continue
        plt.plot(px, py, 'ro')  # Map point
        plt.text(px, py, f'({px:.2f}, {py:.2f})\nDist: {dist:.2f}', 
                 color='green', fontsize=8, ha='right', va='bottom')   # Annotate distance
    
    # Configure the plot
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Corridor and Map Points (Sample B={B_idx}, T={T_idx})")
    from matplotlib.ticker import MultipleLocator
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))  # 设置X轴网格间距为2
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))  # 设置Y轴网格间距为0.5
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.savefig('vis/corridor_loss.png', dpi=400)


@LOSSES.register_module()
class CorridorCollisionLoss(nn.Module):
    """Planning constraint to push ego vehicle away from other agents.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        agent_thresh (float, optional): confidence threshold to filter agent predictions.
        x_dis_thresh (float, optional): distance threshold between ego and other agents in x-axis.
        y_dis_thresh (float, optional): distance threshold between ego and other agents in y-axis.
        point_cloud_range (list, optional): point cloud range.
    """

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
        agent_thresh=0.5,
        x_dis_thresh=1.5,
        y_dis_thresh=3.0,
        point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    ):
        super(CorridorCollisionLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.agent_thresh = agent_thresh
        self.x_dis_thresh = x_dis_thresh
        self.y_dis_thresh = y_dis_thresh
        self.pc_range = point_cloud_range

    def forward(self,
                ego_fut_preds,
                agent_preds,
                agent_fut_preds,
                agent_score_preds,
                agent_fut_cls_preds,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            ego_fut_preds (Tensor): [B, fut_ts, 5]
            agent_preds (Tensor): [B, num_agent, 10]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_fut_cls_preds (Tensor): [B, num_agent, fut_mode]
            agent_score_preds (Tensor): [B, num_agent, 10]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        # detach
        agent_preds = agent_preds.detach()
        agent_fut_preds = agent_fut_preds.detach()
        agent_score_preds = agent_score_preds.detach()
        agent_fut_cls_preds = agent_fut_cls_preds.detach()

        # filter agent element according to confidence score
        agent_max_score_preds, agent_max_score_idxs = agent_score_preds.max(dim=-1)
        not_valid_agent_mask = agent_max_score_preds < self.agent_thresh
        # filter low confidence preds
        agent_fut_preds[not_valid_agent_mask] = 1e6
        # # filter not vehicle preds
        # not_veh_pred_mask = agent_max_score_idxs > 4  # veh idxs are 0-4
        # agent_fut_preds[not_veh_pred_mask] = 1e6
        # only use best mode pred
        best_mode_idxs = torch.argmax(agent_fut_cls_preds, dim=-1).tolist()
        batch_idxs = [[i] for i in range(agent_fut_cls_preds.shape[0])]
        agent_num_idxs = [[i for i in range(agent_fut_cls_preds.shape[1])] for j in range(agent_fut_cls_preds.shape[0])]
        agent_fut_preds = agent_fut_preds[batch_idxs, agent_num_idxs, best_mode_idxs]

        loss_bbox = self.loss_weight * plan_col_loss(ego_fut_preds, agent_preds,
                                                           agent_fut_preds=agent_fut_preds, weight=weight,
                                                           reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def plan_col_loss(
    pred,
    target,
    agent_fut_preds,
):
    """Planning ego-agent collsion constraint.

    Args:
        pred (torch.Tensor): ego_fut_preds, [B, fut_ts, 5].
        target (torch.Tensor): agent_preds, [B, num_agent, 10].
            x, y, z, x_size, y_size, z_size, yaw, ....
        agent_fut_preds (Tensor): [B, num_agent, fut_ts, 2].
        weight (torch.Tensor): [B, fut_ts].

    Returns:
        torch.Tensor: Calculated loss [B, fut_ts]
    """
    agent_fut_preds = agent_fut_preds.cumsum(dim=-2)
    target_fut = target[:, :, None, :2] + agent_fut_preds
    target_expanded = calculate_vertices_target(target, target_fut) # [B, T, num_agent, 4, 2]
    ego_pred_expanded = pred.unsqueeze(2).unsqueeze(3)  # [B, T, 1, 1, 5]
    dist = calculate_distance_to_corridor_agent(ego_pred_expanded, target_expanded)  # [B, T, num_agent*8]
    # with torch.no_grad():
    #     plot_corridor_and_agent_points(ego_pred_expanded, target_expanded, dist)
    max_idxs = torch.argmax(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
    max_dist = dist[batch_idxs, ts_idxs, max_idxs]
    loss = max_dist

    return loss

def calculate_vertices_target(target, target_fut):
    """
    Calculate vertices of each agent at each future timestep based on its size and yaw.

    Args:
        target (torch.Tensor): Agent predictions, shape [B, num_agent, 10].
        target_fut (torch.Tensor): Future positions of agents, shape [B, num_agent, fut_ts, 2].

    Returns:
        torch.Tensor: Vertices for each agent boundary at each future timestep, shape [B, T, num_agent, 4, 2].
    """
    # Extract agent dimensions and yaw
    x_size = target[:, :, [2]].exp() / 2  # Half w
    y_size = target[:, :, [3]].exp() / 2  # Half l
    sin_yaw = target[:, :, [6]]
    cos_yaw = target[:, :, [7]] 
    # yaw = torch.atan2(sin_yaw, cos_yaw)

    # Define local corners relative to the center
    local_corners = torch.cat([
        torch.cat([-x_size, -y_size], dim=-1).unsqueeze(2),
        torch.cat([ x_size, -y_size], dim=-1).unsqueeze(2),
        torch.cat([ x_size,  y_size], dim=-1).unsqueeze(2),
        torch.cat([-x_size,  y_size], dim=-1).unsqueeze(2),
    ], dim=2)  # Shape [B, num_agent, 4, 2]

    # Compute rotation matrix based on yaw
    rotation_matrix = torch.cat([torch.cat([cos_yaw, -sin_yaw], dim=-1).unsqueeze(2),
                                   torch.cat([sin_yaw, cos_yaw], dim=-1).unsqueeze(2)], dim=2)  # [B, num_agent, 2, 2]

    # Rotate corners based on yaw and then translate to future positions
    rotated_corners = torch.matmul(local_corners, rotation_matrix)  # [B, num_agent, 4, 2]
    vertices = rotated_corners.unsqueeze(-3) + target_fut.unsqueeze(-2)  # [B, num_agent, fut_ts, 4, 2]

    return vertices.permute(0, 2, 1, 3, 4)  # [B, fut_ts, num_agent, 4, 2]

def calculate_distance_to_corridor_agent(corridor, map_points):
    """
    Calculate distance from map points to the nearest boundary of the corridor.

    Parameters:
        corridor (torch.Tensor): Tensor of shape [B, T, 1, 1, 5] with each entry as (x, y, theta, l, w).
        map_points (torch.Tensor): Tensor of shape [1, 1, A, 4, 2] with each entry as (px, py).

    Returns:
        torch.Tensor: Distance tensor of shape [B, T, A*8] representing distance to nearest boundary.
    """
    # Extract parameters
    corridor_center = corridor[..., :2]  # [B, T, 1, 1, 2] - (x, y)
    theta = corridor[..., 2]             # [B, T, 1, 1] - rotation angle
    length = corridor[..., 3] / 2        # [B, T, 1, 1] - half-length
    width = corridor[..., 4] / 2         # [B, T, 1, 1] - half-width

    # Translate map points to corridor center
    translated_points = map_points - corridor_center  # [B, T, A, 4, 2] due to broadcasting

    # Rotation matrix (cos, sin) for each corridor
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    R = torch.stack([torch.stack([cos_theta, sin_theta], dim=-1),
                     torch.stack([-sin_theta, cos_theta], dim=-1)], dim=-2)  # [B, T, 1, 1, 2, 2]

    # Rotate map points into corridor's local coordinate system
    local_points = torch.matmul(R, translated_points.unsqueeze(-1)).squeeze(-1)  # [B, T, M, P, 2]

    # Calculate distances to the corridor boundary in the local coordinate system
    dx = torch.abs(local_points[..., 0]) - length  # [B, T, M, P]
    dy = torch.abs(local_points[..., 1]) - width   # [B, T, M, P]

    # Calculate the distance to the nearest boundary (only for points inside the corridor)
    distance_to_boundary = torch.where(
        (dx <= 0) & (dy <= 0),  # Check if the point is inside the corridor
        -torch.maximum(torch.clamp_max(dx, 0), torch.clamp_max(dy, 0)),  # Distance to nearest boundary
        torch.tensor(0.0, device=corridor.device)  # Outside corridor has zero loss
    )
    distance_to_boundary = distance_to_boundary.view(distance_to_boundary.shape[0], distance_to_boundary.shape[1], -1)

    return distance_to_boundary  # Shape [B, T, M*P]

def plot_corridor_and_agent_points(corridor, agent_points, distance_to_agent, sample_idx=(0, 0)):
    """
    Plot the corridor, map points, and distances for debugging.
    
    Parameters:
        corridor (torch.Tensor): Tensor of shape [B, T, 1, 1, 5] representing the corridor.
        agent_points (torch.Tensor): Tensor of shape [B, T, A, 4, 2] representing map points.
        distance_to_agent (torch.Tensor): Distance tensor of shape [B, T, A*8].
        sample_idx (tuple): Tuple of (B_idx, T_idx) to select a specific example to plot.
    """
    B_idx, T_idx = sample_idx
    
    # Extract specific sample
    corridor_sample = corridor[B_idx, T_idx, 0, 0]  # [5]
    map_points_sample = agent_points[B_idx, T_idx]            # [A, 4, 2]
    distance_sample = distance_to_agent[B_idx, T_idx]  # [A*8]

    # Extract corridor parameters
    x, y, theta, l, w = corridor_sample.detach().cpu().numpy()
    l, w = l / 2, w / 2  # Convert to half-length and half-width

    # Calculate the rectangle vertices in corridor's local coordinate
    corners = np.array([[-l, -w], [l, -w], [l, w], [-l, w], [-l, -w]])
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta)], 
                                [-np.sin(theta), np.cos(theta)]])
    rotated_corners = np.dot(corners, rotation_matrix) + np.array([x, y])

    # Prepare the plot
    plt.figure(figsize=(10, 10))
    plt.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'b-', linewidth=2, label="Corridor Boundary")

    # Plot map points and annotate distances
    map_points_np = map_points_sample.detach().cpu().numpy().reshape(-1, 2)  # Flattened to [M*P, 2]
    distances_np = distance_sample.detach().cpu().numpy().flatten()  # Flatten to [M*P]

    for (px, py), dist in zip(map_points_np, distances_np):
        if abs(px) > 50 or abs(py) > 50:
            continue
        plt.plot(px, py, 'ro')  # Map point
        plt.text(px, py, f'({px:.2f}, {py:.2f})\nDist: {dist:.2f}', 
                 color='green', fontsize=8, ha='right', va='bottom')   # Annotate distance
    
    # Configure the plot
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Corridor and Map Points (Sample B={B_idx}, T={T_idx})")
    from matplotlib.ticker import MultipleLocator
    plt.gca().xaxis.set_major_locator(MultipleLocator(1.0))  # 设置X轴网格间距为2
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))  # 设置Y轴网格间距为0.5
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.savefig('vis/corridor_loss.png', dpi=400)

@LOSSES.register_module()
class CorridorAreaLoss(nn.Module):
    """Constraint to penalize small aeras.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
    ):
        super(CorridorAreaLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight


    def forward(self,
                ego_cor_preds,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_bbox = self.loss_weight * cor_area_loss(ego_cor_preds, ego_cor_preds, weight=weight,
                                                           reduction=reduction, avg_factor=avg_factor)
        return loss_bbox
    
@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def cor_area_loss(
    corridor,
    target
):
    """Planning ego-agent collsion constraint.

    Args:
        pred (torch.Tensor): corridor, [B, fut_ts, 5].
        

    Returns:
        torch.Tensor: Calculated loss [B, fut_ts]
    """
    
    w = corridor[..., 3]  
    l = corridor[..., 4]
    alpha = 0.01
    loss = torch.exp(-alpha*w*l)
    return loss