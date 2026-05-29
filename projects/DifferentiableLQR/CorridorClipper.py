import torch
import matplotlib.pyplot as plt
from tools.data_converter.simple_corridor_generator_jit import RectangleInflator
import numpy as np

class CorridorClipper():
    def __init__(self,
        map_thresh=0.5,
        agent_thresh=0.5,
        lane_bound_cls_idx=2,
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                 ):
        self.map_thresh = map_thresh
        self.agent_thresh = agent_thresh
        self.lane_bound_cls_idx = lane_bound_cls_idx
        self.pc_range = point_cloud_range

        corridor_range=[7.5, -7.5, 15, -15]
        self.inflator = RectangleInflator(corridor_range)

    def ProcessCorridors(self, 
                        corridors, 
                        traj, 
                        lane_preds, 
                        lane_score_preds, 
                        agent_preds,
                        agent_fut_preds,
                        agent_score_preds,
                        agent_fut_cls_preds):
        '''
            gt_corridors (Tensor): [B, fut_ts, 5]
            traj (Tensor): [B, fut_ts, 2]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            agent_preds (Tensor): [B, num_agent, 10]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_score_preds (Tensor): [B, num_agent, 10]
            agent_fut_cls_preds (Tensor): [B, num_agent, fut_mode]
        '''
        B, T, _ = corridors.size()
        assert B == 1, 'only used in test mode with batchsize 1'

        ### map ###
        not_lane_bound_mask = lane_score_preds[..., self.lane_bound_cls_idx] < self.map_thresh
        lane_bound_preds = lane_preds.clone()
        lane_bound_preds[...,0:1] = (lane_bound_preds[..., 0:1] * (self.pc_range[3] -
                                self.pc_range[0]) + self.pc_range[0])
        lane_bound_preds[...,1:2] = (lane_bound_preds[..., 1:2] * (self.pc_range[4] -
                                self.pc_range[1]) + self.pc_range[1])
        # pad not-lane-boundary cls and low confidence preds
        # lane_bound_preds[not_lane_bound_mask] = 1e6 # [B, num_vec, num_pts, 2]
        lane_bound_mask = ~not_lane_bound_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, lane_bound_preds.size(2), lane_bound_preds.size(3))
        lane_bound_preds = lane_bound_preds[lane_bound_mask].view(lane_bound_preds.size(0), -1, lane_bound_preds.size(2), lane_bound_preds.size(3))  # [B, num_bound, num_pts(20), 2]
        map_points = lane_bound_preds.unsqueeze(1)  # [1, 1, M, P, 2]

        ### agent ###
        # filter agent element according to confidence score
        # # filter not vehicle preds
        # not_veh_pred_mask = agent_max_score_idxs > 4  # veh idxs are 0-4
        # agent_fut_preds[not_veh_pred_mask] = 1e6
        # only use best mode pred
        best_mode_idxs = torch.argmax(agent_fut_cls_preds, dim=-1).tolist()
        batch_idxs = [[i] for i in range(agent_fut_cls_preds.shape[0])]
        agent_num_idxs = [[i for i in range(agent_fut_cls_preds.shape[1])] for j in range(agent_fut_cls_preds.shape[0])]
        agent_fut_preds = agent_fut_preds[batch_idxs, agent_num_idxs, best_mode_idxs]
        
        # filter low confidence preds
        agent_max_score_preds, agent_max_score_idxs = agent_score_preds.max(dim=-1)
        not_valid_agent_mask = agent_max_score_preds < self.agent_thresh
        # agent_fut_preds[not_valid_agent_mask] = 1e6
        valid_agent_mask = ~not_valid_agent_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, agent_fut_preds.size(2), agent_fut_preds.size(3))
        agent_fut_preds = agent_fut_preds[valid_agent_mask].view(agent_fut_preds.size(0), -1, agent_fut_preds.size(2), agent_fut_preds.size(3))  # [B, num_agents, T(6), 2]
        agent_fut_preds = agent_fut_preds.cumsum(dim=-2)
        
        valid_agent_mask = ~not_valid_agent_mask.unsqueeze(-1).expand(-1, -1, agent_preds.size(2))
        agent_preds = agent_preds[valid_agent_mask].view(agent_preds.size(0), -1, agent_preds.size(2))  # [B, num_agents, 10]
        target_fut = agent_preds[:, :, None, :2] + agent_fut_preds
        agent_points = self.calculate_vertices_target(agent_preds, target_fut)  #[B, T, num_agent, 4, 2]
        
        map_points_all = map_points.view(B, 1, -1, 2).repeat(1, T, 1, 1)
        agent_points_all = agent_points.reshape(B, T, -1, 2)
        obs_points = torch.cat([map_points_all, agent_points_all], dim=-2)  # [B, T, n_obs, 2]
        
        obs_np = obs_points.cpu().numpy() 
        corridors_np = corridors.cpu().numpy()  # [B, T, 5]
        processed_corridors_list = []

        for i in range(B):
            batch_corridors = []
            for j in range(T):
                obs_points = obs_np[i, j]
                corridor_init = corridors_np[i, j]
                seed_p = corridor_init[:2]
                seed_yaw = corridor_init[2]
                size_x = corridor_init[3]
                size_y = corridor_init[4]
                if j <= 5:  # 1 for 1s, 3 for 2s
                    corridor = self.inflator.inflateRectangleBound(obs_points, seed_p, seed_yaw, debug=False, 
                                                          x_max=size_x/2, x_min=-size_x/2, y_max=size_y/2, y_min=-size_y/2) # List, len = 5
                else:
                    corridor = self.inflator.inflateRectangle(obs_points, seed_p, seed_yaw, debug=False)
                corridor_np = np.array(corridor, dtype=np.float32)
                batch_corridors.append(corridor_np)
            processed_corridors_list.append(batch_corridors)

        # 将列表转换为张量
        processed_corridors_np = np.array(processed_corridors_list, dtype=np.float32)  # [B, T, 5]
        processed_corridors = torch.tensor(processed_corridors_np, device=corridors.device, dtype=corridors.dtype)

        return processed_corridors # [B, T, 5]

    def calculate_vertices_target(self, target, target_fut):
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