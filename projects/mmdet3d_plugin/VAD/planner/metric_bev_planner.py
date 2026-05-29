'''
calculate planner metric same as stp3
'''
import numpy as np
import torch
import cv2
import copy
import matplotlib.pyplot as plt
from skimage.draw import polygon
from nuscenes.utils.data_classes import Box
from scipy.spatial.transform import Rotation as R
import shapely
from shapely.geometry import Polygon, LineString
from shapely import affinity

# ego_width, ego_length = 1.85, 4.084  ## nusc
ego_width, ego_length = 1.1485*2, 4.049  ## nuplan, width, front_length

class PlanningMetric():
    def __init__(self):
        super().__init__()
        self.X_BOUND = [-50.0, 50.0, 0.1]  # Forward
        self.Y_BOUND = [-50.0, 50.0, 0.1]  # Sides
        self.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
        dx, bx, _ = self.gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.dx, self.bx = dx[:2], bx[:2]

        bev_resolution, bev_start_position, bev_dimension = self.calculate_birds_eye_view_parameters(
            self.X_BOUND, self.Y_BOUND, self.Z_BOUND
        )
        self.bev_resolution = bev_resolution.numpy()
        self.bev_start_position = bev_start_position.numpy()
        self.bev_dimension = bev_dimension.numpy()

        self.W = ego_width
        self.H = ego_length
        self.res_t = 5  # number of stamps for high-res temporal collision check

        # self.category_index = {    
        #     'human':[2,3,4,5,6,7,8],
        #     'vehicle':[14,15,16,17,18,19,20,21,22,23]
        # }
        # NOTE nuPlan
        self.category_index = {    
            'human':[2],
            'vehicle':[1]
        }

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

        return dx, bx, nx
    
    def calculate_birds_eye_view_parameters(self, x_bounds, y_bounds, z_bounds):
        """
        Parameters
        ----------
            x_bounds: Forward direction in the ego-car.
            y_bounds: Sides
            z_bounds: Height

        Returns
        -------
            bev_resolution: Bird's-eye view bev_resolution
            bev_start_position Bird's-eye view first element
            bev_dimension Bird's-eye view tensor spatial dimension
        """
        bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
        bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
        bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                    dtype=torch.long)

        return bev_resolution, bev_start_position, bev_dimension

    def get_label(
            self,
            gt_agent_boxes,
            gt_agent_feats
        ):
        segmentation_np, pedestrian_np, agent_trajs_np = self.get_birds_eye_view_label(gt_agent_boxes,gt_agent_feats)
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0)
        pedestrian = torch.from_numpy(pedestrian_np).long().unsqueeze(0)
        agent_trajs = torch.from_numpy(agent_trajs_np).unsqueeze(0)
        return segmentation, pedestrian, agent_trajs

    def linear_interpolate(self, prev_value, next_value, alpha):
        assert 0 <= alpha <= 1, 'alpha should be between 0 and 1'
        return prev_value + (next_value - prev_value) * alpha


    def get_birds_eye_view_label(
            self,
            gt_agent_boxes,
            gt_agent_feats
        ):
        '''
        gt_agent_boxes (LiDARInstance3DBoxes): list of GT Bboxs.
            dim 9 = (x,y,z)+(w,l,h)+yaw+(vx,vy)  # local frame 
        gt_agent_feats: (B, A, 34)
            dim 34 = fut_traj(6*2) + fut_mask(6) + goal(1) + lcf_feat(9) + fut_yaw(6)
            lcf_feat (x, y, yaw, vx, vy, width, length, height, type) in the global frame
        ego_lcf_feats: (B, 9) 
            dim 8 = (vx, vy, ax, ay, w, length, width, vel, steer)
        '''
        T = 6
        time_stamps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        segmentation = np.zeros((T,self.bev_dimension[0], self.bev_dimension[1]))
        pedestrian = np.zeros((T,self.bev_dimension[0], self.bev_dimension[1]))
        agent_num = gt_agent_feats.shape[1]

        gt_agent_boxes = gt_agent_boxes.tensor.cpu().numpy()  #(N, 9)
        gt_agent_feats = gt_agent_feats.cpu().numpy()

        gt_agent_fut_trajs = gt_agent_feats[..., :T*2].reshape(-1, 6, 2)
        gt_agent_fut_mask = gt_agent_feats[..., T*2:T*3].reshape(-1, 6)
        # gt_agent_lcf_feat = gt_agent_feats[..., T*3+1:T*3+10].reshape(-1, 9)
        gt_agent_fut_yaw = gt_agent_feats[..., T*3+10:T*4+10].reshape(-1, 6, 1)
        gt_agent_fut_trajs = np.cumsum(gt_agent_fut_trajs, axis=1)
        gt_agent_fut_yaw = np.cumsum(gt_agent_fut_yaw, axis=1)

        gt_agent_boxes[:,6:7] = -1*(gt_agent_boxes[:,6:7] + np.pi/2) # NOTE: convert yaw to lidar frame
        gt_agent_fut_trajs = gt_agent_fut_trajs + gt_agent_boxes[:, np.newaxis, 0:2]
        gt_agent_fut_yaw = gt_agent_fut_yaw + gt_agent_boxes[:, np.newaxis, 6:7]
        
        gt_agent_fut_shape = gt_agent_boxes[:, np.newaxis, 3:5]
        gt_agent_fut_shape = np.repeat(gt_agent_fut_shape, 6, axis=1)
        gt_agent_fut_polys_t0 = gt_agent_boxes[:, np.newaxis, [0,1,6,3,4]] # t=0 , A, 1, 5
        gt_agent_fut_polys = np.concatenate((gt_agent_fut_trajs, gt_agent_fut_yaw, gt_agent_fut_shape), axis=2) # A, 6, 5
        gt_agent_fut_polys = np.concatenate((gt_agent_fut_polys_t0, gt_agent_fut_polys), axis=1)

        for t in range(T):
            for i in range(agent_num):
                if gt_agent_fut_mask[i][t] == 1:
                    # Filter out all non vehicle instances
                    category_index = int(gt_agent_feats[0,i][27])
                    agent_length, agent_width = gt_agent_boxes[i][4], gt_agent_boxes[i][3]
                    x_a = gt_agent_fut_trajs[i, t, 0]
                    y_a = gt_agent_fut_trajs[i, t, 1]
                    yaw_a = gt_agent_fut_yaw[i, t, 0]
                    param = [x_a,y_a,yaw_a,agent_length, agent_width]
                    if (category_index in self.category_index['vehicle']):
                        poly_region = self._get_poly_region_in_image(param)
                        segmentation[t] = cv2.fillPoly(segmentation[t], [poly_region], 1.0)
                    if (category_index in self.category_index['human']):
                        poly_region = self._get_poly_region_in_image(param)
                        cv2.fillPoly(pedestrian[t], [poly_region], 1.0)
        
        return segmentation, pedestrian, gt_agent_fut_polys
        

    def _get_poly_region_in_image(self,param):
        # lidar2cv_rot = np.array([[1,0], [0,-1]]) #NOTE whatever it is
        lidar2cv_rot = np.array([[1,0], [0,1]])
        x_a,y_a,yaw_a,agent_length, agent_width = param
        trans_a = np.array([[x_a,y_a]]).T
        rot_mat_a = np.array([[np.cos(yaw_a), -np.sin(yaw_a)],
                                [np.sin(yaw_a), np.cos(yaw_a)]])
        agent_corner = np.array([
            [agent_length/2, -agent_length/2, -agent_length/2, agent_length/2],
            [agent_width/2, agent_width/2, -agent_width/2, -agent_width/2]]) #(2,4)
        agent_corner_lidar = np.matmul(rot_mat_a, agent_corner) + trans_a #(2,4)
        # convert to cv frame
        agent_corner_cv2 = (np.matmul(lidar2cv_rot, agent_corner_lidar) \
            - self.bev_start_position[:2,None] + self.bev_resolution[:2,None] / 2.0).T / self.bev_resolution[:2] #(4,2)
        agent_corner_cv2 = np.round(agent_corner_cv2).astype(np.int32)

        return agent_corner_cv2


    def evaluate_single_coll(self, traj, segmentation, input_gt, gt_traj=None, index=None):
        '''
        traj: torch.Tensor (n_future, 2)
            自车IMU系为轨迹参考系

                0------->
                |        x
                |
                |y
                
        segmentation: torch.Tensor (n_future, 200, 200)
        '''
        # transform all to lidar frame

        # 0.985793 is the distance betweem the LiDAR and the IMU(ego).
        # width=1.1485 * 2.0, self.W
        # front_length=4.049, self.H
        # rear_length=1.127
        import mmcv
        pts = np.array([
            [-1.127, self.W / 2.],
            [self.H, self.W / 2.],
            [self.H, -self.W / 2.],
            [-1.127, -self.W / 2.],
        ])
        pts = (pts - self.bx.cpu().numpy() ) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:,1], pts[:,0])
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)
        rc_ori = rc + (self.bx.cpu().numpy() / self.dx.cpu().numpy())


        traj_with_ego = torch.cat([traj.new_zeros(1, 2), traj], 0)
        rc_yaw = []
        rotate_angle = 0
        for i in range(traj.size(0)):
            delta = traj_with_ego[i+1] - traj_with_ego[i]
            cur_rotate_angle = torch.atan2(*delta[[1, 0]])
            if delta.norm()<1: cur_rotate_angle = 0
            rotate_angle = cur_rotate_angle
            rotate_angle = -torch.tensor(rotate_angle) 
            rot_sin = torch.sin(rotate_angle)
            rot_cos = torch.cos(rotate_angle)
            rot_mat = torch.Tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
            tmp = rc_ori @ rot_mat.cpu().numpy() -  (self.bx.cpu().numpy() / self.dx.cpu().numpy())
            tmp = tmp.round().astype(np.int)
            rc_yaw.append(tmp)
           
        rc_yaw = np.stack(rc_yaw)

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)

        trajs_ = copy.deepcopy(trajs)
        trajs_ = trajs_ / self.dx.to(trajs.device)
        trajs_ = trajs_.cpu().numpy() + rc_yaw # (n_future, 32, 2)

        r = trajs_[:,:,0].astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs_[:,:,1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision2 = np.full(n_future, False)
        # obs_occ = copy.deepcopy(segmentation).cpu().numpy() * 0
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
           
            collision2[t] = np.any(segmentation[t,  cc[I], rr[I]].cpu().numpy())      
        return torch.from_numpy(collision2).to(device=traj.device)
    
    def debug(self, traj, segmentation, map_segmentation):
        '''
        traj: torch.Tensor (n_future, 2)
            自车IMU系为轨迹参考系

                0------->
                |        x
                |
                |y
                
        segmentation: torch.Tensor (n_future, 200, 200)
        '''
        # transform all to lidar frame

        # 0.985793 is the distance betweem the LiDAR and the IMU(ego).
        # width=1.1485 * 2.0, self.W
        # front_length=4.049, self.H
        # rear_length=1.127
        import mmcv
        pts = np.array([
            [-1.127, self.W / 2.],
            [self.H, self.W / 2.],
            [self.H, -self.W / 2.],
            [-1.127, -self.W / 2.],
        ])
        pts = (pts - self.bx.cpu().numpy() ) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:,1], pts[:,0])
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)
        rc_ori = rc + (self.bx.cpu().numpy() / self.dx.cpu().numpy())


        traj_with_ego = torch.cat([traj.new_zeros(1, 2), traj], 0)
        rc_yaw = []
        rotate_angle = 0
        for i in range(traj.size(0)):
            delta = traj_with_ego[i+1] - traj_with_ego[i]
            cur_rotate_angle = torch.atan2(*delta[[1, 0]])
            if delta.norm()<1: cur_rotate_angle = 0
            rotate_angle = cur_rotate_angle
            rotate_angle = -torch.tensor(rotate_angle) 
            rot_sin = torch.sin(rotate_angle)
            rot_cos = torch.cos(rotate_angle)
            rot_mat = torch.Tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
            tmp = rc_ori @ rot_mat.cpu().numpy() -  (self.bx.cpu().numpy() / self.dx.cpu().numpy())
            tmp = tmp.round().astype(np.int)
            rc_yaw.append(tmp)
           
        rc_yaw = np.stack(rc_yaw)

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)

        trajs_ = copy.deepcopy(trajs)
        trajs_ = trajs_ / self.dx.to(trajs.device)
        trajs_ = trajs_.cpu().numpy() + rc_yaw # (n_future, 32, 2)

        r = trajs_[:,:,0].astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs_[:,:,1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision = np.full(n_future, False)
        collision_map = np.full(n_future, False)
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
           
            collision[t] = np.any(segmentation[t,  cc[I], rr[I]].cpu().numpy())   # cv frame 
            collision_map[t] = np.any(map_segmentation[t,  cc[I], rr[I]].cpu().numpy())
        
        '''
        # debug plot
        if n_future == 6:
            # fig, axes = plt.subplots(1, n_future, figsize=(20, 5))
            # axes.axis('off')
            for t in range(n_future):
                fig, axes = plt.subplots(figsize=(5, 5))
                img = segmentation[t].cpu().numpy()
                img_map = map_segmentation[t].cpu().numpy()
                color_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                color_img[img == 0] = [50, 50, 50]  # 灰色背景
                color_img[img == 1] = [255, 255, 255]  # 白色线条
                color_img[img_map == 1] = [200, 200, 200]
                 # 绘制轨迹曲线
                trajs_plot = (traj.cpu().numpy() - self.bx.cpu().numpy()) / self.dx.cpu().numpy()
                traj_x, traj_y = trajs_plot[:, 0], trajs_plot[:, 1]
                axes.plot(traj_x, traj_y, color='green', alpha=0.5, linewidth=2)  # 绿色轨迹线
                rr = r[t]
                cc = c[t]
                color_img[cc, rr] = [255, 0, 0]  # 红色车辆多边形
                axes.imshow(color_img, cmap='gray')
                axes.set_title(f'Time step {t}')
                # axes[t].axis('off')
                axes.grid(False)
                axes.invert_yaxis()
                plt.savefig('segmentation_with_trajectory'+str(t)+'.png')
        print('\nc1:', collision)
        print('c2_map:', collision_map)
        '''
        
        return torch.from_numpy(collision).to(device=traj.device), torch.from_numpy(collision_map).to(device=traj.device)

    def debug_fine(self, traj, agent_trajs, map_boundaries):
        pts = np.array([
            [-1.127, self.W / 2.],
            [self.H, self.W / 2.],
            [self.H, -self.W / 2.],
            [-1.127, -self.W / 2.],
        ])
        base_polygon = Polygon(pts)

        traj_with_ego = torch.cat([traj.new_zeros(1, 2), traj], 0)
        agent_trajs = agent_trajs[0].numpy()
        # agent_trajs = agent_trajs[:1]  # debug the first
        n_future = traj.shape[0]
        A, _, _ = agent_trajs.shape
        agent_polys = []
        for i in range(A):
            w, l = agent_trajs[i, 0, 3:5]
            pts = np.array([
                [-l/2, w/2],
                [l/2, w/2],
                [l/2, -w/2],
                [-l/2, -w/2],
            ])
            polygon_agent = Polygon(pts)
            agent_polys.append(polygon_agent)
        
        collision2 = np.full(n_future, False)
        collision2_map = np.full(n_future, False)
        yaw0 = 0.0

        for t in range(traj.size(0)):
            # fig, ax = plt.subplots(figsize=(4, 2))
            plt.xlim(xmin=-10, xmax=30)
            plt.ylim(ymin=-10, ymax=10)
            delta = traj_with_ego[t+1] - traj_with_ego[t]
            yaw1 = torch.atan2(*delta[[1, 0]])
            cur_rotate_angle = yaw1 - yaw0
            if delta.norm()<0.5: 
                cur_rotate_angle = 0
            for j in range(1, self.res_t+1):
                center_p = delta*j/self.res_t + traj_with_ego[t]
                center_yaw = yaw0 + cur_rotate_angle*j/self.res_t
                ego_polygon = affinity.rotate(base_polygon, np.degrees(center_yaw), origin=(0, 0))
                ego_polygon = affinity.translate(ego_polygon, xoff=center_p[0].item(), yoff=center_p[1].item())
                # ax.plot(*ego_polygon.exterior.xy, color='blue')
                # agents
                agent_p = self.linear_interpolate(agent_trajs[:, t, :2], agent_trajs[:, t+1, :2], j/self.res_t)
                agent_yaw = self.linear_interpolate(agent_trajs[:, t, 2], agent_trajs[:, t+1, 2], j/self.res_t)
                for a, agent_poly in enumerate(agent_polys):
                    agent_poly_traj = affinity.rotate(agent_poly, np.degrees(agent_yaw[a]), origin=(0, 0))
                    agent_poly_traj = affinity.translate(agent_poly_traj, xoff=agent_p[a, 0].item(), yoff=agent_p[a, 1].item())
                    # ax.plot(*agent_poly_traj.exterior.xy, color='red')
                    if ego_polygon.intersects(agent_poly_traj):
                        collision2[t] = True
                # map
                for boundary in map_boundaries:
                    # ax.plot(*boundary.xy, color='green')
                    if ego_polygon.intersects(boundary):
                        collision2_map[t] = True
                # traj
                trajs_plot = traj.cpu()
                traj_x, traj_y = trajs_plot[:, 0], trajs_plot[:, 1]
                # ax.plot(traj_x, traj_y, color='orange') 
            if delta.norm()>=0.5:  # update yaw
                yaw0 = yaw1
            # ax.set_title(f'Time step {t}')
            # ax.set_aspect('equal')
            # plt.savefig('segmentation_polygons'+str(t)+'.png')

        # print('c3:', collision2)
        # print('c4_map:', collision2_map)
        return torch.from_numpy(collision2).to(device=traj.device), torch.from_numpy(collision2_map).to(device=traj.device)
    
    def evaluate_single_coll_fine(self, traj, agent_trajs):
        '''
        traj: torch.Tensor (n_future, 2)
            自车IMU系为轨迹参考系

                0------->
                |        x
                |
                |y
                
        agent_trajs: torch.Tensor  (A, n_future+1,  5) x, y, yaw, w, l
        '''

        # nuplan vehicle params
        # width=1.1485 * 2.0, self.W
        # front_length=4.049, self.H
        # rear_length=1.127
        pts = np.array([
            [-1.127, self.W / 2.],
            [self.H, self.W / 2.],
            [self.H, -self.W / 2.],
            [-1.127, -self.W / 2.],
        ])
        base_polygon = Polygon(pts)

        traj_with_ego = torch.cat([traj.new_zeros(1, 2), traj], 0)
        agent_trajs = agent_trajs.numpy()
        n_future = traj.shape[0]
        A, _, _ = agent_trajs.shape
        agent_polys = []
        for i in range(A):
            w, l = agent_trajs[i, 0, 3:5]
            pts = np.array([
                [-l/2, w/2],
                [l/2, w/2],
                [l/2, -w/2],
                [-l/2, -w/2],
            ])
            polygon_agent = Polygon(pts)
            agent_polys.append(polygon_agent)
        
        collision2 = np.full(n_future, False)
        yaw0 = 0.0
        for t in range(traj.size(0)):
            delta = traj_with_ego[t+1] - traj_with_ego[t]
            yaw1 = torch.atan2(*delta[[1, 0]])
            cur_rotate_angle = yaw1 - yaw0
            if delta.norm()<0.5: 
                cur_rotate_angle = 0
            for j in range(1, self.res_t+1):
                center_p = delta*j/self.res_t + traj_with_ego[t]
                center_yaw = yaw0 + cur_rotate_angle*j/self.res_t
                ego_polygon = affinity.rotate(base_polygon, np.degrees(center_yaw), origin=(0, 0))
                ego_polygon = affinity.translate(ego_polygon, xoff=center_p[0].item(), yoff=center_p[1].item())
                # agents
                agent_p = self.linear_interpolate(agent_trajs[:, t, :2], agent_trajs[:, t+1, :2], j/self.res_t)
                agent_yaw = self.linear_interpolate(agent_trajs[:, t, 2], agent_trajs[:, t+1, 2], j/self.res_t)
                for a, agent_poly in enumerate(agent_polys):
                    agent_poly_traj = affinity.rotate(agent_poly, np.degrees(agent_yaw[a]), origin=(0, 0))
                    agent_poly_traj = affinity.translate(agent_poly_traj, xoff=agent_p[a, 0].item(), yoff=agent_p[a, 1].item())
                    if ego_polygon.intersects(agent_poly_traj):
                        collision2[t] = True
            if delta.norm()>=0.5:  # update yaw
                yaw0 = yaw1

                   
        return torch.from_numpy(collision2).to(device=traj.device)

    def evaluate_coll(
            self, 
            trajs, 
            gt_trajs, 
            segmentation,
            index=None,
            ignore_gt=True,
        ):
        '''
        trajs: torch.Tensor (B, n_future, 2)
        自车IMU系为轨迹参考系

                0------->
                |        x
                |
                |y
        gt_trajs: torch.Tensor (B, n_future, 2)
        segmentation: torch.Tensor (B, n_future, 200, 200)

        '''
        #前n_future的轨迹
        B, n_future, _ = trajs.shape
        # trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        # gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(B):
            # whether gt collide, usually false
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i].cpu(), segmentation[i], input_gt=True)

            xx, yy = trajs[i,:,0], trajs[i, :, 1]

            xi = ((-self.bx[0] + xx) / self.dx[0]).long()
            yi = ((-self.bx[1] + yy) / self.dx[1]).long()

            # whether in bev range
            m1 = torch.logical_and(
                torch.logical_and(xi >= 0, xi < self.bev_dimension[0]),
                torch.logical_and(yi >= 0, yi < self.bev_dimension[1]),
            ).to(gt_box_coll.device)
            # in bev and not collide, indicating valid
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))

            # time steps
            ti = torch.arange(n_future).to(segmentation.device)
            # gt collide for obj_coll_sum
            # segmentation: B, T, H, W
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1], xi[m1]].long()   #this is for center position

            # gt not collide
            m2 = torch.logical_not(gt_box_coll)
            box_coll = self.evaluate_single_coll(trajs[i],    # this if for the whole rectangle
                    segmentation[i],
                    gt_traj=gt_trajs[i],
                    input_gt=False
                    ).to(segmentation.device)
            if ignore_gt:
                # obj_box_coll_sum += (gt_box_coll).long()                
                obj_box_coll_sum += (box_coll).long()                
            else:
                obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()
        return obj_coll_sum, obj_box_coll_sum
    
    def evaluate_coll_fine(
            self, 
            trajs, 
            gt_trajs, 
            agent_trajs,
            ignore_gt=True,
        ):
        '''
        trajs: torch.Tensor (B, n_future, 2)
        自车IMU系为轨迹参考系

                0------->
                |        x
                |
                |y
        gt_trajs: torch.Tensor (B, n_future, 2)
        agent_trajs: (B=1, A, n_future,  5) x, y, yaw, w, l

        '''
        #前n_future的轨迹
        B, n_future, _ = trajs.shape

        obj_box_coll_sum = torch.zeros(n_future, device=agent_trajs.device)

        for i in range(B):
            # whether gt collide, usually false
            gt_box_coll = self.evaluate_single_coll_fine(gt_trajs[i].cpu(), agent_trajs[i])

            # time steps
            ti = torch.arange(n_future).to(agent_trajs.device)

            # gt not collide
            m2 = torch.logical_not(gt_box_coll)
            box_coll = self.evaluate_single_coll_fine(trajs[i],    # this if for the whole rectangle
                    agent_trajs[i]
                    ).to(agent_trajs.device)
            if ignore_gt:
                # obj_box_coll_sum += (gt_box_coll).long()     
                obj_box_coll_sum += box_coll.long()   
            else:
                obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()
        return obj_box_coll_sum

    def evaluate_single_map_fine(self, traj, map_boundaries):
        '''
        traj: torch.Tensor (n_future, 2)
        agent_trajs: torch.Tensor  (A, n_future+1,  5) x, y, yaw, w, l
        '''

        # nuplan vehicle params
        # width=1.1485 * 2.0, self.W
        # front_length=4.049, self.H
        # rear_length=1.127
        pts = np.array([
            [-1.127, self.W / 2.],
            [self.H, self.W / 2.],
            [self.H, -self.W / 2.],
            [-1.127, -self.W / 2.],
        ])
        base_polygon = Polygon(pts)

        traj_with_ego = torch.cat([traj.new_zeros(1, 2), traj], 0)
        n_future = traj.shape[0]
        
        collision2 = np.full(n_future, False)
        yaw0 = 0.0
        for t in range(traj.size(0)):
            delta = traj_with_ego[t+1] - traj_with_ego[t]
            yaw1 = torch.atan2(*delta[[1, 0]])
            cur_rotate_angle = yaw1 - yaw0
            if delta.norm()<0.5: 
                cur_rotate_angle = 0
            for j in range(1, self.res_t+1):
                center_p = delta*j/self.res_t + traj_with_ego[t]
                center_yaw = yaw0 + cur_rotate_angle*j/self.res_t
                ego_polygon = affinity.rotate(base_polygon, np.degrees(center_yaw), origin=(0, 0))
                ego_polygon = affinity.translate(ego_polygon, xoff=center_p[0].item(), yoff=center_p[1].item())
                # map
                for boundary in map_boundaries:
                    if ego_polygon.intersects(boundary):
                        collision2[t] = True
            if delta.norm()>=0.5:  # update yaw
                yaw0 = yaw1

                   
        return torch.from_numpy(collision2).to(device=traj.device)

    def evaluate_map_fine(
            self, 
            trajs, 
            gt_trajs, 
            map_boundaries,
            ignore_gt=True,
        ):
        '''
        trajs: torch.Tensor (B, n_future, 2)
        map_boundaries: List[shapely.LineString]

        '''
        #前n_future的轨迹
        B, n_future, _ = trajs.shape

        obj_map_coll_sum = torch.zeros(n_future, device=trajs.device)

        for i in range(B):
            # whether gt collide, usually false
            gt_map_coll = self.evaluate_single_map_fine(gt_trajs[i].cpu(), map_boundaries)

            # time steps
            ti = torch.arange(n_future).to(trajs.device)

            # gt not collide
            m2 = torch.logical_not(gt_map_coll)
            map_coll = self.evaluate_single_map_fine(trajs[i],    # this if for the whole rectangle
                    map_boundaries
                    ).to(trajs.device)
            if ignore_gt:
                # obj_map_coll_sum += (gt_map_coll).long()    
                obj_map_coll_sum += (map_coll).long()    
            else:
                obj_map_coll_sum[ti[m2]] += (map_coll[ti[m2]]).long()
        return obj_map_coll_sum
    

    def compute_L2(self, trajs, gt_trajs):
        '''
        trajs: torch.Tensor (n_future, 2)
        gt_trajs: torch.Tensor (n_future, 2)
        '''
        # return torch.sqrt(((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2).sum(dim=-1))
        pred_len = trajs.shape[0]
        ade = float(
            sum(
                torch.sqrt(
                    (trajs[i, 0] - gt_trajs[i, 0]) ** 2
                    + (trajs[i, 1] - gt_trajs[i, 1]) ** 2
                )
                for i in range(pred_len)
            )
            / pred_len
        )
        
        return ade
    

    def get_map_label(self, segmentation, gt_map_bbox, gt_map_label):
        segmentation_plus = segmentation[0].permute(1, 2, 0).cpu().clone().numpy()  # batch_size = 1
        segmentation_plus *= 0 # only consider boudnary, temporal
        map_gt_bboxes_3d = gt_map_bbox.fixed_num_sampled_points
        map_gt_bboxes_3d= map_gt_bboxes_3d[gt_map_label.data==2]
        # LineString List
        map_points_np = map_gt_bboxes_3d.cpu().numpy()
        line_list = []
        for map_instance in map_points_np:
            line_list.append(LineString(map_instance))
        map_gt_bboxes_3d = (map_gt_bboxes_3d - self.bx.cpu().numpy() ) / (self.dx.cpu().numpy())
        a = segmentation_plus[:, :, :3].copy()
        a = np.ascontiguousarray(a, dtype=np.uint8)
        b = segmentation_plus[:, :, :3].copy()
        b = np.ascontiguousarray(a, dtype=np.uint8)
        for line in map_gt_bboxes_3d:
            line = line.clip(0, 999).numpy().astype(np.int32)
            for i, corner in enumerate(line[:-1]):
                a = cv2.line(a, tuple(line[i]), tuple(line[i+1]), color=(1, 1, 1), thickness=1)
                b = cv2.line(b, tuple(line[i]), tuple(line[i+1]), color=(1, 1, 1), thickness=1)   
        segmentation_plus = torch.cat([torch.tensor(a), torch.tensor(b)], -1).permute(2, 0, 1).unsqueeze(0)
        return segmentation_plus, line_list

        