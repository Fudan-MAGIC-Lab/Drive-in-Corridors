import numpy as np
import math
import matplotlib.pyplot as plt
import time
from numba import jit
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box, MultiLineString
# from mpl_toolkits.mplot3d import Axes3D

class RectangleInflator:
    def __init__(self, range = [20.0, -20.0, 10.0, -10.0]): 
        # inflate boundaries
        self.x_max = range[0]  # length, x
        self.x_min = range[1]
        self.y_max = range[2]  # width, y
        self.y_min = range[3]

    def coverEgoRectangle(self, RR):
        ego_xm = -0.93  # 0.93
        ego_xp = 0.93
        ego_ym = -1.45
        ego_yp = 2.62
        xm, xp, ym, yp = RR[:4]
        ret = (xm < ego_xm) and (xp > ego_xp) and (ym < ego_ym) and (yp > ego_yp)
        return ret
    

    def inflateRectangle(self, obstacles: np.array, obs_geo, seed_p: np.array, seed_yaw: float, debug: bool =False) -> np.array:
        # obstacles: n * 2
        R = np.array([[math.cos(seed_yaw), -math.sin(seed_yaw)],
                    [math.sin(seed_yaw), math.cos(seed_yaw)]])
        projected_points = R.T @ (obstacles.T - seed_p[:,np.newaxis])
        projected_points = projected_points.T
        pts = np.empty((0, 2))
        for point in projected_points:
            if self.x_min < point[0] < self.x_max and self.y_min < point[1] < self.y_max:
                pts = np.vstack([pts, point])
                
        '''
            find the largest empty rectangle containing zero point
            Based on 'On the maximum empty rectangle problem, A.Naamad'
        '''
        # RR: np.array([x_min, x_max, y_min, y_max, S])
        X_MIN, X_MAX, Y_MIN, Y_MAX, S = 0, 1, 2, 3, 4
        start_time = time.time()
        RRs = []
        RRs = self._findType1RR(pts, self.x_max, self.x_min, self.y_max, self.y_min)
        RRs += self._findType2RR(pts, self.x_max, self.x_min, self.y_max, self.y_min)
        RRs += self._findType3RR(pts, self.x_max, self.x_min, self.y_max, self.y_min)
        RRs += self._findType4RR(pts, self.x_max, self.x_min, self.y_max, self.y_min)

        sorted_RR = sorted(RRs, key=lambda rr: rr[S])
        sorted_RR.reverse()

        try:
            RR_max = None
            for RR in sorted_RR:
                if self.coverEgoRectangle(RR):
                    RR_max = RR
                    break
            assert RR_max is not None
            RR = RR_max
        except:
            # return box 0 
            
            # debug: plot ego
            # RR = [-0.93, 0.93, -1.45, 2.62]
            # l = RR[X_MAX] - RR[X_MIN]
            # w = RR[Y_MAX] - RR[Y_MIN]
            # center = np.array([(RR[X_MAX]+RR[X_MIN])/2, (RR[Y_MAX]+RR[Y_MIN])/2])
            # center = R @ center + seed_p
            # rectangle = [center[0], center[1], seed_yaw, l, w]
            # self.debug(obstacles, rectangle)
            
            RR = sorted_RR[0]
        
        end_time = time.time()
        duration = end_time - start_time
        # print(f"MER Problem took {duration:.6f} seconds with {pts.shape[0]:d} points")


        # decode into [x, y, yaw, l, w]
        l = RR[X_MAX] - RR[X_MIN]
        w = RR[Y_MAX] - RR[Y_MIN]
        center = np.array([(RR[X_MAX]+RR[X_MIN])/2, (RR[Y_MAX]+RR[Y_MIN])/2])
        center = R @ center + seed_p
        rectangle = [center[0], center[1], seed_yaw, l, w]

        # self.debug(obstacles, rectangle)
        if debug:
            self.debug(obstacles, obs_geo, rectangle, ego_p=seed_p)
            self.debug1(obstacles, rectangle, seed_p, seed_yaw, sorted_RR)
        return rectangle
    
    def inflateRectangleBound(self, obstacles: np.array, seed_p: np.array, seed_yaw: float, debug: bool =False,
                        x_max: float=10, x_min: float=-10, y_max: float=10, y_min: float=-10) -> np.array:
        # obstacles: n * 2
        R = np.array([[math.cos(seed_yaw), -math.sin(seed_yaw)],
                    [math.sin(seed_yaw), math.cos(seed_yaw)]])
        projected_points = R.T @ (obstacles.T - seed_p[:,np.newaxis])
        projected_points = projected_points.T
        pts = np.empty((0, 2))
        for point in projected_points:
            if x_min < point[0] < x_max and y_min < point[1] < y_max:
                pts = np.vstack([pts, point])
                
        '''
            find the largest empty rectangle containing zero point
            Based on 'On the maximum empty rectangle problem, A.Naamad'
        '''
        # RR: np.array([x_min, x_max, y_min, y_max, S])
        X_MIN, X_MAX, Y_MIN, Y_MAX, S = 0, 1, 2, 3, 4
        start_time = time.time()
        RR = []
        RR = self._findType1RR(pts, x_max, x_min, y_max, y_min)
        RR += self._findType2RR(pts, x_max, x_min, y_max, y_min)
        RR += self._findType3RR(pts, x_max, x_min, y_max, y_min)
        RR += self._findType4RR(pts, x_max, x_min, y_max, y_min)

        sorted_RR = sorted(RR, key=lambda rr: rr[S])
        sorted_RR.reverse()

        try:
            RR_max = None
            for RR in sorted_RR:
                if self.coverEgoRectangle(RR):
                    RR_max = RR
                    break
            assert RR_max is not None
            RR = RR_max
        except:
            # return box 0 
            
            # debug: plot ego
            # RR = [-0.93, 0.93, -1.45, 2.62]
            # l = RR[X_MAX] - RR[X_MIN]
            # w = RR[Y_MAX] - RR[Y_MIN]
            # center = np.array([(RR[X_MAX]+RR[X_MIN])/2, (RR[Y_MAX]+RR[Y_MIN])/2])
            # center = R @ center + seed_p
            # rectangle = [center[0], center[1], seed_yaw, l, w]
            # self.debug(obstacles, rectangle)
            
            RR = sorted_RR[0]
        
        end_time = time.time()
        duration = end_time - start_time
        # print(f"MER Problem took {duration:.6f} seconds with {pts.shape[0]:d} points")


        # decode into [x, y, yaw, l, w]
        l = RR[X_MAX] - RR[X_MIN]
        w = RR[Y_MAX] - RR[Y_MIN]
        center = np.array([(RR[X_MAX]+RR[X_MIN])/2, (RR[Y_MAX]+RR[Y_MIN])/2])
        center = R @ center + seed_p
        rectangle = [center[0], center[1], seed_yaw, l, w]

        # self.debug(obstacles, rectangle)
        if debug:
            self.debug(obstacles, rectangle)
        return rectangle
    
    @staticmethod
    @jit(nopython=True)
    def _findType1RR(pts, x_max, x_min, y_max, y_min):
        '''
            type 1, two opposite edges of the region
            only consider [min x+, max x-] & [min y+, max y-]
            2 in total 
        '''
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        # # min x+
        positive_x_indices = np.where(x_coords > 0)[0]
        if len(positive_x_indices) > 0:
            min_x_index = positive_x_indices[np.argmin(x_coords[positive_x_indices])]
            min_x_plus = pts[min_x_index]
        else:
            min_x_plus = None
        # max x-
        negative_x_indices = np.where(x_coords < 0)[0]
        if len(negative_x_indices) > 0:
            max_x_index = negative_x_indices[np.argmax(x_coords[negative_x_indices])]
            max_x_minus = pts[max_x_index]
        else:
            max_x_minus = None
        # min y+
        positive_y_indices = np.where(y_coords > 0)[0]
        if len(positive_y_indices) > 0:
            min_y_index = positive_y_indices[np.argmin(y_coords[positive_y_indices])]
            min_y_plus = pts[min_y_index]
        else:
            min_y_plus = None 
        # max y-
        negative_y_indices = np.where(y_coords < 0)[0]
        if len(negative_y_indices) > 0:
            max_y_index = negative_y_indices[np.argmax(y_coords[negative_y_indices])]
            max_y_minus = pts[max_y_index]
        else:
            max_y_minus = None
        
        xm = max_x_minus[0] if max_x_minus is not None else x_min
        xp = min_x_plus[0] if min_x_plus is not None else x_max
        ym = y_min
        yp = y_max
        S = (yp - ym) * (xp - xm)
        RR1_1 = np.array([xm, xp, ym, yp, S])
        xm = x_min
        xp = x_max
        ym = max_y_minus[1] if max_y_minus is not None else y_min
        yp = min_y_plus[1] if min_y_plus is not None else y_max
        S = (yp - ym) * (xp - xm)
        RR1_2 = np.array([xm, xp, ym, yp, S])

        return [RR1_1, RR1_2]

    @staticmethod
    @jit(nopython=True)
    def _findType2RR(pts, x_max, x_min, y_max, y_min):
        '''
            type 2, two adjacent edges of the region
            1. deal with quadrants
            for each quadrant:
                2. sort by x
                3. in the sorted queue, pop the i-th point into RR list, and remove points k whose y_k > y_i in the queue
                4. move to i+1 point until queue is empty
        '''
        RR2 = []

        quadrants = [
            pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)],  # Q1
            pts[(pts[:, 0] < 0) & (pts[:, 1] > 0)],  # Q2
            pts[(pts[:, 0] < 0) & (pts[:, 1] < 0)],  # Q3
            pts[(pts[:, 0] > 0) & (pts[:, 1] < 0)]   # Q4
        ]
        
        
        for i, q_pts in enumerate(quadrants):
            # for point in q_pts:
            for j in range(q_pts.shape[0]):
                point = q_pts[j]
                px, py = point
                interiors = [
                    pts[(pts[:,0] < px) & (pts[:, 1] < py)], # Q1
                    pts[(pts[:,0] > px) & (pts[:, 1] < py)], # Q2
                    pts[(pts[:,0] > px) & (pts[:, 1] > py)], # Q3
                    pts[(pts[:,0] < px) & (pts[:, 1] > py)], # Q4
                ]
                interior = interiors[i]
                if interior.shape[0] == 0:
                    RRs = np.zeros((4,5))
                    RRs[0] = [x_min, px, y_min, py, (px-x_min)*(py-y_min)] # Q1
                    RRs[1] = [px, x_max, y_min, py, (x_max-px)*(py-y_min)] # Q2
                    RRs[2] = [px, x_max, py, y_max, (x_max-px)*(y_max-py)] # Q3
                    RRs[3] = [x_min, px, py, y_max, (px-x_min)*(y_max-py)] # Q4
                    RR = RRs[i]
                    RR2.append(RR)
                else:
                    continue
                
        return RR2

    @staticmethod
    @jit(nopython=True)
    def _findType3RR(pts, x_max, x_min, y_max, y_min):
        '''
            type 3, one edge of the region
            traverse 4 edges, each for N points
            4N RRs at most
        '''
        RR3 = []
        # bottom: y_min
        for i in range(pts.shape[0]):
            x_i, y_i = pts[i]
            if y_i < 0: 
                continue
            # Find closest point on the left with y < y_i
            left_pts = pts[(pts[:, 0] < x_i) & (pts[:, 1] < y_i)]
            if left_pts.size > 0:
                left_x = left_pts[:, 0].max()
            else:
                left_x = x_min 
            # Find closest point on the right with y < y_i
            right_pts = pts[(pts[:, 0] > x_i) & (pts[:, 1] < y_i)]
            if right_pts.size > 0:
                right_x = right_pts[:, 0].min()
            else:
                right_x = x_max
            
            if left_x > 0 or right_x < 0:
                continue
            
            RR = np.array([left_x, right_x, y_min, y_i, (right_x-left_x)*(y_i-y_min)])
            RR3.append(RR)
        # top: y_max
        for i in range(pts.shape[0]):
            x_i, y_i = pts[i]
            if y_i > 0: 
                continue
            # Find closest point on the left with y < y_i
            left_pts = pts[(pts[:, 0] < x_i) & (pts[:, 1] > y_i)]
            if left_pts.size > 0:
                left_x = left_pts[:, 0].max()
            else:
                left_x = x_min 
            # Find closest point on the right with y < y_i
            right_pts = pts[(pts[:, 0] > x_i) & (pts[:, 1] > y_i)]
            if right_pts.size > 0:
                right_x = right_pts[:, 0].min()
            else:
                right_x = x_max
            
            if left_x > 0 or right_x < 0:
                continue
            
            RR = np.array([left_x, right_x, y_i, y_max, (right_x-left_x)*(y_max-y_i)])
            RR3.append(RR)
        # left: x_min
        for i in range(pts.shape[0]):
            x_i, y_i = pts[i]
            if x_i < 0: 
                continue
            # Find closest point on the upper with x < x_i
            upper_pts = pts[(pts[:, 0] < x_i) & (pts[:, 1] > y_i)]
            if upper_pts.size > 0:
                upper_y = upper_pts[:, 1].min()
            else:
                upper_y = y_max 
            # Find closest point on the lower with x < x_i
            lower_pts = pts[(pts[:, 0] < x_i) & (pts[:, 1] < y_i)]
            if lower_pts.size > 0:
                lower_y = lower_pts[:, 1].max()
            else:
                lower_y = y_min
            
            if upper_y < 0 or lower_y > 0:
                continue
            
            RR = np.array([x_min, x_i, lower_y, upper_y, (x_i-x_min)*(upper_y-lower_y)])
            RR3.append(RR)
        # right: x_max
        for i in range(pts.shape[0]):
            x_i, y_i = pts[i]
            if x_i > 0: 
                continue
            # Find closest point on the upper with x > x_i
            upper_pts = pts[(pts[:, 0] > x_i) & (pts[:, 1] > y_i)]
            if upper_pts.size > 0:
                upper_y = upper_pts[:, 1].min()
            else:
                upper_y = y_max 
            # Find closest point on the lower with x < x_i
            lower_pts = pts[(pts[:, 0] > x_i) & (pts[:, 1] < y_i)]
            if lower_pts.size > 0:
                lower_y = lower_pts[:, 1].max()
            else:
                lower_y = y_min
            
            if upper_y < 0 or lower_y > 0:
                continue
            
            RR = np.array([x_i, x_max, lower_y, upper_y, (x_max-x_i)*(upper_y-lower_y)])
            RR3.append(RR)
        return RR3
    
    @staticmethod
    @jit(nopython=True)
    def _findType4RR(pts, x_max, x_min, y_max, y_min):
        '''
            type 4 
            1. pick any 2 points as lower and upper
            2. find left and right
            O(n^2) complexity
        '''
        RR4 = []
        negative_p = pts[pts[:, 1]<0]
        positive_p = pts[pts[:,1]>=0]
        for c in range(negative_p.shape[0]):
            x_c, y_c = negative_p[c]
            for a in range(positive_p.shape[0]):
                x_a, y_a = positive_p[a]
                left = min(x_a, x_c)
                right = max(x_a, x_c)
                available_b = pts[(y_c < pts[:,1]) & (pts[:,1] < y_a) & (pts[:,0] < left)]
                if available_b.shape[0] > 0:
                    x_b = np.max(available_b[:,0])
                    if x_b > 0:
                        continue
                else:
                    continue
                available_d = pts[(y_c < pts[:,1]) & (pts[:,1] < y_a) & (pts[:,0] > right)] 
                if available_d.shape[0] > 0:
                    x_d = np.min(available_d[:,0])
                    if x_d < 0:
                        continue
                else:
                    continue
                if ((y_c < pts[:,1]) & (pts[:,1] < y_a) & (x_b < pts[:, 0]) & (pts[:, 0] < x_d)).any():   # no interior
                    continue
                RR = np.array([x_b, x_d, y_c, y_a, (x_d-x_b)*(y_a-y_c)])
                RR4.append(RR)
        return RR4

    # '''
    def debug(self, obstacles, obs_geo, rectangle, ego_p):
        # Plot the original obstacles
        plt.figure(figsize=(5, 5))
        plt.xlim(xmin=-15, xmax=15)
        plt.ylim(ymin=-20, ymax=20)

        for obs in obs_geo:  
            if isinstance(obs, Polygon):
                points = np.array(obs.exterior.coords)
                plt.plot(points[:, 0], points[:, 1], color='#FFA400', linewidth=2, alpha=0.8)  # yellow
            elif isinstance(obs, LineString):
                points = np.array(obs.coords)
                plt.plot(points[:, 0], points[:, 1], color='#177CB0', linewidth=2, alpha=0.3)  # blue
            else:
                raise TypeError("wrong obs type")
            
        plt.scatter(obstacles[:, 0], obstacles[:, 1], s=1, color='#3D3B4F', edgecolors='none', alpha=1.0, zorder=3)  # dar grey

        # Plot the final rectangle
        seed_p = rectangle[:2]
        yaw = rectangle[2]
        s = rectangle[3]
        l = rectangle[4]

        # Rectangle vertices in local coordinates
        rect_local = np.array([
            [s/2, l/2],
            [s/2, -l/2],
            [-s/2, -l/2],
            [-s/2, l/2],
            [s/2, l/2]  # Closing the rectangle
        ])

        # Rotation matrix
        R = np.array([[math.cos(yaw), -math.sin(yaw)],
                      [math.sin(yaw), math.cos(yaw)]])

        # # Transform rectangle to global coordinates
        # rect_global = (R @ rect_local.T).T + seed_p

    

        # plt.plot(rect_global[:, 0], rect_global[:, 1], color='red', linewidth=1, label='Inflated Rectangle')
        plt.scatter(ego_p[0], ego_p[1], color='#F00056', marker='x', alpha=0.8, label='Seed Point', s=10)

        plt.axis('equal')
        plt.axis('off')
        plt.grid(False)  # Disable grid lines
        # plt.legend()
        plt.savefig('vis/corridor_debug.png', bbox_inches='tight', dpi=1000)
        plt.close()
    '''

    '''
    def debug(self, obstacles, obs_geo, rectangle, ego_p):

        # Total number of frames
        num_frames = 50
        from matplotlib.figure import Figure

        for frame_idx in range(num_frames):
            # Calculate dynamic alpha values for fading effect
            scatter_alpha = frame_idx / (num_frames - 1)  # Gradually increase to 1
            obs_alpha = 1.0 - scatter_alpha  # Gradually decrease to 0

            # Create a new figure for each frame
            fig = Figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.set_xlim(-20, 20)
            ax.set_ylim(-30, 30)

            # Draw obs_geo with decreasing alpha
            for obs in obs_geo:
                if isinstance(obs, Polygon):
                    points = np.array(obs.exterior.coords)
                    ax.plot(points[:, 0], points[:, 1], color='#FFA400', linewidth=2, alpha=0.8*obs_alpha)  # yellow
                elif isinstance(obs, LineString):
                    points = np.array(obs.coords)
                    ax.plot(points[:, 0], points[:, 1], color='#177CB0', linewidth=2, alpha=0.3*obs_alpha)  # blue
                else:
                    raise TypeError("wrong obs type")

            # Draw scatter points with increasing alpha
            ax.scatter(obstacles[:, 0], obstacles[:, 1], s=5, color='#3D3B4F', edgecolors='none', alpha=scatter_alpha, zorder=2)  # dark grey
            ax.scatter(ego_p[0], ego_p[1], color='#F00056', marker='x', alpha=0.8, label='Seed Point', s=6)

            ax.set_aspect('equal')
            ax.axis('off')

            # Save frame
            fig.savefig('vis/video/BEV_obs'+str(frame_idx)+'.png', bbox_inches='tight', dpi=500)
            plt.close(fig)
    '''

    # '''
    def debug1(self, obstacles, rectangle, seed_p, seed_yaw, sorted_RR):
        # Create a 3D figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-20, 20)
        ax.set_zlim(-10, 0)
        z_max = 2

        # Plot rectangles in sorted_RR uniformly along z
        X_MIN, X_MAX, Y_MIN, Y_MAX, S = 0, 1, 2, 3, 4
        n_RR = len(sorted_RR)
        for i, RR in enumerate(reversed(sorted_RR)):
            z = -(1 - i / (n_RR - 1)) * z_max  # Uniformly distribute heights from 0 to 10
            l = RR[X_MAX] - RR[X_MIN]
            w = RR[Y_MAX] - RR[Y_MIN]
            center = np.array([(RR[X_MAX] + RR[X_MIN]) / 2, (RR[Y_MAX] + RR[Y_MIN]) / 2])
            R = np.array([[math.cos(seed_yaw), -math.sin(seed_yaw)],
                    [math.sin(seed_yaw), math.cos(seed_yaw)]])
            center = R @ center + seed_p
            rect = [center[0], center[1], seed_yaw, l, w]

            rect_p = rect[:2]
            yaw = rect[2]
            s = rect[3]
            l = rect[4]
            rect_local = np.array([
                [s/2, l/2],
                [s/2, -l/2],
                [-s/2, -l/2],
                [-s/2, l/2],
                [s/2, l/2]  # Closing the rectangle
            ])
            R = np.array([[math.cos(yaw), -math.sin(yaw)],
                        [math.sin(yaw), math.cos(yaw)]])
            rect_global = (R @ rect_local.T).T + rect_p
            cmap = 'Reds'
            c =  -1 + i / n_RR * 2
            color = np.array(plt.cm.get_cmap(cmap)(c))[:3]
            ax.plot(rect_global[:, 0], rect_global[:, 1], zs=z, zdir='z', color=color, alpha=0.3, linewidth=0.3)

        # Plot the original obstacles at z=0
        ax.scatter(obstacles[:, 0], obstacles[:, 1], zs=0, zdir='z', s=1, color='#3D3B4F', edgecolors='none', alpha=0.8)

        # Plot the max region at z=0
        rect_p = seed_p
        yaw = seed_yaw
        s = self.x_max - self.x_min
        l = self.y_max - self.y_min
        rect_local = np.array([
            [s/2, l/2],
            [s/2, -l/2],
            [-s/2, -l/2],
            [-s/2, l/2],
            [s/2, l/2]  # Closing the rectangle
        ])
        R = np.array([[math.cos(yaw), -math.sin(yaw)],
                    [math.sin(yaw), math.cos(yaw)]])
        rect_global = (R @ rect_local.T).T + rect_p
        ax.plot(rect_global[:, 0], rect_global[:, 1], zs=0, zdir='z', color='#003371', linewidth=1, linestyle='--')

        # Plot the final rectangle at z=10
        rect_p = rectangle[:2]
        yaw = rectangle[2]
        s = rectangle[3]
        l = rectangle[4]
        rect_local = np.array([
            [s/2, l/2],
            [s/2, -l/2],
            [-s/2, -l/2],
            [-s/2, l/2],
            [s/2, l/2]  # Closing the rectangle
        ])
        R = np.array([[math.cos(yaw), -math.sin(yaw)],
                    [math.sin(yaw), math.cos(yaw)]])
        rect_global = (R @ rect_local.T).T + rect_p
        cmap_corridor = 'plasma'
        c = 0.5
        color_corridor = np.array(plt.cm.get_cmap(cmap_corridor)(c))[:3]
        ax.plot(rect_global[:, 0], rect_global[:, 1], zs=0.0, zdir='z', color=color_corridor, alpha=1.0, linewidth=2, zorder=3)

        

        # Plot the seed point
        ax.scatter(seed_p[0], seed_p[1], zs=0, zdir='z', color='#F00056', marker='x', alpha=0.8, s=6)

        # # Define the plane z=0
        # x = np.linspace(-15, 15, 10)
        # y = np.linspace(-30, 30, 10)
        # x, y = np.meshgrid(x, y)
        # z = np.zeros_like(x)  # Plane at z=0

        # # Plot the plane
        # ax.plot_surface(x, y, z, color='#A1AFC9', alpha=0.1, label='Plane z=0')

        ax.set_box_aspect([1, 1, 0.8])  # aspect ratio for 3D
        ax.grid(False)
        ax.set_axis_off()
        plt.savefig('vis/corridor_debug3d.png', bbox_inches='tight', dpi=500)
        plt.close()
    # '''

    '''
    def debug1(self, obstacles, rectangle, seed_p, seed_yaw, sorted_RR):
        import os
        import cv2
        import glob

        # Number of frames for the animation
        num_frames = len(sorted_RR) + 20  # Gradually transition to the final rectangle
        z_max = 2  # Maximum z-height
        cmap = 'Reds'  # Color map for rectangles
        cmap_corridor = 'plasma'  # Color map for the final rectangle

        # Generate frames
        # for frame_idx in range(num_frames):
        for frame_idx in range(10):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(-15, 15)
            ax.set_ylim(-20, 20)
            ax.set_zlim(-10, 0)
            ax.set_box_aspect([1, 1, 0.8])  # Aspect ratio for 3D
            ax.grid(False)
            ax.set_axis_off()

            # Plot sorted_RR rectangles up to the current frame
            n_RR = len(sorted_RR)
            for i, RR in enumerate(reversed(sorted_RR)):
                if frame_idx < i:
                    continue  # Only render up to the current frame
                z = -(1 - i / (n_RR - 1)) * z_max  # Uniformly distribute heights
                l = RR[1] - RR[0]
                w = RR[3] - RR[2]
                center = np.array([(RR[1] + RR[0]) / 2, (RR[3] + RR[2]) / 2])
                R = np.array([[math.cos(seed_yaw), -math.sin(seed_yaw)],
                            [math.sin(seed_yaw), math.cos(seed_yaw)]])
                center = R @ center + seed_p
                rect = [center[0], center[1], seed_yaw, l, w]

                rect_p = rect[:2]
                yaw = rect[2]
                s = rect[3]
                l = rect[4]
                rect_local = np.array([
                    [s / 2, l / 2],
                    [s / 2, -l / 2],
                    [-s / 2, -l / 2],
                    [-s / 2, l / 2],
                    [s / 2, l / 2]  # Closing the rectangle
                ])
                R = np.array([[math.cos(yaw), -math.sin(yaw)],
                            [math.sin(yaw), math.cos(yaw)]])
                rect_global = (R @ rect_local.T).T + rect_p
                c = i / n_RR 
                color = np.array(plt.cm.get_cmap(cmap)(c))[:3]
                ax.plot(rect_global[:, 0], rect_global[:, 1], zs=z, zdir='z', color=color, alpha=0.3, linewidth=0.3)
            
            # Plot the final rectangle
            if frame_idx >= n_RR:
                rect_p = rectangle[:2]
                yaw = rectangle[2]
                s = rectangle[3]
                l = rectangle[4]
                rect_local = np.array([
                    [s / 2, l / 2],
                    [s / 2, -l / 2],
                    [-s / 2, -l / 2],
                    [-s / 2, l / 2],
                    [s / 2, l / 2]  # Closing the rectangle
                ])
                R = np.array([[math.cos(yaw), -math.sin(yaw)],
                            [math.sin(yaw), math.cos(yaw)]])
                rect_global = (R @ rect_local.T).T + rect_p
                c = 0.5
                color_corridor = np.array(plt.cm.get_cmap(cmap_corridor)(c))[:3]
                ax.plot(rect_global[:, 0], rect_global[:, 1], zs=0.0, zdir='z', color=color_corridor, alpha=1.0, linewidth=2, zorder=3)
            
            rect_p = seed_p
            yaw = seed_yaw
            s = self.x_max - self.x_min
            l = self.y_max - self.y_min
            rect_local = np.array([
                [s/2, l/2],
                [s/2, -l/2],
                [-s/2, -l/2],
                [-s/2, l/2],
                [s/2, l/2]  # Closing the rectangle
            ])
            R = np.array([[math.cos(yaw), -math.sin(yaw)],
                        [math.sin(yaw), math.cos(yaw)]])
            rect_global = (R @ rect_local.T).T + rect_p
            ax.plot(rect_global[:, 0], rect_global[:, 1], zs=0, zdir='z', color='#003371', linewidth=1, linestyle='--')

            # Plot obstacles at z=0
            ax.scatter(obstacles[:, 0], obstacles[:, 1], zs=0, zdir='z', s=1, color='#3D3B4F', edgecolors='none', alpha=0.8)

            # Plot the seed point
            ax.scatter(seed_p[0], seed_p[1], zs=0, zdir='z', color='#F00056', marker='x', alpha=0.8, s=10)

            # Save the frame
            frame_path = os.path.join('vis/video', f'frame_{frame_idx:03d}.png')
            plt.savefig(frame_path, bbox_inches='tight', dpi=500)
            plt.close()

        # Compose frames into a video
        video_path = 'corridor_debug3d_video.mp4'
        frame_files = sorted(glob.glob(os.path.join('vis/video', 'frame_*.png')))
        if not frame_files:
            print("No frames found.")
            return

        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        height, width, layers = first_frame.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

        # Add frames to video
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            video.write(frame)

        video.release()
        print(f"Video saved to {video_path}")
    '''

    def drawCorridor3D(self, vis_pack):
        # Create a 3D figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-20, 20)
        ax.set_zlim(0, 10)
        z_max = 3
        ax.view_init(elev=20, azim=70)

        n_corridor = len(vis_pack)
        # print('n_corridor', n_corridor)
        for i, pack in enumerate(vis_pack): # [seed_p, seed_yaw, obstacles, corridor]
            seed_p, seed_yaw, obstacles, corridor = pack
            
            if i == n_corridor -1:
                zi = 0.0
                x_min, x_max = -15, 15  # Define x-range
                y_min, y_max = -30, 30  # Define y-range
                for obs in obstacles:  
                    if isinstance(obs, Polygon):
                        points = np.array(obs.exterior.coords)
                        # ax.plot(points[:, 0], points[:, 1], zs=zi, zdir='z', color='#FFB61E', linewidth=2, alpha=0.5)  # yellow
                    elif isinstance(obs, LineString):
                        points = np.array(obs.coords)
                        # ax.plot(points[:, 0], points[:, 1], zs=zi, zdir='z', color='#177CB0', linewidth=2, alpha=0.3)  # blue
                    else:
                        raise TypeError("wrong obs type")
                    
                    if np.all((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                        (points[:, 1] >= y_min) & (points[:, 1] <= y_max)):
                        color = '#FFA400' if isinstance(obs, Polygon) else '#177CB0'
                        linewidth = 2
                        alpha = 0.8 if isinstance(obs, Polygon) else 0.3
                        # alpha = 0.8 if i==n_corridor-1 else alpha
                        ax.plot(points[:, 0], points[:, 1], zs=zi, zdir='z', color=color, linewidth=linewidth, alpha=alpha)
            
            '''
            # Create a grid of x and y values for the plane
            x = np.linspace(x_min, x_max, 100)  # Adjust range and resolution as needed
            y = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.full_like(X, zi)  # Plane at z = 0
            # Plot the plane
            ax.plot_surface(X, Y, Z, color='gray', alpha=0.3, zorder=0)
            '''

            # Plot the rectangle at t
            z_cor = z_max * (1 - (i+1) / n_corridor)
            rect_p = corridor[:2]
            yaw = corridor[2]
            s = corridor[3]
            l = corridor[4]
            rect_local = np.array([
                [s/2, l/2],
                [s/2, -l/2],
                [-s/2, -l/2],
                [-s/2, l/2],
                [s/2, l/2]  # Closing the rectangle
            ])
            R = np.array([[math.cos(yaw), -math.sin(yaw)],
                        [math.sin(yaw), math.cos(yaw)]])
            rect_global = (R @ rect_local.T).T + rect_p
            cmap_corridor = 'plasma'
            c = (i/n_corridor) * 0.5 + 0.5
            color_corridor = np.array(plt.cm.get_cmap(cmap_corridor)(c))[:3]
            ax.plot(rect_global[:, 0], rect_global[:, 1], zs=z_cor, zdir='z', color=color_corridor, alpha=0.8, linewidth=2, zorder=i+1)

        ax.set_box_aspect([1, 1, 0.8])  # aspect ratio for 3D
        ax.grid(False)
        ax.set_axis_off()
        plt.savefig('vis/corridor_final_3d.png', bbox_inches='tight', dpi=500)
        plt.close()

    def interpolate_points(self, points1, points2, n_frames):
        """Linearly interpolate between two sets of points."""
        t = np.linspace(0, 1, n_frames + 2)  # Include the start and end points
        return [(1 - ti) * points1 + ti * points2 for ti in t]

#     def drawCorridor3D(self, vis_pack, n_frames=50):
#         # Create a 3D figure
#         fig = plt.figure(figsize=(8, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.set_xlim(-15, 15)
#         ax.set_ylim(-20, 20)
#         ax.set_zlim(-5, 10)
#         z_max = 10

#         n_corridor = len(vis_pack)
#         dz = z_max / ((n_corridor - 1) * (n_frames + 1))  # Step in z for each frame

#         for i in range(len(vis_pack) - 1):
#             seed_p, seed_yaw, obstacles1, corridor1 = pack1
#             seed_p2, seed_yaw2, obstacles2, corridor2 = pack2

#             # Interpolate corridors
#             rect_p1, yaw1, s1, l1 = corridor1[:2], corridor1[2], corridor1[3], corridor1[4]
#             rect_p2, yaw2, s2, l2 = corridor2[:2], corridor2[2], corridor2[3], corridor2[4]

#             for j in range(n_frames + 1):
#                 # Linear interpolation
#                 t = j / (n_frames + 1)
#                 rect_p = (1 - t) * np.array(rect_p1) + t * np.array(rect_p2)
#                 yaw = (1 - t) * yaw1 + t * yaw2
#                 s = (1 - t) * s1 + t * s2
#                 l = (1 - t) * l1 + t * l2

#                 # Rectangle vertices in local coordinates
#                 rect_local = np.array([
#                     [s / 2, l / 2],
#                     [s / 2, -l / 2],
#                     [-s / 2, -l / 2],
#                     [-s / 2, l / 2],
#                     [s / 2, l / 2]  # Closing the rectangle
#                 ])
#                 R = np.array([[np.cos(yaw), -np.sin(yaw)],
#                             [np.sin(yaw), np.cos(yaw)]])
#                 rect_global = (R @ rect_local.T).T + rect_p

#                 zi = i * (n_frames + 1) * dz + j * dz
#                 cmap_corridor = 'plasma'
#                 c = (i + t) / n_corridor
#                 color_corridor = np.array(plt.cm.get_cmap(cmap_corridor)(c))[:3]
#                 ax.plot(rect_global[:, 0], rect_global[:, 1], zs=zi, zdir='z', color=color_corridor, alpha=1.0, linewidth=2)
        
#         for i,p in enumerate(vis_pack):


# # Create a 3D figure
#         fig = plt.figure(figsize=(8, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.set_xlim(-15, 15)
#         ax.set_ylim(-20, 20)
#         ax.set_zlim(-10, 0)
#         z_max = 3

#         # Plot rectangles in sorted_RR uniformly along z
#         X_MIN, X_MAX, Y_MIN, Y_MAX, S = 0, 1, 2, 3, 4
#         n_RR = len(sorted_RR)
#         for i, ppp in enumerate(vis_pack):
#             seed_p, seed_yaw, obstacles, corridor = ppp

#             # obstacles
#             for obs in obstacles:  
#                 if isinstance(obs, Polygon):
#                     points = np.array(obs.exterior.coords)
#                     # ax.plot(points[:, 0], points[:, 1], zs=zi, zdir='z', color='#FFB61E', linewidth=2, alpha=0.5)  # yellow
#                 elif isinstance(obs, LineString):
#                     points = np.array(obs.coords)
#                     # ax.plot(points[:, 0], points[:, 1], zs=zi, zdir='z', color='#177CB0', linewidth=2, alpha=0.3)  # blue
#                 else:
#                     raise TypeError("wrong obs type")
                
#                 if np.all((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
#                     (points[:, 1] >= y_min) & (points[:, 1] <= y_max)):
#                     color = '#FFB61E' if isinstance(obs, Polygon) else '#177CB0'
#                     linewidth = 2
#                     alpha = 0.5 if isinstance(obs, Polygon) else 0.3
#                     alpha = 0.8 if i==n_corridor-1 else alpha
#                     ax.plot(points[:, 0], points[:, 1], zs=zi, zdir='z', color=color, linewidth=linewidth, alpha=alpha)


#             z = (1 - i / (n_RR - 1)) * z_max  # Uniformly distribute heights from 0 to 10
#             l = RR[X_MAX] - RR[X_MIN]
#             w = RR[Y_MAX] - RR[Y_MIN]
#             center = np.array([(RR[X_MAX] + RR[X_MIN]) / 2, (RR[Y_MAX] + RR[Y_MIN]) / 2])
#             R = np.array([[math.cos(seed_yaw), -math.sin(seed_yaw)],
#                     [math.sin(seed_yaw), math.cos(seed_yaw)]])
#             center = R @ center + seed_p
#             rect = [center[0], center[1], seed_yaw, l, w]

#             rect_p = rect[:2]
#             yaw = rect[2]
#             s = rect[3]
#             l = rect[4]
#             rect_local = np.array([
#                 [s/2, l/2],
#                 [s/2, -l/2],
#                 [-s/2, -l/2],
#                 [-s/2, l/2],
#                 [s/2, l/2]  # Closing the rectangle
#             ])
#             R = np.array([[math.cos(yaw), -math.sin(yaw)],
#                         [math.sin(yaw), math.cos(yaw)]])
#             rect_global = (R @ rect_local.T).T + rect_p
#             cmap = 'Reds'
#             c =  -1 + i / n_RR * 2
#             color = np.array(plt.cm.get_cmap(cmap)(c))[:3]
#             ax.plot(rect_global[:, 0], rect_global[:, 1], zs=z, zdir='z', color=color, alpha=0.1, linewidth=0.3)

#         # Plot the original obstacles at z=0
#         ax.scatter(obstacles[:, 0], obstacles[:, 1], zs=0, zdir='z', s=1, color='#3D3B4F', edgecolors='none', alpha=0.8)

#         # Plot the max region at z=0
#         rect_p = seed_p
#         yaw = seed_yaw
#         s = self.x_max - self.x_min
#         l = self.y_max - self.y_min
#         rect_local = np.array([
#             [s/2, l/2],
#             [s/2, -l/2],
#             [-s/2, -l/2],
#             [-s/2, l/2],
#             [s/2, l/2]  # Closing the rectangle
#         ])
#         R = np.array([[math.cos(yaw), -math.sin(yaw)],
#                     [math.sin(yaw), math.cos(yaw)]])
#         rect_global = (R @ rect_local.T).T + rect_p
#         ax.plot(rect_global[:, 0], rect_global[:, 1], zs=0, zdir='z', color='#003371', linewidth=1, linestyle='--')

#         # Plot the final rectangle at z=10
#         rect_p = rectangle[:2]
#         yaw = rectangle[2]
#         s = rectangle[3]
#         l = rectangle[4]
#         rect_local = np.array([
#             [s/2, l/2],
#             [s/2, -l/2],
#             [-s/2, -l/2],
#             [-s/2, l/2],
#             [s/2, l/2]  # Closing the rectangle
#         ])
#         R = np.array([[math.cos(yaw), -math.sin(yaw)],
#                     [math.sin(yaw), math.cos(yaw)]])
#         rect_global = (R @ rect_local.T).T + rect_p
#         cmap_corridor = 'plasma'
#         c = 0.5
#         color_corridor = np.array(plt.cm.get_cmap(cmap_corridor)(c))[:3]
#         ax.plot(rect_global[:, 0], rect_global[:, 1], zs=0.0, zdir='z', color=color_corridor, alpha=1.0, linewidth=2, zorder=3)




#         ax.set_box_aspect([1, 1, 0.8])  # aspect ratio for 3D
#         ax.grid(False)
#         ax.set_axis_off()
#         plt.savefig('vis/corridor_final_3d_interpolated.png', bbox_inches='tight', dpi=500)
#         plt.close()