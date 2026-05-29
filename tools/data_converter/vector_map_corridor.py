import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
import shapely.geometry 
from shapely import affinity, ops
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box, MultiLineString
import networkx as nx
from math import cos, sin
from tools.data_converter.simple_corridor_generator_jit import RectangleInflator
from scipy.spatial.transform import Rotation as R

########################### from MapTRv2 #########################################
def to_patch_coord(new_polygon, patch_angle, patch_x, patch_y):
    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                  origin=(patch_x, patch_y), use_radians=False)
    new_polygon = affinity.affine_transform(new_polygon,
                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
    return new_polygon


class VectorizedLocalMap(object):
    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'ped_crossing': 1,
        'contours': 2,
        'others': -1
    }
    def __init__(self,
                 nusc_map,
                 map_explorer,
                 patch_size,
                 map_classes=['divider','ped_crossing','boundary'],
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 centerline_classes=['lane_connector','lane'],
                 use_simplify=True,
                 ):
        super().__init__()
        self.nusc_map = nusc_map
        self.map_explorer = map_explorer
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.centerline_classes = centerline_classes
        self.patch_size = patch_size


    def gen_vectorized_samples(self, lidar2global_translation, lidar2global_rotation):
        '''
        use lidar2global to get gt map layers
        '''
        
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        # import ipdb;ipdb.set_trace()
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        map_dict = {'divider':[],'ped_crossing':[],'boundary':[]}
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes)
                line_instances_dict = self.line_geoms_to_instances(line_geom)     
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        map_dict[vec_class].append(np.array(instance.coords))
                        # vectors.append((instance, self.CLASS2LABEL.get(line_type, -1)))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    # vectors.append((instance, self.CLASS2LABEL.get('ped_crossing', -1)))
                    map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for instance in poly_bound_list:
                    # import ipdb;ipdb.set_trace()
                    map_dict[vec_class].append(np.array(instance.coords))
                    # vectors.append((contour, self.CLASS2LABEL.get('contours', -1)))
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')
        # import ipdb;ipdb.set_trace()
        return map_dict

    def get_map_geom(self, patch_box, patch_angle, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(patch_box, patch_angle, layer_name)
                # map_geom.append((layer_name, geoms))
                map_geom[layer_name] = geoms
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(patch_box, patch_angle, layer_name)
                # map_geom.append((layer_name, geoms))
                map_geom[layer_name] = geoms
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, patch_angle)
                # map_geom.append((layer_name, geoms))
                map_geom[layer_name] = geoms
        return map_geom

    def get_divider_line(self,patch_box,patch_angle,layer_name):
        if layer_name not in self.map_explorer.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == 'traffic_light':
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        line_list = []
        records = getattr(self.map_explorer.map_api, layer_name)
        for record in records:
            line = self.map_explorer.map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)

        return line_list

    def get_contour_line(self,patch_box,patch_angle,layer_name):
        if layer_name not in self.map_explorer.map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_explorer.map_api, layer_name)

        polygon_list = []
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_explorer.map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.map_explorer.map_api.extract_polygon(record['polygon_token'])

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list


    def get_ped_crossing_line(self, patch_box, patch_angle):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)
        polygon_list = []
        records = getattr(self.map_explorer.map_api, 'ped_crossing')
        # records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer.map_api.extract_polygon(record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        return polygon_list

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = dict()
        for line_type, a_type_of_lines in line_geom.items():
            one_type_instances = self._one_type_line_geom_to_instances(a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []
        
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def ped_poly_geoms_to_instances(self, ped_geom):
        # ped = ped_geom[0][1]
        # import ipdb;ipdb.set_trace()
        ped = ped_geom['ped_crossing']
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)


    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom['road_segment']
        lanes = polygon_geom['lane']
        # import ipdb;ipdb.set_trace()
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    
    def get_divider_instances(self, lidar2global_translation, lidar2global_rotation):
        '''
            get shapely instances 'divider' 
        ''' 
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180 
        line_instances_list = []

        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)
        for layer_name in self.line_classes:
            records = getattr(self.map_explorer.map_api, layer_name)
            for record in records:
                line = self.map_explorer.map_api.extract_line(record['line_token'])
                if line.is_empty:  # Skip lines without nodes.
                    continue

                new_line = line.intersection(patch)
                if not new_line.is_empty:
                    new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                    new_line = affinity.affine_transform(new_line,
                                                        [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    line_instances_list.append(new_line)
        
        # line_geoms_to_instances, we dont care the key here
        instances_list = self._one_type_line_geom_to_instances(line_instances_list)
        return instances_list
    
    def get_boundary_instances(self, lidar2global_translation, lidar2global_rotation):
        '''
            get shapely instances 'boundary' 
        ''' 
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180 
        
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)
        geoms = {}
        for layer_name in self.polygon_classes:
            polygon_list = []
            records = getattr(self.map_explorer.map_api, layer_name)
            if layer_name == 'drivable_area':
                for record in records:
                    polygons = [self.map_explorer.map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]

                    for polygon in polygons:
                        new_polygon = polygon.intersection(patch)
                        if not new_polygon.is_empty:
                            new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                        origin=(patch_x, patch_y), use_radians=False)
                            new_polygon = affinity.affine_transform(new_polygon,
                                                                    [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                            if new_polygon.geom_type == 'Polygon':
                                new_polygon = MultiPolygon([new_polygon])
                            polygon_list.append(new_polygon)

            else:
                for record in records:
                    polygon = self.map_explorer.map_api.extract_polygon(record['polygon_token'])

                    if polygon.is_valid:
                        new_polygon = polygon.intersection(patch)
                        if not new_polygon.is_empty:
                            new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                        origin=(patch_x, patch_y), use_radians=False)
                            new_polygon = affinity.affine_transform(new_polygon,
                                                                    [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                            if new_polygon.geom_type == 'Polygon':
                                new_polygon = MultiPolygon([new_polygon])
                            polygon_list.append(new_polygon)
            geoms[layer_name] = polygon_list
        
        # poly_geoms_to_instances, copy
        roads = geoms['road_segment']
        lanes = geoms['lane']
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)
        
        instances_list = self._one_type_line_geom_to_instances(results)
        return instances_list


    def sample_pts_from_line(self, line, sample_dist = 0.1):
        distances = np.arange(0, line.length, sample_dist)
        sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

        return sampled_points

    def interpolate_polygon_by_distance(self, polygon, distance=0.1):
        # Ensure the input is a Polygon
        if not isinstance(polygon, Polygon):
            raise TypeError("Input must be a shapely.geometry.Polygon")

        # Extract the exterior of the polygon
        boundary = polygon.exterior

        # Total length of the boundary
        length = boundary.length

        # Create a list of interpolated points
        interpolated_points = []
        
        # Initialize the starting distance
        current_distance = 0.0

        # Interpolate points along the boundary at intervals of 'distance'
        while current_distance < length:
            point = boundary.interpolate(current_distance)
            interpolated_points.append((point.x, point.y))
            current_distance += distance

        # Ensure the polygon is closed by adding the first point at the end
        if interpolated_points[0] != interpolated_points[-1]:
            interpolated_points.append(interpolated_points[0])

        return np.array(interpolated_points)


#########################################################################
############################# end of class ##############################

class CorridorConstructor():
    def __init__(self, root_path, MAPS, corridor_range=[7.5, -7.5, 15, -15]):  # |x| < |y| for nuscenes in the lidar frame
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=root_path, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])
        
        self.inflator = RectangleInflator(corridor_range)
        
    # main map function
    def obtain_vectormap(self, info, point_cloud_range):
        # import ipdb;ipdb.set_trace()
        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = info['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = info['ego2global_translation']

        lidar2global = ego2global @ lidar2ego

        lidar2global_translation = list(lidar2global[:3,3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

        location = info['map_location']
        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']

        patch_h = point_cloud_range[4]-point_cloud_range[1]
        patch_w = point_cloud_range[3]-point_cloud_range[0]
        patch_size = (patch_h, patch_w)
        vector_map = VectorizedLocalMap(self.nusc_maps[location], self.map_explorer[location],patch_size)
        map_anns = vector_map.gen_vectorized_samples(lidar2global_translation, lidar2global_rotation)
        return map_anns, vector_map


    def lidar2Vertex(self, position, heading, 
                     lidar2ego_t=[0.943713, 0.0, 1.84023],
                     lidar2ego_r=[0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817]):
        '''
            Convert from lidar frame (widely used in nusc) to vertices according to the vehicle shape.
        '''
        # Vehicle dimensions (ego frame)
        W, H = 1.85, 4.084
        # Define the vehicle's corner points in the ego frame
        pts = np.array([
            [H / 2. + 0.5 + 0.985793, W / 2.],    # Front left
            [H / 2. + 0.5 + 0.985793, -W / 2.],   # Front right
            [-H / 2. + 0.5 + 0.985793, W / 2.],   # Front left
            [-H / 2. + 0.5 + 0.985793, -W / 2.],  # Rear left
        ])

        # Lidar to ego translation and rotation (from dataset)
        lidar2ego_t = np.array(lidar2ego_t)  # Translation vector
        lidar2ego_r = Quaternion(lidar2ego_r)  # Rotation quaternion

        # Apply the rotation to the ego frame points (rotate lidar to ego)
        pts_ego = np.hstack((pts, np.zeros((4, 1))))
        pts_ego = pts_ego.T - lidar2ego_t[:, np.newaxis]
        pts_lidar = lidar2ego_r.rotation_matrix.T @ pts_ego
        pts_lidar = pts_lidar[:2]

        # World frame transformation (based on position and heading)
        # Create rotation matrix for the heading (world orientation)
        theta = heading
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, -s], [s, c]])

        # Apply rotation and translation to convert ego_pts to the world frame
        vtx_pts = np.dot(rotation_matrix, pts_lidar).T + position[:2]

        return vtx_pts

        
        

    def obtain_corridor(self,
                        info, 
                        nusc,
                        vector_map, 
                        sample,
                        fut_ts=6, 
                        dt=0.5,
                        intent_his=2.0,
                        intent_fut=5.0):
        ######################################
        ######### CORRIDR CONSTRUCT ##########
        ######################################
        future_corridor = []
        future_sample = sample

        # 1. map & intent
        # 1.1 get ego trajectory [-intent_his seconds, intent_fut/dt]
        history_trajectory = []
        future_trajectory = []
        history_step = int(intent_his/dt)
        future_step = int(intent_fut/dt)
        ptr = future_sample
        for i in range(history_step): 
            if ptr['prev'] != '': 
                scene_token = ptr['scene_token']
                ptr = nusc.get('sample', ptr['prev'])
                if ptr['scene_token'] != scene_token:
                    break
                pose_mat = self.get_global_sensor_pose(ptr, nusc)
                history_trajectory.insert(0, pose_mat)
            else:
                break
        ptr = future_sample
        for i in range(future_step): # 5s
            if ptr['next'] != '':
                scene_token = ptr['scene_token']
                ptr = nusc.get('sample', ptr['next'])
                if ptr['scene_token'] != scene_token:
                    break
                pose_mat = self.get_global_sensor_pose(ptr, nusc)
                future_trajectory.append(pose_mat)
            else:
                break
        pose_mat = self.get_global_sensor_pose(future_sample, nusc)
        intent_trajectory = history_trajectory + [pose_mat] + future_trajectory

        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = info['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = info['ego2global_translation']
        lidar2global = ego2global @ lidar2ego
        lidar2global_translation = list(lidar2global[:3,3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
        ego_t = np.array(info['ego2global_translation'])
        ego_r = Quaternion(info['ego2global_rotation'])
        lidar2ego_t = info['lidar2ego_translation']
        lidar2ego_r = info['lidar2ego_rotation']
        

        # 1.2. remove interction with ego and lanes
        divider_line_instances = vector_map.get_divider_instances(lidar2global_translation, lidar2global_rotation)
        boundary_line_instances = vector_map.get_boundary_instances(lidar2global_translation, lidar2global_rotation)
        for pose_mat in intent_trajectory:
            intent_pose_t = pose_mat[:3, 3]
            intent_pose_r = pose_mat[:3, :3]
            global2ego_t = ego_t
            global2ego_r = ego_r.rotation_matrix
            intent_pose_t = global2ego_r.T @ (intent_pose_t - global2ego_t).T
            intent_pose_r = global2ego_r.T @ intent_pose_r

            # ego to lidar
            ego2lidar_t = lidar2ego_t
            ego2lidar_r = Quaternion(lidar2ego_r).rotation_matrix
            intent_pose_t = ego2lidar_r.T @ (intent_pose_t - ego2lidar_t).T
            intent_pose_r = ego2lidar_r.T @ intent_pose_r
            future_q = R.from_matrix(intent_pose_r)
            _, _, intent_yaw = future_q.as_euler('xyz', degrees=False)

            vtx = self.lidar2Vertex(intent_pose_t[:2], intent_yaw, lidar2ego_t, lidar2ego_r)

            ego_polygon = shapely.geometry.Polygon(vtx)
            for line in divider_line_instances:
                if ego_polygon.intersects(line):
                    divider_line_instances.remove(line)
        

        vis_pack = []  # used to draw 3D
        for i in range(fut_ts+1): # samples in the future, fut_ts+1 rectangles in case the future frames are not available
            # move to next sample
            if i != 0:
                if future_sample['next'] == '':
                    for j in range(i, fut_ts+1):
                        future_corridor.append(future_corridor[i-1]) 
                    break
                future_sample = nusc.get('sample', future_sample['next'])

            obstacles = []
            # 2.1 get bounding boxes
            box_instance_polygons = []
            
            pose_mat = self.get_global_sensor_pose(future_sample, nusc)
            future_t = pose_mat[:3, 3]
            future_r = pose_mat[:3, :3]

            lidar_token = future_sample['data']['LIDAR_TOP']
            _, boxes, _ = nusc.get_sample_data(lidar_token)
            for box in boxes:  
                # future lidar frame -> world frame
                box_t_global = future_t + future_r @ np.array(box.center)
                box_r_global = future_r @ box.orientation.rotation_matrix
                # world frame -> ego  t0
                global2ego_t = ego_t
                global2ego_r = ego_r.rotation_matrix
                ego_box_t = global2ego_r.T @ (box_t_global - global2ego_t).T
                ego_box_r = global2ego_r.T @ box_r_global
                # ego to lidar
                ego2lidar_t = lidar2ego_t
                ego2lidar_r = Quaternion(lidar2ego_r).rotation_matrix
                future_box_t = ego2lidar_r.T @ (ego_box_t - ego2lidar_t).T
                future_box_r = ego2lidar_r.T @ ego_box_r 
                
                x_rel, y_rel = future_box_t[0], future_box_t[1]
                yaw = R.from_matrix(future_box_r).as_euler('xyz', degrees = False)[2]
                w = box.wlh[0]
                h = box.wlh[1]
                # convert to shapely polygon
                dx1 = h/2*cos(yaw) + w/2*sin(yaw)
                dy1 = h/2*sin(yaw) - w/2*cos(yaw)
                dx2 = h/2*cos(yaw) - w/2*sin(yaw)
                dy2 = h/2*sin(yaw) + w/2*cos(yaw)
                p1 = [x_rel + dx1, y_rel + dy1]
                p2 = [x_rel + dx2, y_rel + dy2]
                p3 = [x_rel - dx1, y_rel - dy1]
                p4 = [x_rel - dx2, y_rel - dy2] 
                box_polygon = shapely.geometry.Polygon([p1, p2, p3, p4])
                box_instance_polygons.append(box_polygon)

            # 2.2 get obstacles points
            obstacles = divider_line_instances + box_instance_polygons + boundary_line_instances
            obs_points = np.empty((0,2))
            for obs in obstacles:  
                if isinstance(obs, Polygon):
                    points = vector_map.interpolate_polygon_by_distance(obs, distance = 0.5)
                elif isinstance(obs, LineString):
                    points = vector_map.sample_pts_from_line(obs, sample_dist = 0.5)
                else:
                    raise TypeError("wrong obs type")
                obs_points = np.append(obs_points, points, axis=0)

            ######### 2. corrdor inflation ###########            
            # global to ego t0
            future_pose_mat = self.get_global_sensor_pose(future_sample, nusc)
            future_pose_t = future_pose_mat[:3, 3]
            future_pose_r = future_pose_mat[:3, :3]
            global2ego_t = ego_t
            global2ego_r = ego_r.rotation_matrix
            future_pose_t = global2ego_r.T @ (future_pose_t - global2ego_t).T
            future_pose_r = global2ego_r.T @ future_pose_r

            # ego to lidar
            ego2lidar_t = lidar2ego_t
            ego2lidar_r = Quaternion(lidar2ego_r).rotation_matrix
            future_pose_t = ego2lidar_r.T @ (future_pose_t - ego2lidar_t).T
            future_pose_r = ego2lidar_r.T @ future_pose_r

            seed_p = future_pose_t[:2]
            future_q = R.from_matrix(future_pose_r)
            _, _, seed_yaw = future_q.as_euler('xyz', degrees=False) #rpy

            corridor = self.inflator.inflateRectangle(obs_points, obstacles, seed_p, seed_yaw, debug = (i==0))
            # corridor = self.inflator.inflateRectangle(obs_points, seed_p, seed_yaw, debug = False)
            future_corridor.append(corridor)
            vis_pack.append([seed_p, seed_yaw, obstacles, corridor])
        
        self.inflator.drawCorridor3D(vis_pack[1:])

        ############# FINISH! #############
        
        # end_time = time.time()
        # duration = end_time - start_time
        # print(f"Corridor Generation took {duration:.6f} seconds")

        # end loop for future frames
        future_corridor = future_corridor[1:]  # no time 0
        future_corridor = np.asarray(future_corridor)

        # end loop
        return future_corridor

    def get_global_sensor_pose(self, rec, nusc):
        lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])

        sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
        sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
        return pose
    ####################################################################################################3
