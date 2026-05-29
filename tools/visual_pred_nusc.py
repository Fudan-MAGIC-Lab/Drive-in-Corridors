import argparse
import sys
sys.path.append('')
import os
import os.path as osp
import pickle
import mmcv
import numpy as np
from typing import List, Dict
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from matplotlib.collections import LineCollection
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from projects.mmdet3d_plugin.core.bbox.structures.nuscenes_box import CustomNuscenesBox, CustomDetectionBox, color_map

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
import time
import math

from nuscenes.nuscenes import NuScenes
from projects.mmdet3d_plugin.VAD.utils.corridor_utils import normalize_2d_bbox_corridor, denormalize_2d_bbox_corridor, to_real_corridor


out_path = 'visuals'
image_size = (1600, 900) 

# nuplan vehicle params
width=1.1485 * 2.0
half_w = width/2.0
front_length=4.049
rear_length=1.127

pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]


def add_border(image, border_thickness, border_color=(0, 255, 0)):
    # Make a copy of the original image
    bordered_image = image.copy()

    # Draw the border directly on the image
    h, w = image.shape[:2]

    # Top border
    cv2.rectangle(bordered_image, (0, 0), (w, border_thickness), border_color, -1)
    # Bottom border
    cv2.rectangle(bordered_image, (0, h - border_thickness), (w, h), border_color, -1)
    # Left border
    cv2.rectangle(bordered_image, (0, 0), (border_thickness, h), border_color, -1)
    # Right border
    cv2.rectangle(bordered_image, (w - border_thickness, 0), (w, h), border_color, -1)

    return bordered_image


def draw_boxes(image, cam, result, data):
    boxes = result[0]['pts_bbox']['boxes_3d']
    scores = result[0]['pts_bbox']['scores_3d']
    labels = result[0]['pts_bbox']['labels_3d']

    # boxes = data['gt_bboxes_3d'][0].data[0][0]
    # labels = data['gt_labels_3d'][0].data[0][0]

    lidar2img = data['img_metas'][0].data[0][0]['lidar2img'][cam]  ## this is augmented
    compensate = np.eye(4)  # transform3d.py
    compensate[0, 0] = 1/0.8  # 0.4 for tiny ... 0.8 for base
    compensate[1, 1] = 1/0.8
    lidar2img = compensate @ np.array(lidar2img)

    for score, box in zip(scores, boxes):
        if score < 0.4:
            continue
        corners_3d = compute_box_3d(box)
        corners_2d = project_to_image(corners_3d, lidar2img)

        # Draw the box
        if corners_2d is not None:
            draw = ImageDraw.Draw(image)
            for i in range(4):
                fill = 'yellow' if i == 0 else 'red'
                draw.line([tuple(corners_2d[i]), tuple(corners_2d[(i+1) % 4])], fill=fill, width=3)
                draw.line([tuple(corners_2d[i+4]), tuple(corners_2d[(i+1) % 4 + 4])], fill=fill, width=3)
                if i == 1:
                    draw.line([tuple(corners_2d[i]), tuple(corners_2d[i+4])], fill='yellow', width=3)
                else:
                    draw.line([tuple(corners_2d[i]), tuple(corners_2d[i+4])], fill=fill, width=3)
    return image

def compute_box_3d(box):
    # Compute the 3D corners of the bounding box
    x, y, z, w, l, h, yaw, _, _ = box.numpy()   # z is bottom center
    yaw = -np.pi/2-yaw  # data_convertor line 851
    corners_3d = np.array([
        [l/2, w/2, 0.0],
        [l/2, -w/2, 0.0],
        [-l/2, -w/2, 0.0],
        [-l/2, w/2, 0.0],
        [l/2, w/2, h],
        [l/2, -w/2, h],
        [-l/2, -w/2, h],
        [-l/2, w/2, h]
    ])
    
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    corners_3d = np.dot(R, corners_3d.T).T 
    corners_3d += np.array([x, y, z])
    
    return corners_3d

def project_to_image(corners_3d, lidar2img):
    # Transform from lidar coordinates to camera coordinates
    # cam2lidar_r = np.array(cam2lidar_r)
    # cam2lidar_t = np.array(cam2lidar_t)
    # cam2lidar_r_inv = np.linalg.inv(cam2lidar_r)
    # corners_3d_cam = np.dot(cam2lidar_r_inv, (corners_3d - cam2lidar_t).T).T

    # # Filter out points with negative z values
    # if np.any(corners_3d_cam[:, 2] <= 0):
    #     return None

    # corners_2d = np.dot(camera_intrinsic, corners_3d_cam.T).T
    # corners_2d = corners_2d[:, :2] / corners_2d[:, 2, np.newaxis]
    one_vector = np.ones((1, corners_3d.shape[0]))
    corners_3d_homo = np.vstack((corners_3d.T, one_vector))

    corners_2d = np.dot(lidar2img, corners_3d_homo).T
    # Filter out points with negative z values
    if np.any(corners_2d[:, 2] <= 0):
        return None
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2, np.newaxis]
    
    return corners_2d

def get_render_box_gt(box, fut_traj , label):
    """
    Map box from global coordinates to the vehicle's sensor coordinate system.
    :param boxes: The boxes in global coordinates.
    :param fut_trajs: The fut_trajs in ? coordinates.
    :return: The transformed boxes.
    """
    # Create Box instance.
    translation = box.center.numpy().T
    translation = translation[:,0] 
    size = box.dims.numpy().T
    # rotation = Quaternion(axis=[0, 0, 1], angle=-np.pi/2-box.yaw)  # 
    rotation = Quaternion(axis=[0, 0, 1], angle=box.yaw)  # 
    # rotation = Quaternion(axis=[0, 0, 1], angle=-np.pi/2-box[6])  # data_convertor line 851
    traj = fut_traj.numpy()
    # detection_name = category_to_detection_name()  # str
    detection_name = label

    box_out = CustomNuscenesBox(
        translation, size, rotation, traj, name = detection_name
    )
    return box_out


def get_render_box(box, fut_traj , label):
    """
    Map box from global coordinates to the vehicle's sensor coordinate system.
    :param boxes: The boxes in global coordinates.
    :param fut_trajs: The fut_trajs in ? coordinates.
    :return: The transformed boxes.
    """
    # Create Box instance.
    translation = box.center.numpy().T
    translation = translation[:,0] 
    size = box.dims.numpy().T
    rotation = Quaternion(axis=[0, 0, 1], angle=-np.pi/2-box.yaw)  # 
    # rotation = Quaternion(axis=[0, 0, 1], angle=box.yaw)  # 
    # rotation = Quaternion(axis=[0, 0, 1], angle=-np.pi/2-box[6])  # data_convertor line 851
    traj = fut_traj.numpy()
    # detection_name = category_to_detection_name()  # str
    detection_name = label

    box_out = CustomNuscenesBox(
        translation, size, rotation, traj, name = detection_name
    )
    return box_out

def rect2points(rectangle, real=False):
    if not real:
        rectangle = to_real_corridor(rectangle, pc_range)
    rectangle = rectangle.numpy()
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

    # Transform rectangle to global coordinates
    points = (R @ rect_local.T).T + seed_p
    return points

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sensor2lidar_rotation = R.T  # points @ R.T + T
    sensor2lidar_translation = T

    return sensor2lidar_rotation, sensor2lidar_translation


def render_sample_data(data=None,
                    pred=None,
                    savepath: str = None, 
                    colors_plt = ['#96C37D', '#C497B2', '#14517C', 'royalblue'],
                    class2label={'divider':0, 'ped_crossing':1, 'boundary':2, 'centerline': 3}) -> None:
    """
    Render sample data onto axis.
    """
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    plt.xlim(xmin=-30, xmax=30)
    plt.ylim(ymin=-30, ymax=30)

    # directly use render from CustomBox and visualzation.py
    
    boxes_pred = pred[0]['pts_bbox']['boxes_3d']
    boxes_scores = pred[0]['pts_bbox']['scores_3d']
    boxes_labels = pred[0]['pts_bbox']['labels_3d']
    boxes_trajs = pred[0]['pts_bbox']['trajs_3d']
    ignore_list = ['barrier', 'bicycle', 'traffic_cone'] 

    for i, (score, label, traj) in enumerate(zip(boxes_scores, boxes_labels, boxes_trajs)):
        if score < 0.4:
            continue
        box = boxes_pred[i]
        box_render = get_render_box(box, traj, label)
        box_render.render(axes, view=np.eye(4), colors=('tomato', 'tomato', 'tomato'), linewidth=1, box_idx=None, alpha=score.item())
        mode_idx = [0, 1, 2, 3, 4, 5]
        box_render.render_fut_trajs_grad_color(axes, linewidth=1, mode_idx=mode_idx, fut_ts=6, cmap='autumn')

    # show maps
    map_preds = pred[0]['pts_bbox']['map_pts_3d']
    map_labels = pred[0]['pts_bbox']['map_labels_3d']
    map_scores = pred[0]['pts_bbox']['map_scores_3d']
    for map_points, label, score in zip(map_preds, map_labels, map_scores):
        if score < 0.6: 
            continue
        points = map_points.numpy()
        pts_x = np.array([pt[0] for pt in points])
        pts_y = np.array([pt[1] for pt in points]) 
        
        axes.plot(pts_x, pts_y, color=colors_plt[label],linewidth=1,alpha=score.item(),zorder=-1)
        axes.scatter(pts_x, pts_y, color=colors_plt[label],s=1,alpha=0.8,zorder=-1) 
        
    # '''
    # Corridors
    # gt_corridors = data['gt_corridor']
    cmd = np.argmax(pred[0]['pts_bbox']['ego_fut_cmd'])
    corridor_preds = pred[0]['pts_bbox']['ego_cor_preds'][-1][cmd]
    n_corridor = len(corridor_preds)
    for j, gt_corridor in enumerate(corridor_preds):
        # cmap_corridor = 'jet'
        cmap_corridor = 'Reds'
        # a = 1-j/n_corridor
        a = (1-j/n_corridor) *0.5 + 0.5
        color_corridor = np.array(plt.cm.get_cmap(cmap_corridor)(a))[:3]
        corridor_points = rect2points(gt_corridor)
        axes.plot(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor,linewidth=2,alpha=0.2,zorder=-1)
        axes.scatter(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor, s=1,alpha=0.1,zorder=-1)
    
    '''
    # Clipped Corridors
    corridor_preds = pred[0]['pts_bbox']["clipped_cor"][0] # pred
    n_corridor = len(corridor_preds)
    for j, gt_corridor in enumerate(corridor_preds):
        cmap_corridor = 'Blues'
        a = (1-j/n_corridor) * 0.5 + 0.5
        color_corridor = np.array(plt.cm.get_cmap(cmap_corridor)(a))[:3]
        corridor_points = rect2points(gt_corridor, real=True)
        axes.plot(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor,linewidth=1,alpha=0.5,zorder=-1)
        axes.scatter(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor, s=1,alpha=0.1,zorder=-1)
    '''
        
    # opt traj
    cmd = np.argmax(pred[0]['pts_bbox']['ego_fut_cmd'])
    traj_opt = pred[0]['pts_bbox']['ego_fut_mpc'][cmd]
    traj_opt_plot = np.array(traj_opt).cumsum(axis=0)
    xy = np.stack((traj_opt_plot[:,0], traj_opt_plot[:,1]), axis=1)
    xy = np.stack((xy[:-1], xy[1:]), axis=1)
    y = np.sin(np.linspace(3/2*np.pi, 5/2*np.pi, traj_opt_plot.shape[0]))
    colors = color_map(y[:-1], 'spring')
    line_segments = LineCollection(xy, colors=colors, linewidths=2, cmap='spring')
    axes.add_collection(line_segments)


    # Show Planning.
    axes.plot([-0.9, -0.9], [-2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([-0.9, 0.9], [2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.9, 0.9], [2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.9, -0.9], [-2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.0, 0.0], [0.0, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    
    cmd = np.argmax(pred[0]['pts_bbox']['ego_fut_cmd'])
    plan_traj = pred[0]['pts_bbox']['ego_fut_preds'][cmd, :, :]
    plan_traj = plan_traj.cumsum(axis=0)
    plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
    plan_traj = np.stack((plan_traj[:-1], plan_traj[1:]), axis=1)

    plan_vecs = None
    for i in range(plan_traj.shape[0]):
        plan_vec_i = plan_traj[i]
        x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
        y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
        xy = np.stack((x_linspace, y_linspace), axis=1)
        xy = np.stack((xy[:-1], xy[1:]), axis=1)
        if plan_vecs is None:
            plan_vecs = xy
        else:
            plan_vecs = np.concatenate((plan_vecs, xy), axis=0)

    cmap = 'winter'
    y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 301))
    colors = color_map(y[:-1], cmap)
    line_segments = LineCollection(plan_vecs, colors=colors, linewidths=0.3, alpha=0.8, linestyles='solid', cmap=cmap)
    axes.add_collection(line_segments)

    axes.axes.xaxis.set_ticks([])
    axes.axes.yaxis.set_ticks([])
    axes.axis('off')
    fig.set_tight_layout(True)
    fig.canvas.draw()
    plt.savefig(savepath+'/bev_pred.png', bbox_inches='tight', dpi=200)
    plt.close()
    return

def render_sample_data_analysis(data=None,
                    pred=None,
                    savepath: str = None, 
                    index=0,
                    colors_plt = ['#96C37D', '#C497B2', '#14517C', 'royalblue'],
                    class2label={'divider':0, 'ped_crossing':1, 'boundary':2, 'centerline': 3}) -> None:
    """
    Render sample data onto axis.
    Use gt detection, prediction and map
    """
    fig, axes = plt.subplots(1, 1, figsize=(2, 3))
    plt.xlim(xmin=-15, xmax=15)
    plt.ylim(ymin=-15, ymax=30)

    '''
    # show boxes
    boxes_pred = data['gt_bboxes_3d'][0].data[0][0]
    boxes_labels = data['gt_labels_3d'][0].data[0][0]
    boxes_trajs = data['gt_attr_labels'][0].data[0][0][:,:12]
    boxes_trajs = boxes_trajs.unsqueeze(1)
    ignore_list = ['barrier', 'bicycle', 'traffic_cone'] 

    for i, (label, traj) in enumerate(zip(boxes_labels, boxes_trajs)):
        box = boxes_pred[i]
        box_render = get_render_box_gt(box, traj, label)
        box_render.render(axes, view=np.eye(4), colors=('tomato', 'tomato', 'tomato'), linewidth=1, box_idx=None)
        mode_idx = [0]
        box_render.render_fut_trajs_grad_color(axes, linewidth=1, alpha=0.2, mode_idx=mode_idx, fut_ts=6, cmap='autumn')

    # show maps
    maps = data['map_gt_bboxes_3d'].data[0][0].fixed_num_sampled_points
    maps_label = data['map_gt_labels_3d'].data[0][0]
    for map, map_label in zip(maps, maps_label):
        points = map.numpy()
        pts_x = np.array([pt[0] for pt in points])
        pts_y = np.array([pt[1] for pt in points]) 
        
        axes.plot(pts_x, pts_y, color=colors_plt[map_label],linewidth=1,alpha=0.8,zorder=-1)
        axes.scatter(pts_x, pts_y, color=colors_plt[map_label],s=1,alpha=0.8,zorder=-1) 
    
    # Corridors
    cmd = np.argmax(pred[0]['pts_bbox']['ego_fut_cmd'])
    corridor_preds = pred[0]['pts_bbox']['ego_cor_preds'][0, cmd] # pred
    # corridor_preds = data['gt_corridor'][-1][0]  # gt
    n_corridor = len(corridor_preds)
    for j, gt_corridor in enumerate(corridor_preds):
        # cmap_corridor = 'jet'
        # a = 1-j/n_corridor
        cmap_corridor = 'Reds'
        a = (1-j/n_corridor) * 0.5 + 0.5
        color_corridor = np.array(plt.cm.get_cmap(cmap_corridor)(a))[:3]
        corridor_points = rect2points(gt_corridor)
        # axes.plot(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor,linewidth=2,alpha=0.1,zorder=-1)
        axes.plot(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor,linewidth=2,alpha=0.2,zorder=-1)
        axes.scatter(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor, s=1,alpha=0.1,zorder=-1)
    
    
    # opt traj
    cmd = np.argmax(pred[0]['pts_bbox']['ego_fut_cmd'])
    traj_opt = pred[0]['pts_bbox']['ego_fut_mpc'][cmd]
    traj_opt_plot = np.array(traj_opt).cumsum(axis=0)
    traj_opt_plot = np.concatenate((np.zeros((1, traj_opt_plot.shape[1])), traj_opt_plot), axis=0)
    traj_opt_plot = np.stack((traj_opt_plot[:-1], traj_opt_plot[1:]), axis=1)

    plan_vecs = None
    for i in range(traj_opt_plot.shape[0]):
        plan_vec_i = traj_opt_plot[i]
        x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
        y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
        xy = np.stack((x_linspace, y_linspace), axis=1)
        xy = np.stack((xy[:-1], xy[1:]), axis=1)
        if plan_vecs is None:
            plan_vecs = xy
        else:
            plan_vecs = np.concatenate((plan_vecs, xy), axis=0)

    cmap = 'spring'
    y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 601))
    colors = color_map(y[:-1], cmap)
    line_segments = LineCollection(plan_vecs, colors=colors, linewidths=2, linestyles='solid', cmap=cmap, label='MPC Traj')
    axes.add_collection(line_segments)



    axes.plot([-0.9, -0.9], [-2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([-0.9, 0.9], [2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.9, 0.9], [2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.9, -0.9], [-2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.0, 0.0], [0.0, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    
    # nn pred
    cmd = np.argmax(pred[0]['pts_bbox']['ego_fut_cmd'])
    plan_traj = pred[0]['pts_bbox']['ego_fut_preds'][cmd, :, :]
    plan_traj = plan_traj.cumsum(axis=0)
    plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)

    x_coords = [point[0] for point in plan_traj]
    y_coords = [point[1] for point in plan_traj]
    # axes.scatter(x_coords, y_coords, marker='o', color='red', s=4.0, alpha=0.8, zorder=-1)
    nn_pred_scatter = axes.scatter(
        [point[0] for point in plan_traj],
        [point[1] for point in plan_traj],
        marker='o', color='red', s=4.0, alpha=0.8, zorder=-1, label='NN Prediction'
    )

    # gt 
    plan_traj = data['ego_fut_trajs'][0].data[0][0][0]
    plan_traj = plan_traj.cumsum(axis=0)
    plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
    plan_traj = np.stack((plan_traj[:-1], plan_traj[1:]), axis=1)

    plan_vecs = None
    for i in range(plan_traj.shape[0]):
        plan_vec_i = plan_traj[i]
        x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
        y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
        xy = np.stack((x_linspace, y_linspace), axis=1)
        xy = np.stack((xy[:-1], xy[1:]), axis=1)
        if plan_vecs is None:
            plan_vecs = xy
        else:
            plan_vecs = np.concatenate((plan_vecs, xy), axis=0)

    cmap = 'viridis'
    y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 301))
    colors = color_map(y[:-1], cmap)
    # line_segments = LineCollection(plan_vecs, colors=colors, linewidths=1, linestyles='solid', cmap=cmap)
    gt_line_segments = LineCollection(plan_vecs, colors=colors, linewidths=1, linestyles='solid', cmap=cmap, label='Ground Truth')
    axes.add_collection(gt_line_segments)

    collision = pred[0]['pts_bbox']['coll']
    intersect = pred[0]['pts_bbox']['inter']
    axes.text(-8,-9,'collision:'+str(collision))
    axes.text(-8,-7.5,'intersection:'+str(intersect))

    axes.axes.xaxis.set_ticks([])
    axes.axes.yaxis.set_ticks([])
    axes.axis('off')
    axes.legend(loc='lower right', fontsize=6)
    fig.set_tight_layout(True)
    fig.canvas.draw()
    '''

    boxes_pred = data['gt_bboxes_3d'][0].data[0][0]
    boxes_labels = data['gt_labels_3d'][0].data[0][0]
    boxes_trajs = data['gt_attr_labels'][0].data[0][0][:,:12]
    boxes_trajs = boxes_trajs.unsqueeze(1)
    ignore_list = ['barrier', 'bicycle', 'traffic_cone'] 

    for i, (label, traj) in enumerate(zip(boxes_labels, boxes_trajs)):
        box = boxes_pred[i]
        box_render = get_render_box_gt(box, traj, label)
        box_render.render(axes, view=np.eye(4), colors=('tomato', 'tomato', 'tomato'), linewidth=1, box_idx=None)
        mode_idx = [0]
        box_render.render_fut_trajs_grad_color(axes, linewidth=1, mode_idx=mode_idx, fut_ts=6, cmap='autumn')
    
    # show maps
    maps = data['map_gt_bboxes_3d'].data[0][0].fixed_num_sampled_points
    maps_label = data['map_gt_labels_3d'].data[0][0]
    for map, map_label in zip(maps, maps_label):
        points = map.numpy()
        pts_x = np.array([pt[0] for pt in points])
        pts_y = np.array([pt[1] for pt in points]) 
        
        axes.plot(pts_x, pts_y, color=colors_plt[map_label],linewidth=1,alpha=0.8,zorder=-1)
        axes.scatter(pts_x, pts_y, color=colors_plt[map_label],s=1,alpha=0.8,zorder=-1)  
    
    # PRED results
    
    # pred traj
    cmd = np.argmax(pred[0]['pts_bbox']['ego_fut_cmd'])
    # traj_opt = pred[0]['pts_bbox']['ego_fut_mpc'][cmd]
    traj_opt = pred[0]['pts_bbox']['ego_fut_preds'][cmd]
    traj_opt_plot = np.array(traj_opt).cumsum(axis=0)
    traj_opt_plot = np.concatenate((np.zeros((1, traj_opt_plot.shape[1])), traj_opt_plot), axis=0)
    traj_opt_plot = np.stack((traj_opt_plot[:-1], traj_opt_plot[1:]), axis=1)

    plan_vecs = None
    for i in range(traj_opt_plot.shape[0]):
        plan_vec_i = traj_opt_plot[i]
        x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
        y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
        xy = np.stack((x_linspace, y_linspace), axis=1)
        xy = np.stack((xy[:-1], xy[1:]), axis=1)
        if plan_vecs is None:
            plan_vecs = xy
        else:
            plan_vecs = np.concatenate((plan_vecs, xy), axis=0)


    cmap = 'winter'
    y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 601))
    colors = color_map(y[:-1], cmap)
    line_segments = LineCollection(plan_vecs, colors=colors, linewidths=2, alpha=0.8, linestyles='solid', cmap=cmap, label='MPC Traj')
    axes.add_collection(line_segments)

    corridor_preds = pred[0]['pts_bbox']["ego_cor_preds"][0, cmd] # pred
    n_corridor = len(corridor_preds)
    for j, gt_corridor in enumerate(corridor_preds):
        cmap_corridor = 'plasma'
        a = (1 - j/n_corridor) * 0.2 + 0.2
        c = (j/n_corridor) * 0.5 + 0.5
        color_corridor = np.array(plt.cm.get_cmap(cmap_corridor)(c))[:3]
        corridor_points = rect2points(gt_corridor, real=False)  # for epoch_normlized_12.pth
        # corridor_points = rect2points(gt_corridor, real=True)
        axes.plot(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor,linewidth=2,alpha=a,zorder=-1)
        # axes.scatter(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor, s=1,alpha=0.1,zorder=-1)

    axes.plot([-0.9, -0.9], [-2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([-0.9, 0.9], [2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.9, 0.9], [2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.9, -0.9], [-2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.0, 0.0], [0.0, 2], color='mediumseagreen', linewidth=1, alpha=0.8)

    # Add a colorbar for the trajectory
    x_linspace = np.linspace(-28, -28, 301)
    y_linspace = np.linspace(-28, -18, 301)
    xy = np.stack((x_linspace, y_linspace), axis=1)
    xy = np.stack((xy[:-1], xy[1:]), axis=1)
    cmap = 'winter'
    y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 601))
    colors = color_map(y[:-1], cmap)
    line_segments = LineCollection(xy, colors=colors, linewidths=5, alpha=0.8, linestyles='solid', cmap=cmap)
    axes.add_collection(line_segments)
    
    '''
    # Add a colorbar for the corridor
    x_linspace = np.linspace(-25, -25, 301)
    y_linspace = np.linspace(-28, -18, 301)
    xy = np.stack((x_linspace, y_linspace), axis=1)
    xy = np.stack((xy[:-1], xy[1:]), axis=1)
    cmap = 'plasma'
    y = np.linspace(0.5, 1.0, 301)
    colors = color_map(y[:-1], cmap)
    line_segments = LineCollection(xy, colors=colors, linewidths=5, alpha=0.4, linestyles='solid', cmap=cmap)
    axes.add_collection(line_segments)

    axes.text(-29, -31, 't=0s', fontsize=10)
    axes.text(-29, -17, 't=3s', fontsize=10)
    '''

    axes.axes.xaxis.set_ticks([])
    axes.axes.yaxis.set_ticks([])
    axes.axis('off')
    fig.set_tight_layout(True)
    fig.canvas.draw()

    # plt.savefig(savepath+'/normalized/pln_result'+str(index)+'.png', bbox_inches='tight', dpi=500)
    plt.close()
    return

def render_pred_results(data=None,
                    pred=None,
                    savepath: str = None, 
                    colors_plt = ['#96C37D', '#C497B2', '#14517C', 'royalblue'],
                    class2label={'divider':0, 'ped_crossing':1, 'boundary':2, 'centerline': 3}) -> None:
    """
    Render sample data onto axis.
    """
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    plt.xlim(xmin=-30, xmax=30)
    plt.ylim(ymin=-30, ymax=30)

    # directly use render from CustomBox and visualzation.py
    
    boxes_pred = pred[0]['pts_bbox']['boxes_3d']
    boxes_scores = pred[0]['pts_bbox']['scores_3d']
    boxes_labels = pred[0]['pts_bbox']['labels_3d']
    boxes_trajs = pred[0]['pts_bbox']['trajs_3d']
    ignore_list = ['barrier', 'bicycle', 'traffic_cone'] 

    for i, (score, label, traj) in enumerate(zip(boxes_scores, boxes_labels, boxes_trajs)):
        if score < 0.4:
            continue
        box = boxes_pred[i]
        box_render = get_render_box(box, traj, label)
        box_render.render(axes, view=np.eye(4), colors=('tomato', 'tomato', 'tomato'), linewidth=1, box_idx=None, alpha=score.item())
        mode_idx = [0, 1, 2, 3, 4, 5]
        box_render.render_fut_trajs_grad_color(axes, linewidth=1, mode_idx=mode_idx, fut_ts=6, cmap='autumn')

    # show maps
    map_preds = pred[0]['pts_bbox']['map_pts_3d']
    map_labels = pred[0]['pts_bbox']['map_labels_3d']
    map_scores = pred[0]['pts_bbox']['map_scores_3d']
    for map_points, label, score in zip(map_preds, map_labels, map_scores):
        if score < 0.6: 
            continue
        points = map_points.numpy()
        pts_x = np.array([pt[0] for pt in points])
        pts_y = np.array([pt[1] for pt in points]) 
        
        axes.plot(pts_x, pts_y, color=colors_plt[label],linewidth=1,alpha=score.item(),zorder=-1)
        axes.scatter(pts_x, pts_y, color=colors_plt[label],s=1,alpha=0.8,zorder=-1) 
        
    '''
    # Corridors
    # gt_corridors = data['gt_corridor']
    cmd = np.argmax(pred[0]['pts_bbox']['ego_fut_cmd'])
    corridor_preds = pred[0]['pts_bbox']['ego_cor_preds'][-1][cmd]
    n_corridor = len(corridor_preds)
    for j, gt_corridor in enumerate(corridor_preds):
        # cmap_corridor = 'jet'
        cmap_corridor = 'Reds'
        # a = 1-j/n_corridor
        a = (1-j/n_corridor) *0.5 + 0.5
        color_corridor = np.array(plt.cm.get_cmap(cmap_corridor)(a))[:3]
        corridor_points = rect2points(gt_corridor)
        axes.plot(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor,linewidth=2,alpha=0.2,zorder=-1)
        axes.scatter(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor, s=1,alpha=0.1,zorder=-1)
    '''

    # '''
    # Clipped Corridors
    corridor_preds = pred[0]['pts_bbox']["clipped_cor"][0] # pred
    n_corridor = len(corridor_preds)
    for j, gt_corridor in enumerate(corridor_preds):
        cmap_corridor = 'plasma'
        a = (1 - j/n_corridor) * 0.2 + 0.2
        c = (j/n_corridor) * 0.5 + 0.5
        color_corridor = np.array(plt.cm.get_cmap(cmap_corridor)(c))[:3]
        corridor_points = rect2points(gt_corridor, real=True)
        axes.plot(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor,linewidth=2,alpha=a,zorder=-1)
    # '''

    '''    
    # opt traj
    cmd = np.argmax(pred[0]['pts_bbox']['ego_fut_cmd'])
    traj_opt = pred[0]['pts_bbox']['ego_fut_mpc'][cmd]
    traj_opt_plot = np.array(traj_opt).cumsum(axis=0)
    xy = np.stack((traj_opt_plot[:,0], traj_opt_plot[:,1]), axis=1)
    xy = np.stack((xy[:-1], xy[1:]), axis=1)
    y = np.sin(np.linspace(3/2*np.pi, 5/2*np.pi, traj_opt_plot.shape[0]))
    colors = color_map(y[:-1], 'spring')
    line_segments = LineCollection(xy, colors=colors, linewidths=2, cmap='spring')
    axes.add_collection(line_segments)
    '''

    # Show Planning.
    axes.plot([-0.9, -0.9], [-2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([-0.9, 0.9], [2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.9, 0.9], [2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.9, -0.9], [-2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.0, 0.0], [0.0, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    
    cmd = np.argmax(pred[0]['pts_bbox']['ego_fut_cmd'])
    plan_traj = pred[0]['pts_bbox']['ego_fut_preds'][cmd, :, :]
    plan_traj = plan_traj.cumsum(axis=0)
    plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
    plan_traj = np.stack((plan_traj[:-1], plan_traj[1:]), axis=1)

    plan_vecs = None
    for i in range(plan_traj.shape[0]):
        plan_vec_i = plan_traj[i]
        x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
        y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
        xy = np.stack((x_linspace, y_linspace), axis=1)
        xy = np.stack((xy[:-1], xy[1:]), axis=1)
        if plan_vecs is None:
            plan_vecs = xy
        else:
            plan_vecs = np.concatenate((plan_vecs, xy), axis=0)

    cmap = 'viridis'
    y = np.sin(np.linspace(3/2*np.pi, 1/2*np.pi, 301))
    colors = color_map(y[:-1], cmap)
    line_segments = LineCollection(plan_vecs, colors=colors, linewidths=2, alpha=0.8, linestyles='solid', cmap=cmap, label='PRED Traj')
    axes.add_collection(line_segments)

    # Add a colorbar for the trajectory
    x_linspace = np.linspace(-28, -28, 301)
    y_linspace = np.linspace(-28, -18, 301)
    xy = np.stack((x_linspace, y_linspace), axis=1)
    xy = np.stack((xy[:-1], xy[1:]), axis=1)
    cmap = 'viridis'
    y = np.sin(np.linspace(3/2*np.pi, 1/2*np.pi, 301))
    colors = color_map(y[:-1], cmap)
    line_segments = LineCollection(xy, colors=colors, linewidths=5, alpha=0.8, linestyles='solid', cmap=cmap)
    axes.add_collection(line_segments)

    # Add a colorbar for the corridor
    x_linspace = np.linspace(-25, -25, 301)
    y_linspace = np.linspace(-28, -18, 301)
    xy = np.stack((x_linspace, y_linspace), axis=1)
    xy = np.stack((xy[:-1], xy[1:]), axis=1)
    cmap = 'plasma'
    y = np.linspace(0.5, 1.0, 301)
    colors = color_map(y[:-1], cmap)
    line_segments = LineCollection(xy, colors=colors, linewidths=5, alpha=0.4, linestyles='solid', cmap=cmap)
    axes.add_collection(line_segments)

    # axes.text(-29, -31, 't=0s', fontsize=10)
    # axes.text(-29, -17, 't=3s', fontsize=10)
    axes.text(-29, -31, 't=0s', fontsize=10)
    axes.text(-29, -17, 't=3s', fontsize=10)

    axes.axes.xaxis.set_ticks([])
    axes.axes.yaxis.set_ticks([])
    axes.axis('off')
    fig.set_tight_layout(True)
    fig.canvas.draw()
    plt.savefig(savepath+'/bev_pred.png', bbox_inches='tight', dpi=200)
    plt.close()
    return

def render_pln_results(data=None,
                    pred=None,
                    savepath: str = None, 
                    colors_plt = ['#96C37D', '#C497B2', '#14517C', 'royalblue'],
                    class2label={'divider':0, 'ped_crossing':1, 'boundary':2, 'centerline': 3}) -> None:
    """
    Render sample data onto axis.
    """
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    plt.xlim(xmin=-30, xmax=30)
    plt.ylim(ymin=-30, ymax=30)

    # directly use render from CustomBox and visualzation.py
    
    boxes_pred = data['gt_bboxes_3d'][0].data[0][0]
    boxes_labels = data['gt_labels_3d'][0].data[0][0]
    boxes_trajs = data['gt_attr_labels'][0].data[0][0][:,:12]
    boxes_trajs = boxes_trajs.unsqueeze(1)
    ignore_list = ['barrier', 'bicycle', 'traffic_cone'] 

    for i, (label, traj) in enumerate(zip(boxes_labels, boxes_trajs)):
        box = boxes_pred[i]
        box_render = get_render_box_gt(box, traj, label)
        box_render.render(axes, view=np.eye(4), colors=('tomato', 'tomato', 'tomato'), linewidth=1, box_idx=None)
        mode_idx = [0]
        box_render.render_fut_trajs_grad_color(axes, linewidth=1, mode_idx=mode_idx, fut_ts=6, cmap='autumn')
    
    # show maps
    maps = data['map_gt_bboxes_3d'].data[0][0].fixed_num_sampled_points
    maps_label = data['map_gt_labels_3d'].data[0][0]
    for map, map_label in zip(maps, maps_label):
        points = map.numpy()
        pts_x = np.array([pt[0] for pt in points])
        pts_y = np.array([pt[1] for pt in points]) 
        
        axes.plot(pts_x, pts_y, color=colors_plt[map_label],linewidth=1,alpha=0.8,zorder=-1)
        axes.scatter(pts_x, pts_y, color=colors_plt[map_label],s=1,alpha=0.8,zorder=-1)  
        
    '''
    # Corridors
    # gt_corridors = data['gt_corridor']
    cmd = np.argmax(pred[0]['pts_bbox']['ego_fut_cmd'])
    corridor_preds = pred[0]['pts_bbox']['ego_cor_preds'][-1][cmd]
    n_corridor = len(corridor_preds)
    for j, gt_corridor in enumerate(corridor_preds):
        # cmap_corridor = 'jet'
        cmap_corridor = 'Reds'
        # a = 1-j/n_corridor
        a = (1-j/n_corridor) *0.5 + 0.5
        color_corridor = np.array(plt.cm.get_cmap(cmap_corridor)(a))[:3]
        corridor_points = rect2points(gt_corridor)
        axes.plot(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor,linewidth=2,alpha=0.2,zorder=-1)
        axes.scatter(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor, s=1,alpha=0.1,zorder=-1)
    '''
    
    # PRED results
    corridor_preds = pred[0]['pts_bbox']["clipped_cor"][0] # pred
    n_corridor = len(corridor_preds)
    for j, gt_corridor in enumerate(corridor_preds):
        cmap_corridor = 'plasma'
        a = (1 - j/n_corridor) * 0.2 + 0.2
        c = (j/n_corridor) * 0.5 + 0.5
        color_corridor = np.array(plt.cm.get_cmap(cmap_corridor)(c))[:3]
        corridor_points = rect2points(gt_corridor, real=True)
        axes.plot(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor,linewidth=2,alpha=a,zorder=-1)
        # axes.scatter(corridor_points[:, 0], corridor_points[:, 1], color=color_corridor, s=1,alpha=0.1,zorder=-1)

    '''    
    # opt traj
    cmd = np.argmax(pred[0]['pts_bbox']['ego_fut_cmd'])
    traj_opt = pred[0]['pts_bbox']['ego_fut_mpc'][cmd]
    traj_opt_plot = np.array(traj_opt).cumsum(axis=0)
    xy = np.stack((traj_opt_plot[:,0], traj_opt_plot[:,1]), axis=1)
    xy = np.stack((xy[:-1], xy[1:]), axis=1)
    y = np.sin(np.linspace(3/2*np.pi, 5/2*np.pi, traj_opt_plot.shape[0]))
    colors = color_map(y[:-1], 'spring')
    line_segments = LineCollection(xy, colors=colors, linewidths=2, cmap='spring')
    axes.add_collection(line_segments)
    '''

    # pred traj
    cmd = np.argmax(pred[0]['pts_bbox']['ego_fut_cmd'])
    traj_opt = pred[0]['pts_bbox']['ego_fut_mpc'][cmd]
    traj_opt_plot = np.array(traj_opt).cumsum(axis=0)
    traj_opt_plot = np.concatenate((np.zeros((1, traj_opt_plot.shape[1])), traj_opt_plot), axis=0)
    traj_opt_plot = np.stack((traj_opt_plot[:-1], traj_opt_plot[1:]), axis=1)

    plan_vecs = None
    for i in range(traj_opt_plot.shape[0]):
        plan_vec_i = traj_opt_plot[i]
        x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
        y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
        xy = np.stack((x_linspace, y_linspace), axis=1)
        xy = np.stack((xy[:-1], xy[1:]), axis=1)
        if plan_vecs is None:
            plan_vecs = xy
        else:
            plan_vecs = np.concatenate((plan_vecs, xy), axis=0)


    cmap = 'winter'
    y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 601))
    colors = color_map(y[:-1], cmap)
    line_segments = LineCollection(plan_vecs, colors=colors, linewidths=2, alpha=0.8, linestyles='solid', cmap=cmap, label='MPC Traj')
    axes.add_collection(line_segments)

    axes.plot([-0.9, -0.9], [-2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([-0.9, 0.9], [2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.9, 0.9], [2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.9, -0.9], [-2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.0, 0.0], [0.0, 2], color='mediumseagreen', linewidth=1, alpha=0.8)

    # Add a colorbar for the trajectory
    x_linspace = np.linspace(-28, -28, 301)
    y_linspace = np.linspace(-28, -18, 301)
    xy = np.stack((x_linspace, y_linspace), axis=1)
    xy = np.stack((xy[:-1], xy[1:]), axis=1)
    cmap = 'winter'
    y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 601))
    colors = color_map(y[:-1], cmap)
    line_segments = LineCollection(xy, colors=colors, linewidths=5, alpha=0.8, linestyles='solid', cmap=cmap)
    axes.add_collection(line_segments)

    # Add a colorbar for the corridor
    x_linspace = np.linspace(-25, -25, 301)
    y_linspace = np.linspace(-28, -18, 301)
    xy = np.stack((x_linspace, y_linspace), axis=1)
    xy = np.stack((xy[:-1], xy[1:]), axis=1)
    cmap = 'plasma'
    y = np.linspace(0.5, 1.0, 301)
    colors = color_map(y[:-1], cmap)
    line_segments = LineCollection(xy, colors=colors, linewidths=5, alpha=0.4, linestyles='solid', cmap=cmap)
    axes.add_collection(line_segments)

    axes.text(-29, -31, 't=0s', fontsize=10)
    axes.text(-29, -17, 't=3s', fontsize=10)

    axes.axes.xaxis.set_ticks([])
    axes.axes.yaxis.set_ticks([])
    axes.axis('off')
    fig.set_tight_layout(True)
    fig.canvas.draw()
    plt.savefig(savepath+'/bev_pred.png', bbox_inches='tight', dpi=200)
    plt.close()
    return

# Ensure all corridor_points have positive y values
def interpolate_positive_y(corridor_points):
    interpolated_points = []
    for i in range(corridor_points.shape[0] - 1):
        p1 = corridor_points[i]
        p2 = corridor_points[i + 1]
        
        # Check if both points have negative y values
        if p1[1] < 0 and p2[1] < 0:
            # Skip adding these segments
            continue
        elif p1[1] < 0 or p2[1] < 0:
            # Interpolate the point where y crosses zero
            t = abs(p1[1]) / (abs(p1[1]) + abs(p2[1]))
            p_intersect = p1 + t * (p2 - p1)
            p_intersect[1] = 0  # Force y = 0 for intersection
            
            if p1[1] < 0:
                interpolated_points.append(p_intersect)
                interpolated_points.append(p2)
            else:
                interpolated_points.append(p1)
                interpolated_points.append(p_intersect)
        else:
            interpolated_points.append(p1)
            interpolated_points.append(p2)
    
    return np.array(interpolated_points)

def render_result(result, data, nusc, i):
    mmcv.mkdir_or_exist(out_path)
    # GT    
    # render_sample_data(data=data, pred=result, savepath=out_path)
    # bev_gt_path = osp.join(out_path, 'bev_pred.png')
    # bev_gt_img = cv2.imread(bev_gt_path)
    # bev_pred_img = cv2.resize(bev_gt_img, (bev_gt_img.shape[0]//2*3, bev_gt_img.shape[1]//2*3))
    # os.remove(bev_gt_path)

    # render_sample_data_analysis(data=data, pred=result, savepath=out_path, index=i)

    render_pred_results(data=data, pred=result, savepath=out_path)
    bev_gt_path = osp.join(out_path, 'bev_pred.png')
    bev_gt_img = cv2.imread(bev_gt_path)
    bev_pred_img = cv2.resize(bev_gt_img, (bev_gt_img.shape[0]//2*3, bev_gt_img.shape[1]//2*3))
    os.remove(bev_gt_path)

    render_pln_results(data=data, pred=result, savepath=out_path)
    bev_gt_path = osp.join(out_path, 'bev_pred.png')
    bev_gt_img = cv2.imread(bev_gt_path)
    bev_pln_img = cv2.resize(bev_gt_img, (bev_gt_img.shape[0]//2*3, bev_gt_img.shape[1]//2*3))
    os.remove(bev_gt_path)

    cams = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
    image_names = data['img_metas'][0].data[0][0]['filename']
    sample_token = data['img_metas'][0].data[0][0]['sample_idx']
    sample = nusc.get('sample', sample_token)

    cam_imgs = []
    for i, cam in enumerate(cams):
        # cam_info = sample_info['cams'][cam]
        cam_path = image_names[i]

        if os.path.exists(cam_path):
            img = Image.open(cam_path)
        else:
            print(f"Image not found: {cam_path}")
            img = Image.new('RGB', image_size, (255, 255, 255))
        _, ax = plt.subplots(1, 1, figsize=(6, 12))
        # img = draw_boxes(img, i, result, data)
        ax.imshow(img)

        # show traj on front
        if cam == 'CAM_FRONT':
            lidar_sd_record =  nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_cs_record = nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
            lidar_pose_record = nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])

            # get plan traj [x,y,z,w] quaternion, w=1
            # we set z=-1 to get points near the ground in lidar coord system
            cmd = np.argmax(result[0]['pts_bbox']['ego_fut_cmd'])
            plan_traj = result[0]['pts_bbox']['ego_fut_mpc'][cmd, :, :]
            plan_traj[abs(plan_traj) < 0.01] = 0.0
            plan_traj = plan_traj.cumsum(axis=0)

            plan_traj = np.concatenate((
                plan_traj[:, [0]],
                plan_traj[:, [1]],
                -1.5*np.ones((plan_traj.shape[0], 1)),
                np.ones((plan_traj.shape[0], 1)),
            ), axis=1)
            # add the start point in lcf
            plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
            # plan_traj[0, :2] = 2*plan_traj[1, :2] - plan_traj[2, :2]
            plan_traj[0, 0] = 0.3
            plan_traj[0, 2] = -1.5
            plan_traj[0, 3] = 1.0

            l2e_r = lidar_cs_record['rotation']
            l2e_t = lidar_cs_record['translation']
            e2g_r = lidar_pose_record['rotation']
            e2g_t = lidar_pose_record['translation']
            sample_data_token = sample['data'][cam]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix
            s2l_r, s2l_t = obtain_sensor2top(nusc, sample_data_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(s2l_r)
            lidar2cam_t = s2l_t @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            viewpad = np.eye(4)
        
            sd_record = nusc.get('sample_data', sample_data_token)
            cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            camera_intrinsic = np.array(cs_record['camera_intrinsic'])
            viewpad[:camera_intrinsic.shape[0], :camera_intrinsic.shape[1]] = camera_intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            plan_traj = lidar2img_rt @ plan_traj.T
            plan_traj = plan_traj[0:2, ...] / np.maximum(
                plan_traj[2:3, ...], np.ones_like(plan_traj[2:3, ...]) * 1e-5)
            plan_traj = plan_traj.T
            plan_traj = np.stack((plan_traj[:-1], plan_traj[1:]), axis=1)


            # corridor 
            corridor_preds = result[0]['pts_bbox']["clipped_cor"][0] # pred
            n_corridor = len(corridor_preds)
            for ic in range(n_corridor-1, -1, -1): 
                corridor = corridor_preds[ic]
                corridor_points = rect2points(corridor, real=True)
                corridor_points = interpolate_positive_y(corridor_points)
                corridor_points = np.concatenate((
                    corridor_points[:, [0]],
                    corridor_points[:, [1]],
                    -1.5*np.ones((corridor_points.shape[0], 1)),
                    np.ones((corridor_points.shape[0], 1)),
                ), axis=1)
                corridor_points = lidar2img_rt @ corridor_points.T
                corridor_points = corridor_points[0:2, ...] / np.maximum(
                    corridor_points[2:3, ...], np.ones_like(corridor_points[2:3, ...]) * 1e-5)
                corridor_points = corridor_points.T
                corridor_points = np.stack((corridor_points[:-1], corridor_points[1:]), axis=1)
                # corridor
                plan_vecs = None
                for i in range(corridor_points.shape[0]):
                    plan_vec_i = corridor_points[i]
                    x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
                    y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
                    xy = np.stack((x_linspace, y_linspace), axis=1)
                    xy = np.stack((xy[:-1], xy[1:]), axis=1)
                    if plan_vecs is None:
                        plan_vecs = xy
                    else:
                        plan_vecs = np.concatenate((plan_vecs, xy), axis=0)

                cmap_corridor = 'plasma'
                a = (1 - ic/n_corridor) * 0.2 + 0.2
                c = (ic/n_corridor) * 0.5 + 0.5
                colors = np.array(plt.cm.get_cmap(cmap_corridor)(c))[:3]
                line_segments = LineCollection(plan_vecs, colors=colors, alpha=a, linewidths=3, linestyles='solid', cmap=cmap_corridor)
                ax.add_collection(line_segments)

            # planning
            plan_vecs = None
            for i in range(plan_traj.shape[0]):
                plan_vec_i = plan_traj[i]
                x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
                y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
                xy = np.stack((x_linspace, y_linspace), axis=1)
                xy = np.stack((xy[:-1], xy[1:]), axis=1)
                if plan_vecs is None:
                    plan_vecs = xy
                else:
                    plan_vecs = np.concatenate((plan_vecs, xy), axis=0)

            
            cmap = 'winter'
            y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 601))
            colors = color_map(y[:-1], cmap)
            line_segments = LineCollection(plan_vecs, colors=colors, linewidths=2, linestyles='solid', cmap=cmap)
            ax.add_collection(line_segments)

    
        ax.set_xlim(0, img.size[0])
        ax.set_ylim(img.size[1], 0)
        ax.axis('off')
        if out_path is not None:
            savepath = osp.join(out_path, f'{cam}_PRED')
            plt.savefig(savepath, bbox_inches='tight', dpi=200, pad_inches=0.0)
        plt.close()

        # Load boxes and image.
        data_path = osp.join(out_path, f'{cam}_PRED.png')
        cam_img = cv2.imread(data_path)

        if cam == 'CAM_FRONT_LEFT':
            # Keep only the right half
            cam_img = cam_img[:, cam_img.shape[1] * 2 // 6 : cam_img.shape[1] * 5 // 6, :]
        elif cam == 'CAM_FRONT_RIGHT':
            # Keep only the left half
            cam_img = cam_img[:, cam_img.shape[1] * 1 // 6 : cam_img.shape[1] * 4 // 6, :]

        
        # lw = 6
        # tf = max(lw - 3, 1)
        # w, h = cv2.getTextSize(cam, 0, fontScale=lw / 6, thickness=tf)[0]  # text width, height
        # # color=(0, 0, 0)
        # txt_color=(255, 255, 255)
        # cv2.putText(cam_img,
        #             cam, (10, h + 10),
        #             0,
        #             lw / 6,
        #             txt_color,
        #             thickness=tf,
        #             lineType=cv2.LINE_AA)
        
        cam_imgs.append(cam_img)

    '''
    # clip the pred_img
    blank_image = np.ones((cam_imgs[0].shape[0]*2, bev_pred_img.shape[1], 3) ,dtype=np.uint8) * 255
    blank_shape = blank_image.shape
    start_x = (blank_shape[1] - bev_pred_img.shape[1]) // 2 
    start_y = (blank_shape[0] - bev_pred_img.shape[0]) // 2
    if start_y > 0:
        background = blank_image.copy()
        background[start_y:start_y+bev_pred_img.shape[0], start_x: start_x + bev_pred_img.shape[1]] = bev_pred_img
        bev_pred_img = background


    else:
        start_y = -start_y //2 
        background = blank_image.copy()
        background[0:blank_image.shape[0], start_x: start_x + bev_pred_img.shape[1]] = bev_pred_img[start_y:start_y+blank_image.shape[0],:]
        bev_pred_img = background
    '''

    
    plan_cmd = np.argmax(result[0]['pts_bbox']['ego_fut_cmd'])
    cmd_list = ['Turn Right', 'Turn Left', 'Go Straight']
    plan_cmd_str = cmd_list[plan_cmd]
    # pred_img = cv2.copyMakeBorder(pred_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Line thickness of 2 px
    thickness = 3
    # org
    org = (20, 40)      
    # Blue color in BGR
    color = (0, 0, 0)
    # Using cv2.putText() method
    bev_pred_img = cv2.putText(bev_pred_img, 'PREDICTION', org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    bev_pln_img = cv2.putText(bev_pln_img, 'GROUND TRUTH', org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    '''
    bev_pred_img = cv2.putText(bev_pred_img, plan_cmd_str, (20, blank_shape[0] - 100), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    '''
                    

    # compose 
    # cam_top = cv2.hconcat([cam_imgs[2],cam_imgs[0],cam_imgs[1]])
    # cam_down = cv2.hconcat([cam_imgs[4], cam_imgs[3],cam_imgs[5]])
    # cam_left = cv2.vconcat([cam_top, cam_down])
    # vis_img = cv2.hconcat([cam_left, bev_pred_img]) 

    cam_top = cv2.hconcat([cam_imgs[2], cam_imgs[0], cam_imgs[1]])
    target_width = cam_top.shape[1] // 2

    border_thickness = 2  # Thickness of the border in pixels
    border_color = (0, 0, 0)  # Black color in BGR format

    cam_top_border = add_border(cam_top, border_thickness, border_color)
    bev_pred_img_with_border = add_border(bev_pred_img, border_thickness, border_color)
    bev_pln_img_with_border = add_border(bev_pln_img, border_thickness, border_color)

    # Add borders to the resized BEV images
    # bev_pred_img_with_border = cv2.copyMakeBorder(
    #     bev_pred_img,
    #     border_thickness, border_thickness, border_thickness, border_thickness//2,
    #     cv2.BORDER_CONSTANT,
    #     value=border_color
    # )

    # bev_pln_img_with_border = cv2.copyMakeBorder(
    #     bev_pln_img,
    #     border_thickness, border_thickness, border_thickness//2, border_thickness,
    #     cv2.BORDER_CONSTANT,
    #     value=border_color
    # )

    bev_pred_img_resized = cv2.resize(bev_pred_img_with_border, (target_width, target_width))
    bev_pln_img_resized = cv2.resize(bev_pln_img_with_border, (target_width, target_width))
    cam_down = cv2.hconcat([bev_pred_img_resized, bev_pln_img_resized])
    vis_img = cv2.vconcat([cam_top_border, cam_down])


    return vis_img


def process(model, data_loader, cfg, samples):
    model.eval()
    results = []
    datas = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path = "pred_vis.mp4"
    video = cv2.VideoWriter(video_path, fourcc, 10, (1860, 1453), True)
    # video = cv2.VideoWriter(video_path, fourcc, 10, (1167, 1170), True)
    
    # begin loop
    for i, data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            if i == samples:
                break
            result = model(return_loss=False, rescale=True, **data)
            vis_img = render_result(result, data, nusc, i)
            #  single test
            # cv2.imwrite("visuals/pred_results_"+str(i)+".png", vis_img)  
            # video
            video.write(vis_img)
    # end loop

    video.release()
    cv2.destroyAllWindows()



def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--samples', type=int, default=10, help='number of samples to process')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        assert False
        # model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        process(model, data_loader, cfg, args.samples)



if __name__ == '__main__':
    main()



