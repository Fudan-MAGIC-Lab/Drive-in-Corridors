import torch
from torch import nn
import numpy as np
from projects.DifferentiableLQR.qpth.qp import QPFunction
from projects.DifferentiableLQR.qpth.qp import QPSolvers
import matplotlib.pyplot as plt
import math


class QPControl(nn.Module):
	'''
	  min 1/2 u^T R u + (x - x_ref)^T Qx (x - x_ref)
		Quadratic Programming Control
		x: [x, y, theta, v]
		u: [a, delta]
	'''
	def __init__(self):
		super(QPControl, self).__init__()
		self.weight = nn.Parameter(torch.ones(2), requires_grad=False)

		self.dt = 0.25
		self.horizon = 12 # horizon
		self.n_state = 4 # state
		self.n_ctrl = 2 # control
		self.rhoN = 5.0
		self.rhoS = 1.0
		self.L = 2.0  # nuscenes, same as metric_bp.py
		self.theta_max = np.pi / 2

		self.v_min = -0.1
		self.v_max = 15.0
		self.a_max = 4.0  # TODO, obtain in dataset
		self.delta_max = 1.0 # np.pi / 3
		self.ddelta_max = 1.0
		self.l_front = 4.084/2 + 0.5
		self.l_rear = 4.084/2 - 0.5 
		self.w = 1.85
		self.initMatrices()

		self.lidar2ego_t = torch.tensor([0.943713, 0.0, 1.84023])
		self.lidar2ego_r = torch.tensor([[ 0.00203327,  0.99970406,  0.02424172],
										[-0.99998053,  0.00217566, -0.00584864],
										[-0.00589965, -0.02422936,  0.99968902]])
		self.plot = False
		

		
	def initMatrices(self):
		n, m, N = self.n_state, self.n_ctrl, self.horizon
		ns = 4*4*N//2
		nz = m*N + 4*4*N//2
		self.Qx_ = torch.zeros(n*N, n*N)
		self.Ru_ = torch.eye(m*N)
		# state & control constraints
		# lx_ < = Cx* <= ux_
		# 
		self.Cx_ = torch.zeros(N, n*N)
		self.lx_ = torch.zeros(N, 1)
		self.ux_ = torch.zeros(N, 1)
		self.Cu_ = torch.zeros(3*N, m*N)
		self.lu_ = torch.zeros(3*N, 1)
		self.uu_ = torch.zeros(3*N, 1)
		# p_corners = C_cor_ * X + d_cor_
		# assume theta ~ 0, R = [[1 -theta],[theta 1]]
		# TODO theta !~ 0
		# 8 is the number of corners points
		self.C_cor_ = torch.zeros(8*N//2, n*N)  # only constrain T = 0.5 * k
		self.d_cor_ = torch.zeros(8*N//2)
		Qs = torch.eye(ns)   # 4 points * 4 planes
		self.Q_ = torch.zeros(nz, nz)
		self.Q_[m*N:, m*N:] = Qs
		self.p_ = torch.zeros(nz)
		for i in range(N):
			# only cost on T=0.5
			if i%2 != 0:
				self.Qx_[i*n+0, i*n+0] = 1  # tracking x
				self.Qx_[i*n+1, i*n+1] = 1  # tracking y
			# if i!=N-1:  # to make Q semi positive definite @N
			# 	self.Qx_[i*n+2, i*n+2] = 0  # theta tracking
			# 	self.Qx_[i*n+3, i*n+3] = 0  # v tracking

			self.Cu_[i * 3 + 0, i * m + 0] = 1
			self.Cu_[i * 3 + 1, i * m + 1] = 1
			self.Cu_[i * 3 + 2, i * m + 1] = 1
			self.lu_[i * 3 + 0, 0] = -self.a_max
			self.uu_[i * 3 + 0, 0] = self.a_max
			self.lu_[i * 3 + 1, 0] = -self.delta_max
			self.uu_[i * 3 + 1, 0] = self.delta_max
			self.lu_[i * 3 + 2, 0] = -self.ddelta_max * self.dt
			self.uu_[i * 3 + 2, 0] = self.ddelta_max * self.dt
			if i > 0:
				self.Cu_[i * 3 + 2, (i - 1) * m + 1] = -1

			self.Cx_[i, i*n + 3] = 1  # v
			self.lx_[i] = self.v_min
			self.ux_[i] = self.v_max
			# i = 0, t = 0.25; i = 1, t = 0.5
			if i%2 != 0:
				j = i//2
				# FL.x
				self.C_cor_[8*j+0, n*i+0] = 1
				self.C_cor_[8*j+0, n*i+2] = -self.w/2
				self.d_cor_[8*j+0] = self.l_front
				# FL.y
				self.C_cor_[8*j+1, n*i+1] = 1
				self.C_cor_[8*j+1, n*i+2] = self.l_front
				self.d_cor_[8*j+1] = self.w / 2
				# FR.x
				self.C_cor_[8*j+2, n*i+0] = 1
				self.C_cor_[8*j+2, n*i+2] = self.w/2
				self.d_cor_[8*j+2] = self.l_front
				# FR.y
				self.C_cor_[8*j+3, n*i+1] = 1
				self.C_cor_[8*j+3, n*i+2] = self.l_front
				self.d_cor_[8*j+3] = -self.w / 2
				# RL.x
				self.C_cor_[8*j+4, n*i+0] = 1
				self.C_cor_[8*j+4, n*i+2] = -self.w/2
				self.d_cor_[8*j+4] = -self.l_rear
				# RL.y
				self.C_cor_[8*j+5, n*i+1] = 1
				self.C_cor_[8*j+5, n*i+2] = -self.l_rear
				self.d_cor_[8*j+5] = self.w / 2
				# RR.x
				self.C_cor_[8*j+6, n*i+0] = 1
				self.C_cor_[8*j+6, n*i+2] = self.w/2
				self.d_cor_[8*j+6] = -self.l_rear
				# RR.y
				self.C_cor_[8*j+7, n*i+1] = 1
				self.C_cor_[8*j+7, n*i+2] = -self.l_rear
				self.d_cor_[8*j+7] = -self.w / 2
			
		self.Qx_[N*n - 4, N*n - 4] = self.rhoN
		self.Qx_[N*n - 3, N*n - 3] = self.rhoN	
		# to make Q semi positive definite @N
		self.Qx_[N*n - 2, N*n - 2] = 1
		self.Qx_[N*n - 1, N*n - 1] = 1
		

	def diff(self, state, input):
			phi = state[2]
			v = state[3]
			a = input[0]
			delta = input[1]
			ds = torch.zeros_like(state)
			ds[0] = v * torch.cos(phi)
			ds[1] = v * torch.sin(phi)
			ds[2] = v / self.L * torch.tan(delta)
			ds[3] = a
			return ds
	
	# Runge-Kutta, RK4
	def kinematics_forward(self, state, input):
			k1 = self.diff(state, input)
			k2 = self.diff(state + k1 * self.dt / 2, input)
			k3 = self.diff(state + k2 * self.dt / 2, input)
			k4 = self.diff(state + k3 * self.dt, input)
			state = state + (k1 + k2 * 2 + k3 * 2 + k4) * self.dt / 6
			return state
	
	def linearization(self, state, control):
		'''
			linearize the dynamics
			x_k+1 = A * x_k + B * u_k + g
		'''
		x, y, theta, v = state
		a, delta = control
		A = torch.tensor([
			[1, 0, -v * torch.sin(theta) * self.dt, torch.cos(theta) * self.dt],
			[0, 1, v * torch.cos(theta) * self.dt, torch.sin(theta) * self.dt],
			[0, 0, 1, torch.tan(delta) / self.L * self.dt],
			[0, 0, 0, 1]
		]).to(state.device)
		B = torch.tensor([
			[0, 0],
			[0, 0],
			[0, v / self.L / torch.cos(theta) / torch.cos(theta) * self.dt],
			[self.dt, 0]
		]).to(state.device)
		g = torch.tensor([
            [v * torch.sin(theta) * self.dt * theta],
            [-v * torch.cos(theta) * self.dt * theta],
            [-v / self.L / torch.cos(delta) / torch.cos(delta) * self.dt * delta],
            [0]
        ]).to(state.device)
		return A, B, g

	def lidar2ego_traj(self, tracking_traj):
		# tracking_traj [B, 6, 2]
		
		# Convert 2D coordinates to 3D by adding a z-coordinate (0)
		B, N, _ = tracking_traj.shape  # B = batch size, N = 6
		zero = torch.zeros(B, N, 1).to(tracking_traj.device)
		traj_3d = torch.cat([tracking_traj, zero], dim=-1)  # [B, 6, 3]
		
		# Apply rotation and translation to transform points to the ego frame
		rotated_traj = torch.einsum('ij,bnj->bni', self.lidar2ego_r, traj_3d)
		transformed_traj = rotated_traj + self.lidar2ego_t  # Broadcast translation [3] to [B, 6, 3]
		
		# Return transformed coordinates, keeping only the x and y components
		return transformed_traj[..., :2]  # [B, 6, 2]
	
	def ego2lidar_traj(self, traj_ego):
    	# traj_ego [B, 6, 2]
		
		# Convert 2D coordinates to 3D by adding a z-coordinate (0)
		B, N, _ = traj_ego.shape  # B = batch size, N = 6
		zeros = torch.zeros(B, N, 1).to(traj_ego.device)
		traj_3d = torch.cat([traj_ego, zeros], dim=-1)  # [B, 6, 3]
		
		# Compute the inverse rotation and inverse translation
		inv_rotation = self.lidar2ego_r.T  # Transpose of rotation matrix is its inverse
		inv_translation = -self.lidar2ego_t
		
		# Apply inverse translation first, then inverse rotation
		translated_traj = traj_3d + inv_translation  # Broadcast translation [3] to [B, 6, 3]
		lidar_traj = torch.einsum('ij,bnj->bni', inv_rotation, translated_traj)
		
		# Return transformed coordinates, keeping only the x and y components
		return lidar_traj[..., :2]  # [B, 6, 2]

	def lidar2ego_rect(self, rect):
		# rect [B, T, 5]
		# Extract parameters
		center_x = rect[..., 0]
		center_y = rect[..., 1]
		yaw = rect[..., 2]
		length = rect[..., 3]
		width = rect[..., 4]

		# Calculate half-dimensions for corner offset calculation
		half_length = length / 2
		half_width = width / 2

		# Calculate the corners of the rectangle relative to the center
		# Rotation matrix in 2D for the yaw angle
		cos_yaw = torch.cos(yaw)
		sin_yaw = torch.sin(yaw)

		# Define corners in local frame
		corners = torch.stack([
			torch.stack([half_length, half_width], dim=-1),     # front-right
			torch.stack([half_length, -half_width], dim=-1),    # front-left
			torch.stack([-half_length, -half_width], dim=-1),   # back-left
			torch.stack([-half_length, half_width], dim=-1),    # back-right
		], dim=-2)  # Shape: [4, 2]

		# Apply rotation to corners
		rot_matrix = torch.stack([
			torch.stack([cos_yaw, -sin_yaw], dim=-1),
			torch.stack([sin_yaw, cos_yaw], dim=-1),
		], dim=-2)  # Shape: [B, T, 2, 2]

		rotated_corners = torch.einsum('...ij,...kj->...ki', rot_matrix, corners)  # [B, T, 4, 2]
		# Add center coordinates to the rotated corners
		corners_lidar = rotated_corners + torch.stack([center_x, center_y], dim=-1).unsqueeze(-2)  # [B, T, 4, 2]

		# Transform each corner to the ego frame by extending to 3D and applying lidar2ego transformation
		corners_lidar_3d = torch.cat([corners_lidar, torch.zeros_like(corners_lidar[..., :1])], dim=-1)  # [B, T, 4, 3]
		corners_ego_3d = torch.einsum('ij,...kj->...ki', self.lidar2ego_r, corners_lidar_3d) + self.lidar2ego_t

		# New center in ego frame is the mean of the transformed corners
		center_ego = corners_ego_3d[..., :2].mean(dim=-2)  # [B, T, 2]

		# Calculate new yaw in ego frame based on the direction from back-left to front-left corner
		delta_x = corners_ego_3d[..., 1, 0] - corners_ego_3d[..., 2, 0]
		delta_y = corners_ego_3d[..., 1, 1] - corners_ego_3d[..., 2, 1]
		yaw_ego = torch.atan2(delta_y, delta_x)  # [B, T]

		# Stack results into the same format as input
		rect_ego = torch.stack([center_ego[..., 0], center_ego[..., 1], yaw_ego, length, width], dim=-1)  # [B, T, 5]
		return rect_ego


	
	def forward(self, canbus, ego_lcf_feat, tracking_traj, corridor_rect, train=True):
		'''
			Version 1: minimum control effort with tracking
			min 1/2 u^T R u + (x - x_ref)^T Q (x - x_ref)
			s.t.  x_{t+1} = f(x_t, u_t) --> linearize
				  -u_max < u < u_max
				  x0 = x_init
			tracking_traj: torch.tensor[T, 2] x,y
		'''
		# insert 0.0 at the beginning
		n_batch = tracking_traj.shape[0]
		dType = tracking_traj.dtype
		device = tracking_traj.device
		self.lidar2ego_t = self.lidar2ego_t.to(device)
		self.lidar2ego_r = self.lidar2ego_r.to(device)
		tracking_traj = tracking_traj.cumsum(axis=1)

		weight_normed = self.weight # weight on Ru, rhos, self.weight(2) unused
		weight_u = weight_normed[0]
		weight_s = weight_normed[1]

		if self.plot:
			tracking_traj_debug = tracking_traj
			corridor_debug = corridor_rect

		tracking_traj = self.lidar2ego_traj(tracking_traj)
		n, m, N = self.n_state, self.n_ctrl, self.horizon
		ns = 4*4*N//2
		nz = m*N + 4*4*N//2
		n_ineq = 4 # v a delta ddelta
		yaw = 0.0
		# ego_lcf_feat[0,0,0,8] = 2 * steering / 2.588
		# steer = ego_lcf_feat[0, 0, 0, 8] * 2.588 / 2
		yaw, steer = 0.0, 0.0
		
		# x0 = torch.tensor([0.0, 0.0, yaw, canbus[13]], dtype=dType, device=device)
		x0 = torch.tensor([self.lidar2ego_t[0], self.lidar2ego_t[1], yaw, canbus[13]], dtype=dType, device=device)
		x0 = x0.unsqueeze(1)
		u0 = torch.tensor([canbus[7], steer], dtype=dType, device=device)
		u0 = u0.unsqueeze(1)
		n_cor, p_cor = self.covertCorridors(corridor_rect)
		# n_cor * x <= p_cor, n_cor: [B, 4N, 2], p_cor: [B, N, 4]
		# applied to all corners, 4 corridor edges, 4 corner points (x, y)
		A_cor = torch.zeros(n_batch, ns, 8*N//2, device=device)
		b_cor = torch.zeros(n_batch, ns, device=device)
		I_cor = torch.eye(ns).unsqueeze(0).repeat(n_batch, 1, 1).to(device) # for s
		for i in range(N//2):
			A_cor[:, 16*i:16*i+4, 8*i:8*i+2] = n_cor[:, i] 
			b_cor[:, 16*i:16*i+4] = p_cor[:, i]
			A_cor[:, 16*i+4:16*i+8, 8*i+2:8*i+4] = n_cor[:, i] 
			b_cor[:, 16*i+4:16*i+8] = p_cor[:, i]
			A_cor[:, 16*i+8:16*i+12, 8*i+4:8*i+6] = n_cor[:, i] 
			b_cor[:, 16*i+8:16*i+12] = p_cor[:, i]
			A_cor[:, 16*i+12:16*i+16, 8*i+6:8*i+8] = n_cor[:, i] 
			b_cor[:, 16*i+12:16*i+16] = p_cor[:, i]


		BB = torch.zeros(n_batch, n*N, m*N, dtype=dType, device=device)
		AA = torch.zeros(n_batch, n*N, n, dtype=dType, device=device)
		gg = torch.zeros(n_batch, n*N, 1, dtype=dType, device=device)
		qx = torch.zeros(n_batch, n*N, 1, dtype=dType, device=device)
		xi = x0
		# for state constraints theta, v
		Cx = torch.zeros(n_batch, N, n*N, dtype=dType, device=device)  
		for i in range(N):
			# TODO prediction states
			A, B, g = self.linearization(xi, u0)
			# calculate big state-space matrices
      #                  BB                AA
      #  x1    /       B    0  ... 0 \    /   A \
      #  x2    |      AB    B  ... 0 |    |  A2 |
      #  x3  = |    A^2B   AB  ... 0 |u + | ... |x0 + gg
      #  ...   |     ...  ...  ... 0 |    | ... |
      #  xN    \A^(n-1)B  ...  ... B /    \ A^N /
      #  X = BB * U + AA * x0 + gg
			A_batch = A.unsqueeze(0).repeat(n_batch, 1, 1)
			if i==0:
				BB[:, :n, :m] = B
				AA[:, :n, :n] = A
				gg[:, :n] = g
			else:
				BB[:, i*n:(i+1)*n, :m*N] = torch.bmm(A_batch, BB[:, (i-1)*n:i*n, 0:m*N])
				BB[:, i*n:(i+1)*n, i*m:(i+1)*m] = B
				AA[:, i*n:(i+1)*n, :n] = torch.bmm(A_batch, AA[:, (i-1)*n:i*n, :n])
				gg[:, i*n:(i+1)*n, :1] = torch.bmm(A_batch, gg[:, (i-1)*n:i*n, :1]) + g
			if i%2 != 0:
				j = i//2
				if j == N//2-1:
					qx[:, i*n:i*n+2, 0] = -tracking_traj[:, j] * self.rhoN
				else:
					qx[:, i*n:i*n+2, 0] = -tracking_traj[:, j]
			xi = self.kinematics_forward(xi, u0) # constant u for next iteration
		# end for

		'''
			QP constraints Gz <= H, Az = b
		'''
		#### 1. kinamics 
		#### l <= Ax <= u
		Cx_ = self.Cx_.unsqueeze(0).repeat(n_batch,1,1).to(device)
		lx_ = self.lx_.unsqueeze(0).repeat(n_batch,1,1).to(device)
		ux_ = self.ux_.unsqueeze(0).repeat(n_batch,1,1).to(device)
		x0 = x0.unsqueeze(0).repeat(n_batch,1,1)
		C_cor = self.C_cor_.unsqueeze(0).repeat(n_batch,1,1).to(device)
		d_cor = self.d_cor_.unsqueeze(0).repeat(n_batch,1).unsqueeze(2).to(device)
		Cx = torch.bmm(Cx_, BB)
		lx = lx_ - torch.bmm(torch.bmm(Cx_, AA), x0) - torch.bmm(Cx_, gg)
		ux = ux_ - torch.bmm(torch.bmm(Cx_, AA), x0) - torch.bmm(Cx_, gg)
		A = torch.zeros(n_batch, n_ineq * N, m * N).to(device)
		l = torch.zeros(n_batch, n_ineq * N, 1).to(device)
		u = torch.zeros(n_batch, n_ineq * N, 1).to(device)
		A[:, :N] = Cx
		A[:, N:n_ineq*N] = self.Cu_
		l[:, :N] = lx
		l[:, N:n_ineq*N] = self.lu_
		u[:, :N] = ux
		u[:, N:n_ineq*N] = self.uu_
		# Gx <= h
		G_u = torch.zeros(n_batch, 2 * n_ineq * N , nz).to(device)  # nz = m*N + ns
		H_u = torch.zeros(n_batch, 2 * n_ineq * N, 1).to(device)
		G_u[:, :n_ineq*N, :m*N] = A
		G_u[:, n_ineq*N:, :m*N] = -A
		H_u[:, :n_ineq*N] = u
		H_u[:, n_ineq*N:] = -l
		H_u = H_u.squeeze(2)
		#### 2. corridor constraints
		#### A_Cor * [ C_cor * (BB * U + AA * x0 + gg) + d_cor ] < b_cor + s
		G_cor_u = torch.bmm(torch.bmm(A_cor, C_cor), BB)
		l_cor = - torch.bmm(torch.bmm(C_cor, AA), x0) - torch.bmm(C_cor, gg) - d_cor
		H_cor = torch.bmm(A_cor, l_cor).squeeze(2) + b_cor
		G_cor_s = - I_cor
		G_cor = torch.cat([G_cor_u, G_cor_s], dim=-1)
		#### 3. slack variable
		#### s >0
		G_s = torch.zeros(n_batch, ns, nz).to(device)
		G_s[:, :, -ns:] = -I_cor
		H_s = torch.zeros(n_batch, ns).to(device)

		G = torch.cat([G_u, G_cor, G_s], dim=-2)
		H = torch.cat([H_u, H_cor, H_s], dim=-1)


		# Az = b, no equality constraints
		n_eq = 0
		A = torch.zeros(n_batch, n_eq, N*m).to(device)
		b = torch.zeros(n_batch, n_eq).to(device)
		# x^t Q x + p^t x
		'''
		# objective function
		# u^T R u + (BB u + AA x0 + gg - rho x_ref) Q (...) + rhos s^2
		
		'''
		Ru = weight_u * self.Ru_.unsqueeze(0).repeat(n_batch, 1, 1).to(device)
		Qx_ = self.Qx_.unsqueeze(0).repeat(n_batch, 1, 1).to(device)
		BB_T = torch.transpose(BB, 1, 2)
		Qu = torch.bmm(torch.bmm(BB_T, Qx_), BB) + Ru # try to make Q Semidefinite by adding Ru
		p1 = torch.bmm(AA, x0) + gg
		pu = torch.bmm(torch.bmm(BB_T, Qx_), p1) + torch.bmm(BB_T, qx) 
		pu = pu.squeeze(2)

		Q = weight_s * self.Q_.unsqueeze(0).repeat(n_batch, 1, 1).to(device)
		Q[:, :m*N, :m*N] = Qu

		Q_sym = 0.5 * (Q + Q.transpose(-1, -2))
		p = self.p_.unsqueeze(0).repeat(n_batch, 1).to(device)
		p[:, :m*N] = pu
		sol_u = QPFunction(verbose=-1,solver=QPSolvers.CVXPY, check_Q_spd=False)(
			Q_sym, p, G, H, A, b 
		)


		# Check Value
		if train:
			if torch.isnan(sol_u).any():
				raise ValueError('Infinite values found in gradient.')
			
		# u = sol_u.unsqueeze(2)
		# u_T = u.transpose(-1, -2)
		# cost = torch.bmm(torch.bmm(u_T, Q), u) + torch.bmm(u_T, p.unsqueeze(2))
		
		# sol_u [n_batch, nz]
		sol_u = sol_u[:, :m*N]
		sol_u = sol_u.unsqueeze(2)
		sol_x = torch.bmm(BB, sol_u) + torch.bmm(AA, x0) + gg
		sol_x = sol_x.reshape(n_batch, N, n)
		zeros = torch.zeros(n_batch, 1, 2).to(device)
		sol_x = sol_x[:, 1::2, :2]
		sol_x = self.ego2lidar_traj(sol_x)
		if self.plot:
			with torch.no_grad():
				# propagate u check
				u0 = u0.repeat(N, 1)
				linear_x = torch.bmm(BB, sol_u) + torch.bmm(AA, x0) + gg
				linear_x = linear_x.reshape(n_batch, N, n)
				zeros = torch.zeros(n_batch, 1, 2).to(device)
				# linear_x = linear_x[:, 1::2, :2]
				linear_x = linear_x[:, :, :2]
				linear_x = self.ego2lidar_traj(linear_x)

				self.debug(tracking_traj_debug, linear_x, corridor_debug)
		sol_x = torch.cat((zeros, sol_x), dim=1)
		sol_x = sol_x[:, 1:, :] - sol_x[:, :-1, :]  # step format x,y
		return sol_x

	def covertCorridors(self, rect):
		"""
		Convert rectangle geometry to hPoly (hyperplane) representation.

		Args:
			rect (torch.Tensor): A tensor of shape (n_batch, T, 5), where the last dimension contains
								[center_x, center_y, yaw, length, width].
								
		Returns:
			torch.Tensor: A tensor of shape (n_batch, T, 4, 2), representing 4 hyperplanes for each rectangle,
						where each hyperplane is defined by a normal vector and a point.
						The last dimension contains [normal_x, normal_y, point_x, point_y].
		"""
		rect = self.lidar2ego_rect(rect)

		# Unpack the geometry properties
		center_x = rect[..., 0]
		center_y = rect[..., 1]
		yaw = rect[..., 2]
		length = rect[..., 3]
		width = rect[..., 4]

		# Half length and width for corner computation
		half_length = length / 2.0
		half_width = width / 2.0

		# Calculate the four corners of the rectangle
		cos_yaw = torch.cos(yaw)
		sin_yaw = torch.sin(yaw)

		# Top-right corner
		corner1_x = center_x + cos_yaw * half_length - sin_yaw * half_width
		corner1_y = center_y + sin_yaw * half_length + cos_yaw * half_width

		# Top-left corner
		corner2_x = center_x - cos_yaw * half_length - sin_yaw * half_width
		corner2_y = center_y - sin_yaw * half_length + cos_yaw * half_width

		# Bottom-left corner
		corner3_x = center_x - cos_yaw * half_length + sin_yaw * half_width
		corner3_y = center_y - sin_yaw * half_length - cos_yaw * half_width

		# Bottom-right corner
		corner4_x = center_x + cos_yaw * half_length + sin_yaw * half_width
		corner4_y = center_y + sin_yaw * half_length - cos_yaw * half_width

		# Compute normal vectors for each side
		# 1st hyperplane (between corner1 and corner2)
		normal1_x = -sin_yaw
		normal1_y = cos_yaw
		point1_x = corner1_x 
		point1_y = corner1_y 

		# 2nd hyperplane (between corner2 and corner3)
		normal2_x = -cos_yaw
		normal2_y = -sin_yaw
		point2_x = corner2_x
		point2_y = corner2_y

		# 3rd hyperplane (between corner3 and corner4)
		normal3_x = sin_yaw
		normal3_y = -cos_yaw
		point3_x = corner3_x
		point3_y = corner3_y

		# 4th hyperplane (between corner4 and corner1)
		normal4_x = cos_yaw
		normal4_y = sin_yaw
		point4_x = corner4_x
		point4_y = corner4_y

		# Stack the normal vectors and points together
		A = torch.stack([
			torch.stack([normal1_x, normal1_y], dim=-1),
			torch.stack([normal2_x, normal2_y], dim=-1),
			torch.stack([normal3_x, normal3_y], dim=-1),
			torch.stack([normal4_x, normal4_y], dim=-1)
		], dim=-2)
		p = torch.stack([
			torch.stack([point1_x, point1_y], dim=-1),
			torch.stack([point2_x, point2_y], dim=-1),
			torch.stack([point3_x, point3_y], dim=-1),
			torch.stack([point4_x, point4_y], dim=-1)
		], dim=-2)
		b = torch.sum(A * p, dim=-1)
		
		# A: [B, 4N, 2]
		# b: [B, N, 4]
		return A, b
	

	def debug(self, tracking_traj, opt_traj, corridors):
		n_batch = tracking_traj.shape[0]
		fig, axs = plt.subplots(1, n_batch, figsize=(4 * n_batch, 4))  # Set up a row of subplots
		for i in range(n_batch):
			ax = axs[i]  # Access each subplot

			ax.set_xlim(-15.0, 15.0)
			ax.set_ylim(-5.0, 35.0)

			# Plot the predicted trajectory
			pred_traj = tracking_traj[i].cpu().numpy()
			x_coords = pred_traj[:, 0]
			y_coords = pred_traj[:, 1]
			ax.scatter(x_coords, y_coords, color='red', marker='x', alpha=0.8, zorder=-1)

			# Plot the optimized trajectory
			rollout_states = opt_traj[i].cpu().numpy()
			color_opt = '#4B5CC4'
			x_coords = rollout_states[:, 0]
			y_coords = rollout_states[:, 1]
			ax.plot(x_coords, y_coords, color=color_opt, linewidth=1.5, alpha=0.8, zorder=-1)
			ax.scatter(x_coords, y_coords, color=color_opt, s=3.0, alpha=0.5, zorder=-1)

			# Plot the corridor boundaries
			corridors_plot = corridors[i].cpu().numpy()
			n_corridor = len(corridors_plot)
			for j, gt_corridor in enumerate(corridors_plot):
				cmap_corridor = 'jet'
				a = 1 - j / n_corridor
				color_corridor = np.array(plt.cm.get_cmap(cmap_corridor)(a))[:3]

				seed_p = gt_corridor[:2]
				yaw = gt_corridor[2]
				s = gt_corridor[3]
				l = gt_corridor[4]

				# Define rectangle vertices in local coordinates
				rect_local = np.array([
					[s / 2, l / 2],
					[s / 2, -l / 2],
					[-s / 2, -l / 2],
					[-s / 2, l / 2],
					[s / 2, l / 2]  # Closing the rectangle
				])

				# Rotate and transform the rectangle to global coordinates
				R = np.array([[math.cos(yaw), -math.sin(yaw)],
							[math.sin(yaw), math.cos(yaw)]])
				rect_global = (R @ rect_local.T).T + seed_p
				# ax.plot(rect_global[:, 0], rect_global[:, 1], color=color_corridor, linewidth=1, alpha=0.8, zorder=-1)

			ax.grid(True)

		# Save the entire figure with subplots
		fig.savefig('vis/traj_opt_debug.png', dpi=500)
		plt.close(fig)

