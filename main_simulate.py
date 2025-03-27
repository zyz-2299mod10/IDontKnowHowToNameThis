from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

import math
import trimesh
import cv2

from simulate.simulate_isaac import environment, Mover
from util.read_urdf import get_urdf_info
from util.other_isaacgym_fuction import ( 
    quat_mul_NotForTensor,
    H_2_Transform,
    pq_to_H
    )


''' 
Basic setting:
1. Custom parameters and asset/urdf root path
2. Initiallize robot and table configration
'''
# Add custom arguments
custom_parameters = [ 
    {"name": "--device", "type": str, "default": "cuda", "help": "[cuda, cpu]"},
    {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
    {"name": "--random_seed", "type": int, "default": 10, "help": "Numpy random seed"},
    {"name": "--object", "type": str, "default": "USB"}
]

asset_root = "/home/hcis/isaacgym/assets"
urdf_root = "/home/hcis/YongZhe/obj-and-urdf/urdf"

table_dims = gymapi.Vec3(0.8, 0.8, 0.4)
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)
franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(table_pose.p.x - 0.65 * table_dims.x, table_pose.p.y, table_dims.z)

'''
Set environment
'''
Env = environment(custom_parameters=custom_parameters, robot_root=asset_root, urdf_root=urdf_root, 
                  robot_type='franka', robot_pose=franka_pose)
# table
Env.set_box_asset(dims=table_dims, fix_base=True, disable_gravity=True, pos=table_pose.p, collision=0, name='table')

# usb
usb_urdf_info = get_urdf_info(urdf_root, "USB_Male.urdf")
usb_aabb = usb_urdf_info.get_mesh_aabb_size()
usb_pose = gymapi.Transform()
usb_pose.p = gymapi.Vec3(table_pose.p.x + 0.15, table_pose.p.y + 0.1, table_dims.z + 0.5 * usb_aabb[2])
Env.set_mesh_asset(mesh_file="USB_Male.urdf" , fix_base=False, disable_gravity=False, name='usb', 
                   pos = usb_pose.p, collision=1, color=gymapi.Vec3(1, 0, 0), semantic_id=400)

# socket
socket_urdf_info = get_urdf_info(urdf_root, 'USB_Male_place.urdf')
socket_aabb = socket_urdf_info.get_mesh_aabb_size()
socket_pose = gymapi.Transform()
socket_pose.p.x = usb_pose.p.x
socket_pose.p.y = usb_pose.p.y
socket_pose.p.z = table_dims.z + socket_aabb[2]
Env.set_mesh_asset(mesh_file='USB_Male_place.urdf' , fix_base=False, disable_gravity=False, name='socket', 
                   pos = socket_pose.p, collision=2, color=gymapi.Vec3(0, 1, 0), semantic_id=600,
                   random_pos_range=[[-0.065, 0.065], [-0.15, -0.1], [0, 0]], random_rot=[[[0,0,1], [-0.8, -1.3]]])

# socket_USB_pose = gymapi.Transform()
# socket_USB_pose.p.x = usb_pose.p.x
# socket_USB_pose.p.y = usb_pose.p.y
# socket_USB_pose.p.z = table_dims.z + socket_aabb[2]
# Env.set_mesh_asset(mesh_file='USB_Male_place.urdf' , fix_base=False, disable_gravity=False, name='socket_USB', 
#                    pos = socket_USB_pose.p, rot=usb_pose.r, collision=2, color=gymapi.Vec3(0, 0, 1))

# # obstacle 1
# ob1_dim = gymapi.Vec3(0.02, 0.015, 0.04)
# ob1_pose = gymapi.Transform()
# ob1_pose.p.x = socket_pose.p.x
# ob1_pose.p.y = socket_pose.p.y - 0.09
# ob1_pose.p.z = table_dims.z + ob1_dim.z * 0.5 + 0.15
# ob_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
# Env.set_box_asset(dims=ob1_dim, fix_base=True, disable_gravity=True, pos=ob1_pose.p, collision=0, name='ob1', color=ob_color)

# # obstacle 2
# ob2_dim = gymapi.Vec3(0.02, 0.02, 0.04)
# ob2_pose = gymapi.Transform()
# ob2_pose.p.x = socket_pose.p.x
# ob2_pose.p.y = socket_pose.p.y - 0.09
# ob2_pose.p.z = table_dims.z + ob1_dim.z * 0.5
# ob_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
# Env.set_box_asset(dims=ob1_dim, fix_base=True, disable_gravity=True, pos=ob2_pose.p, collision=0, name='ob2',
#                   random_pos_range=[[-0.05, 0.05], [0, 0], [0.05, 0.3]], color=ob_color)

# # obstacle 3
# ob3_dim = gymapi.Vec3(0.015, 0.025, 0.04)
# ob3_pose = gymapi.Transform()
# ob3_pose.p.x = socket_pose.p.x
# ob3_pose.p.y = socket_pose.p.y - 0.09
# ob3_pose.p.z = table_dims.z + ob1_dim.z * 0.5 + np.random.uniform(0.05, 0.13)
# ob_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
# Env.set_box_asset(dims=ob1_dim, fix_base=True, disable_gravity=True, pos=ob3_pose.p, collision=0, name='ob3',
#                   random_pos_range=[[-0.05, 0.08], [0, 0], [0.05, 0.13]], color=ob_color)

# set camera
p = gymapi.Vec3(0.1, 0, 0.05)
look_down = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(-90))
Env.set_camera(cam_pos=p, cam_rot=look_down, mode="hand", id = 0)

p = gymapi.Vec3(0.9,  0.3,  0.7)
cam_target = gymapi.Vec3(0, -0.4, 0.4)
Env.set_camera(cam_pos=p, cam_look_target=cam_target, mode="side", id = 1)

# create environment(s)
Env.create_env()

for _ in range(50):
    Env.step()

'''
Execute action
'''
mover = Mover(Env)

# mesh_file = "/home/hcis/YongZhe/obj-and-urdf/UsbStand.stl"
# mesh = trimesh.load(mesh_file).apply_scale(0.001)
# pred_pose = mover.get_object_predicted_pose_from_camera_id(mesh=mesh, object_name="socket", camera_id=1 , debug=4)
# print(pred_pose)
# pred_pose = torch.tensor(pred_pose).to(mover.env.device)

# rb_states, dof_pos, _ = mover.env.step()
# print(pq_to_H(p=rb_states[mover.env.obj_idxs['socket'][0], :3], q=rb_states[mover.env.obj_idxs['socket'][0], 3:7])) # ground truth

grasp_pos, grasp_rot = mover.get_grasping_pose('usb')

mover.IK_move(goal_pos=grasp_pos, goal_rot=grasp_rot, closed=False, T=500)
mover.IK_move(goal_pos=grasp_pos, goal_rot=grasp_rot, closed=True, T=200)
mover.IK_move(goal_pos=grasp_pos + torch.tensor([0, 0, 0.075], device=mover.env.device), goal_rot=grasp_rot, closed=True, T=500)

hp, hr = mover.get_predicted_hole_pose(camera_id = 0, env_idx = 0, visualize = True)
mover.IK_move(goal_pos=torch.tensor([hp.x, hp.y, hp.z + 0.25], device=mover.env.device),
              goal_rot=torch.tensor(grasp_rot, device=mover.env.device),
              closed=True,
              T=500)
depth, rgb, segmentation, intrinsic, view_matrix, t = mover.env.get_camera_img(id = 0, env_idx = 0, store = True)

hp_world = np.array([hp.x, hp.y, hp.z, 1.0])
hp_cam = view_matrix @ hp_world
uv = intrinsic.cpu().numpy() @ hp_cam[:3]
u, v = int(uv[0]), int(uv[1])
rgb = np.ascontiguousarray(rgb)
crop_rgb = rgb[v - 20: v + 20,
               u - 20: u + 20, ...]
cv2.imwrite('CropRGB.png', crop_rgb)


# set curobo goal pose
# hand_goal_position = []
# hand_goal_quaternion = []
# rb_states, dof_pos, _ = mover.env.step()
# for env_idx in range(mover.env.num_envs):
#     u = rb_states[mover.env.obj_idxs['usb'][env_idx], :3]
#     ur = rb_states[mover.env.obj_idxs['usb'][env_idx], 3:7]
#     e = rb_states[mover.env.obj_idxs['socket'][env_idx], :3] 
#     er = rb_states[mover.env.obj_idxs['socket'][env_idx], 3:7] 
#     h = rb_states[mover.env.obj_idxs['hand'][env_idx], :3]
#     hr = rb_states[mover.env.obj_idxs['hand'][env_idx], 3:7]
                
#     e[-1] = e[-1] + 0.075 # above hole
#     end_pose_matrix = pq_to_H(e, er)
#     # pred_pose[2, 3] = pred_pose[2, 3] + 0.075
#     usb_pose_matrix = pq_to_H(u, ur)

#     trans = end_pose_matrix @ torch.inverse(usb_pose_matrix)
#     # trans = pred_pose @ torch.inverse(usb_pose_matrix)

#     hand_pose_matrix = pq_to_H(h, hr)

#     goal_pose_matrix = trans @ hand_pose_matrix
#     goal_pose_pq = H_2_Transform(goal_pose_matrix)

#     hand_goal_position.append([goal_pose_pq.p.x, goal_pose_pq.p.y, goal_pose_pq.p.z])
#     hand_goal_quaternion.append([goal_pose_pq.r.w, goal_pose_pq.r.x, goal_pose_pq.r.y, goal_pose_pq.r.z])

# mover.curobo_move(goal_pos=hand_goal_position, goal_rot=hand_goal_quaternion, closed=True)
# mover.IK_move(goal_pos=torch.tensor(hand_goal_position, device=mover.env.device),
#               goal_rot=torch.tensor([[hand_goal_quaternion[0][1], hand_goal_quaternion[0][2], hand_goal_quaternion[0][3], hand_goal_quaternion[0][0]]], 
#                                      device=mover.env.device),
#               closed=True,
#               T=500) # Avoid curobo planning error

# mover.RRT_move(goal_pos=hand_goal_position, goal_rot=hand_goal_quaternion, closed=True, exclude_obj_from_environment=["usb"], draw=True)

# put USB down
# mover.IK_move(goal_pos=torch.tensor(hand_goal_position, device=mover.env.device) + torch.tensor([0, 0, -0.05], device=mover.env.device),
#               goal_rot=torch.tensor([[hand_goal_quaternion[0][1], hand_goal_quaternion[0][2], hand_goal_quaternion[0][3], hand_goal_quaternion[0][0]]], 
#                                      device=mover.env.device),
#               closed=True,
#               T=200)
# mover.IK_move(goal_pos=torch.tensor(hand_goal_position, device=mover.env.device) + torch.tensor([0, 0, -0.05], device=mover.env.device),
#               goal_rot=torch.tensor([[hand_goal_quaternion[0][1], hand_goal_quaternion[0][2], hand_goal_quaternion[0][3], hand_goal_quaternion[0][0]]], 
#                                      device=mover.env.device),
#               closed=False,
#               T=200)
# mover.IK_move(goal_pos=torch.tensor(hand_goal_position, device=mover.env.device) + torch.tensor([0, 0, 0.005], device=mover.env.device),
#               goal_rot=torch.tensor([[hand_goal_quaternion[0][1], hand_goal_quaternion[0][2], hand_goal_quaternion[0][3], hand_goal_quaternion[0][0]]], 
#                                      device=mover.env.device),
#               closed=False,
#               T=200)

# mover.return_initial_pose()

while not Env.gym.query_viewer_has_closed(Env.viewer):
    Env.step()

Env.kill()