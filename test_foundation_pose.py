from isaacgym import gymapi
from isaacgym.torch_utils import *

import trimesh
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import matrix_to_euler_angles

from simulate.simulate_isaac import environment, Mover
from util.read_urdf import get_urdf_info
from util.other_isaacgym_fuction import ( 
    pq_to_H,
    euler_xyz_to_matrix
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
    {"name": "--random_seed", "type": int, "default": 100, "help": "Numpy random seed"},
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
usb_pose.p = gymapi.Vec3(table_pose.p.x + 0.15, table_pose.p.y + 0.1, table_dims.z + 0.5 * usb_aabb[2] + 0.2)
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
                   random_pos_range=[[-0.065, 0.065], [-0.25, -0.15], [0, 0]], random_rot=[[0,0,1], [-0.8, -1.3]])

socket_USB_pose = gymapi.Transform()
socket_USB_pose.p.x = usb_pose.p.x
socket_USB_pose.p.y = usb_pose.p.y
socket_USB_pose.p.z = table_dims.z + socket_aabb[2]
Env.set_mesh_asset(mesh_file='USB_Male_place.urdf' , fix_base=False, disable_gravity=False, name='socket_USB', 
                   pos = socket_USB_pose.p, rot=usb_pose.r, collision=2, color=gymapi.Vec3(0, 0, 1))

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

mesh_file = "/home/hcis/YongZhe/obj-and-urdf/UsbStand_Medium.obj"
mesh = trimesh.load(mesh_file).apply_scale(0.001)

def calculate_pose_distance(T1, T2):
    # Step 1: Extract the translation vectors from the two 4x4 pose matrices
    t1 = T1[:3, 3]  # Translation vector of T1
    t2 = T2[:3, 3]  # Translation vector of T2

    # Step 2: Extract the rotation matrices from the two 4x4 pose matrices
    R1 = T1[:3, :3]  # Rotation matrix of T1
    R2 = T2[:3, :3]  # Rotation matrix of T2

    # Step 3: Compute Euclidean distance between the translation vectors
    position_distance_in_xyz = t1 - t2
    # position_distance = np.linalg.norm(t1 - t2)

    # Step 4: Compute angular distance between the rotation matrices
    # Convert the rotation matrices to quaternion form
    r1 = R.from_matrix(R1)
    r2 = R.from_matrix(R2)

    # Calculate the relative rotation from T1 to T2
    relative_rotation = r1.inv() * r2

    # Calculate the angular distance in degrees (can also be radians by using .magnitude())
    angular_distance = relative_rotation.magnitude() * (180 / np.pi)  # convert to degrees
    euler_error_in_xyz = relative_rotation.as_euler("XYZ", degrees=True)

    return position_distance_in_xyz, euler_error_in_xyz

rb_states, dof_pos, _ = mover.env.step()
average_diff_xyz = []  # translation error
average_diff_euler = []  # rotation error
adjust_diff_xyz = []
adjust_diff_euler = []
for env in range(mover.env.num_envs):
    pred_pose = mover.get_object_predicted_pose_from_camera_id(mesh=mesh, object_name="socket", camera_id=1, env_idx=env, debug=4)
    ground_truth = pq_to_H(p=rb_states[mover.env.obj_idxs['socket'][env], :3], q=rb_states[mover.env.obj_idxs['socket'][env], 3:7]) # ground truth 
    ground_truth = ground_truth.cpu().numpy()

    diff_pos, diff_rot = calculate_pose_distance(pred_pose, ground_truth)

    # Rule-based adjustment
    pred_pose = torch.tensor(pred_pose)
    rot = matrix_to_euler_angles(pred_pose[:3, :3], convention='XYZ')
    # print(rot[2])
    if not (-np.pi/2 < rot[2] < np.pi/2):
        # print("adjust")
        pred_pose[:3, :3] = pred_pose[:3, :3] @ euler_xyz_to_matrix(0, 0, np.pi)[:3, :3]
    
    adjust_diff_pos, adjust_diff_rot = calculate_pose_distance(pred_pose.cpu().numpy(), ground_truth)

    average_diff_xyz.append(np.abs(diff_pos))
    average_diff_euler.append(np.abs(diff_rot))
    
    adjust_diff_xyz.append(np.abs(adjust_diff_pos))
    adjust_diff_euler.append(np.abs(adjust_diff_rot))


averate_diff_xyz = np.abs(np.array(average_diff_xyz))
print("Translation error in x-axis", np.mean(averate_diff_xyz[:, 0]))
print("Translation error in y-axis", np.mean(averate_diff_xyz[:, 1]))
print("Translation error in z-axis", np.mean(averate_diff_xyz[:, 2]))

plt.figure(figsize=(6.4*3, 4.8))
plt.subplot(1, 3, 1)
plt.hist(averate_diff_xyz[:, 0])
plt.xlim(0, 0.01)
plt.ylim(0, 50)
plt.xlabel('Translation error (x-axis)', fontsize=12)

plt.subplot(1, 3, 2)
plt.hist(averate_diff_xyz[:, 1])
plt.xlim(0, 0.01)
plt.ylim(0, 50)
plt.xlabel('Translation error (y-axis)', fontsize=12)

plt.subplot(1, 3, 3)
plt.hist(averate_diff_xyz[:, 2])
plt.xlim(0, 0.01)
plt.ylim(0, 50)
plt.xlabel('Translation error (z-axis)', fontsize=12)

plt.tight_layout()
plt.savefig(f'./average_diff_tran.png')
plt.clf()

average_diff_euler = np.array(average_diff_euler)
print("Average Rotation error in x-axis", np.mean(average_diff_euler[:, 0]))
print("Average Rotation error in y-axis", np.mean(average_diff_euler[:, 1]))
print("Average Rotation error in z-axis", np.mean(average_diff_euler[:, 2]))

plt.figure(figsize=(6.4*3, 4.8))
plt.subplot(1, 3, 1)
plt.hist(average_diff_euler[:, 0], bins=np.arange(0, 190, 10))
plt.xlim(0, 180)
plt.xlabel('Rotation angle error (x-axis)', fontsize=12)

plt.subplot(1, 3, 2)
plt.hist(average_diff_euler[:, 1], bins=np.arange(0, 190, 10))
plt.xlim(0, 180)
plt.xlabel('Rotation angle error (y-axis)', fontsize=12)

plt.subplot(1, 3, 3)
plt.hist(average_diff_euler[:, 2], bins=np.arange(0, 190, 10))
plt.xlim(0, 180)
plt.xlabel('Rotation angle error (z-axis)', fontsize=12)

plt.tight_layout()
plt.savefig(f'./average_diff_rot.png')
plt.clf()

adjust_diff_euler = np.array(adjust_diff_euler)
print("Adjust Rotation error in x-axis", np.mean(adjust_diff_euler[:, 0]))
print("Adjust Rotation error in y-axis", np.mean(adjust_diff_euler[:, 1]))
print("Adjust Rotation error in z-axis", np.mean(adjust_diff_euler[:, 2]))

plt.figure(figsize=(6.4*3, 4.8))
plt.subplot(1, 3, 1)
plt.hist(adjust_diff_euler[:, 0], bins=np.arange(0, 190, 10))
plt.xlim(0, 180)
plt.xlabel('Rotation angle error (x-axis)', fontsize=12)

plt.subplot(1, 3, 2)
plt.hist(adjust_diff_euler[:, 1], bins=np.arange(0, 190, 10))
plt.xlim(0, 180)
plt.xlabel('Rotation angle error (y-axis)', fontsize=12)

plt.subplot(1, 3, 3)
plt.hist(adjust_diff_euler[:, 2], bins=np.arange(0, 190, 10))
plt.xlim(0, 180)
plt.xlabel('Rotation angle error (z-axis)', fontsize=12)

plt.tight_layout()
plt.savefig(f'./adjust_diff_rot.png')
plt.clf()

while not Env.gym.query_viewer_has_closed(Env.viewer):
    Env.step()

Env.kill()