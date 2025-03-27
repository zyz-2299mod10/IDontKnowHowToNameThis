from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import yaml
import pickle
import os
from scipy.spatial.transform import Rotation as R
import cv2
from typing import List
import numpy as np
import copy
import time

from planning.util import same_traj_dim
from planning import RRT
from planning.motion_gener import cuRobo_planning
from util.read_urdf import get_urdf_info
from util.other_isaacgym_fuction import (
    orientation_error,
    H_2_Transform,
    euler_xyz_to_matrix,
    Set_box_oob,
    quat_mul_NotForTensor,
    pq_to_H
    )
from util.camera import compute_camera_intrinsics_matrix
from hole_estimation.predict_hole_pose import CoarseMover
from FoundationPose.estimater import *

class environment:
    def __init__(self, custom_parameters, robot_root, robot_type, urdf_root, robot_pose) -> None: 
        '''
        Usage:
            Set all environments (self.create_env) after setting all objects (self.set_box/mesh_asset)
            Calling-out each object by the "name" you setted (robot hand called "hand")
        input:
            robot pose: gymapi.Transform
        '''
        self.robot_root = robot_root
        self.urdf_root = urdf_root
        self.robot_pose = robot_pose
        self.robot_type = robot_type
        
        self.gym = gymapi.acquire_gym()       

        self.args = gymutil.parse_arguments(
            custom_parameters=custom_parameters,
        )        
        np.random.seed(self.args.random_seed)

        self.num_envs = self.args.num_envs
        
        device = self.args.device
        if device == "cuda":
            self.device = self.args.sim_device if self.args.use_gpu_pipeline else 'cpu'
        else: self.device = 'cpu'
        
        self.prepare_sim()
        self.set_robot_asset()
        
        # prepare asset
        self.assets = {}
        self.camera = {}
    
    def prepare_sim(self):
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.contact_collection = gymapi.ContactCollection(1)
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = self.args.num_threads
            sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")
        
        # create sim
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
    
    def create_viewer(self):
        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")       
        
        # point camera at middle env
        num_per_row = int(math.sqrt(self.num_envs))
        cam_pos = gymapi.Vec3(1,  0.4, 1)
        cam_target = gymapi.Vec3(0, -0.4, 0)
        middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)
        
    def set_mesh_asset(self, 
                       mesh_file:str, 
                       fix_base:bool, 
                       disable_gravity:bool,
                       pos:gymapi.Vec3, 
                       collision:int, 
                       name:str, 
                       rot:gymapi.Quat = None,
                       color:gymapi.Vec3 = None,
                       random_pos_range:List[List] = None, 
                       random_rot:List[List] = None,
                       semantic_id:int = None,):
        '''
        rot: exact quaternion
        random_pos_range: position random offset. format: [[x], [y], [z]]
        random_rot_range: given rotation axis and rotation range (degress). format: [[axis], [range]]
        '''
        if name == None: 
            raise Exception("Need to name object")

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fix_base
        asset_options.disable_gravity = disable_gravity
        asset_options.use_mesh_materials = True
        _asset = self.gym.load_asset(self.sim, self.urdf_root, mesh_file, asset_options)        
        
        # urdf information
        urdf_info = get_urdf_info(self.urdf_root, mesh_file)
        urdf_aabb = urdf_info.get_mesh_aabb_size()
        urdf_collisionMesh_path = urdf_info.get_collision_pathName_scale()["filename"]
        urdf_scale = urdf_info.get_collision_pathName_scale()["scale"]
        dims = gymapi.Vec3(urdf_aabb[0], urdf_aabb[1], urdf_aabb[2])

        asset_info = {
            'dims': dims,
            "asset": _asset,
            "obj_pos": pos,
            "obj_rot": rot,
            "random_pos_range": random_pos_range,
            "random_rot": random_rot,
            "collision": collision,
            "color": color,
            "urdf_collisionMesh_path": urdf_collisionMesh_path,
            "scale": urdf_scale,
            "semantic_id": semantic_id,
        }
        self.assets[name] = asset_info
    
    def set_box_asset(self, 
                      dims:gymapi.Vec3, 
                      fix_base:bool, 
                      disable_gravity:bool, 
                      pos:gymapi.Vec3, 
                      collision:int, 
                      name:str, 
                      rot:gymapi.Quat = None, 
                      color:gymapi.Vec3 = None, 
                      random_pos_range:List[List] = None, 
                      random_rot:List[List] = None,
                      semantic_id:int = None,):
        '''
        rot: exact quaternion
        random_pos_range: position random offset. format: [[x], [y], [z]]
        random_rot_range: given rotation axis and rotation range (degress). format: [[axis], [range]]
        '''        
        if name == None: 
            raise Exception("Need to name object")
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fix_base
        asset_options.disable_gravity = disable_gravity
        asset_options.use_mesh_materials = True
        _asset = self.gym.create_box(self.sim, dims.x, dims.y, dims.z, asset_options)
        
        asset_info = {
            "dims": dims,
            "asset": _asset,
            "obj_pos": pos,
            "obj_rot": rot,
            "random_pos_range": random_pos_range,
            "random_rot": random_rot,
            "collision": collision,
            "color": color,
            "semantic_id": semantic_id,
        }
        self.assets[name] = asset_info
    
    def set_camera(self,
                   cam_pos:gymapi.Vec3,
                   cam_rot:gymapi.Quat = None,
                   cam_look_target:gymapi.Vec3 = None,
                   mode:str = "hand",
                   id:int = 0):
        '''
        mode: hand (wrist) or side camera
            if mode is hand, ur cam_pos and cam_rot will be the relative transformation to the robot hand
        id: To index the camera (default = 0)
        '''

        # create camera
        camera_props = gymapi.CameraProperties()
        camera_props.width = 300
        camera_props.height = 300
        camera_props.horizontal_fov = 90 # default
        camera_props.near_plane = 0.01
        camera_props.far_plane = 1.5
        camera_props.enable_tensors = True

        cam_trans = gymapi.Transform()
        cam_trans.p = cam_pos
        cam_trans.r = cam_rot if cam_rot is not None else gymapi.Quat(0, 0, 0, 1)
        cam_info = {
            "mode":mode,
            "cam_props": camera_props,
            "cam_target": cam_look_target,
            "pose": cam_trans,
        }
        self.camera[id] = cam_info

    def set_robot_asset(self):
        with open('./simulate/robot_type.yml', 'r') as f:
            robot_file = yaml.safe_load(f)[self.robot_type]

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        self.robot_asset = self.gym.load_asset(self.sim, self.robot_root, robot_file, asset_options)
    
    def get_robot_defaultstate_prop(self):
        '''
        Currently only support franka
        '''
        # configure franka dofs
        franka_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        franka_lower_limits = franka_dof_props["lower"]
        franka_upper_limits = franka_dof_props["upper"]
        franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.4 * (franka_upper_limits + franka_lower_limits)

        # use position drive for all dofs
        franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][:7].fill(50.0)
        franka_dof_props["damping"][:7].fill(40.0)
            
        # grippers
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][7:].fill(100.0)
        franka_dof_props["damping"][7:].fill(40.0)

        # default dof states and position targets
        franka_num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
        default_dof_pos[:7] = franka_mids[:7]
        # grippers open
        default_dof_pos[7:] = franka_upper_limits[7:]

        default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        if self.robot_type == 'franka':
            self.franka_link_dict = self.gym.get_asset_rigid_body_dict(self.robot_asset)
        
        return franka_dof_props, default_dof_state, default_dof_pos
    
    def create_env(self):
        '''
        create environments with all asset and robot
        '''
        # configure env grid
        num_envs = self.num_envs
        num_per_row = int(math.sqrt(num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print("Creating %d environments" % num_envs)
        
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
        self.obj_idxs = {}
        for obj_name in self.assets:
            self.obj_idxs[obj_name] = []
        self.obj_idxs["hand"] = []

        self.envs = []
        self.init_pos_list = []
        self.init_rot_list = []
        franka_dof_props, default_dof_state, default_dof_pos = self.get_robot_defaultstate_prop()

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env) 
            
            # set obj
            for obj_name in self.assets:   
                obj = self.assets[obj_name]
                _pose = gymapi.Transform()

                _pose.p = obj["obj_pos"]
                if obj["random_pos_range"] is not None:
                    random_pos_range = obj["random_pos_range"]
                    _pose.p.x += np.random.uniform(random_pos_range[0][0], random_pos_range[0][1])
                    _pose.p.y += np.random.uniform(random_pos_range[1][0], random_pos_range[1][1])
                    _pose.p.z += np.random.uniform(random_pos_range[2][0], random_pos_range[2][1])

                if obj["obj_rot"] is not None:
                    _pose.r = obj["obj_rot"]
                
                elif obj["random_rot"] is not None:
                    random_rot = obj["random_rot"]

                    rot = gymapi.Quat(0, 0, 0, 1)
                    for rotation in random_rot:
                        rot_current = gymapi.Quat.from_axis_angle(gymapi.Vec3(rotation[0][0],
                                                                              rotation[0][1],
                                                                              rotation[0][2]),
                                                                              np.random.uniform(rotation[1][0], rotation[1][1]) * math.pi)
                        rot = quat_mul_NotForTensor(rot, rot_current)
                    
                    _pose.r = rot

                if obj["semantic_id"] is not None:
                    _handle = self.gym.create_actor(env, obj["asset"], _pose, obj_name, i, obj["collision"], segmentationId = obj["semantic_id"])
                else:
                    _handle = self.gym.create_actor(env, obj["asset"], _pose, obj_name, i, obj["collision"])

                if obj["color"] is not None:
                    self.gym.set_rigid_body_color(env, _handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, obj["color"])                
                
                _idx = self.gym.get_actor_rigid_body_index(env, _handle, 0, gymapi.DOMAIN_SIM)
                self.obj_idxs[obj_name].append(_idx)
            
            # add franka
            franka_handle = self.gym.create_actor(env, self.robot_asset, self.robot_pose, "franka", i, 2)
                    
            # set dof properties
            self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

            # get inital hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
            hand_pose = self.gym.get_rigid_transform(env, hand_handle)
            self.init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
            self.obj_idxs["hand"].append(hand_idx)

            for id in self.camera:
                cam_handle = self.gym.create_camera_sensor(env, self.camera[id]["cam_props"])
                
                if self.camera[id]["mode"] == "hand":
                    self.gym.attach_camera_to_body(cam_handle, env, hand_handle, self.camera[id]["pose"], gymapi.FOLLOW_TRANSFORM)
                elif self.camera[id]["cam_target"] is not None: # side camera
                    self.gym.set_camera_location(cam_handle, env, self.camera[id]["pose"].p, self.camera[id]["cam_target"])
                else:
                    self.gym.set_camera_transform(cam_handle, env, self.camera[id]["pose"])

                self.camera[id]["handle"] = cam_handle            

        # create viewer
        self.create_viewer()

        # prepare sim
        self.gym.prepare_sim(self.sim)
    
    def get_camera_img(self,
                       id:int,
                       env_idx:int = 0,
                       store:bool = False):
        '''
        get the camera image (view matrix and camera transform also) from specific camera (id and env)
        if u don't give the env index, it will return first env camera image

        return:
            depth: nparray
            rgb: nparray
            segmentation: nparray
            intrinsic: tensor
            view_matrix: nparray 4*4
            camera_transform: gymapi.Transform
        '''
        if id not in self.camera.keys():
            raise Exception(f"can't find the camera id: [{id}]")

        self.gym.start_access_image_tensors(self.sim)
        # get depth image
        color_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_idx], self.camera[id]["handle"], gymapi.IMAGE_COLOR)
        rgb = gymtorch.wrap_tensor(color_tensor).cpu().numpy()[..., 0:3]
        depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_idx], self.camera[id]["handle"], gymapi.IMAGE_DEPTH)
        depth = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
        segmentation_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_idx], self.camera[id]["handle"], gymapi.IMAGE_SEGMENTATION)
        segmentation = gymtorch.wrap_tensor(segmentation_tensor).cpu().numpy()

        # for hole pose estimate
        height, width, _ = rgb.shape
        intrinsic = compute_camera_intrinsics_matrix(height, width, 90)
        camera_transform = self.gym.get_camera_transform(self.sim, self.envs[env_idx], self.camera[id]["handle"])
        view_matrix = self.gym.get_camera_view_matrix(self.sim, self.envs[env_idx], self.camera[id]["handle"])

        if store:
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            store_depth = depth_normalized.astype(np.uint8)
            
            cv2.imwrite(f"./{self.camera[id]['mode']}_{id}_rgb.png", rgb)
            cv2.imwrite(f"./{self.camera[id]['mode']}_{id}_depth.png", store_depth)
            cv2.imwrite(f"./{self.camera[id]['mode']}_{id}_segmentation.png", segmentation)
        
        self.gym.end_access_image_tensors(self.sim)
        return depth, rgb, segmentation, intrinsic, view_matrix, camera_transform
    
    def get_specific_object_segmentation(self,
                                         id:int,
                                         object:str,
                                         env_idx:int = 0,
                                         store:bool = False)-> np.array:
        '''
        get the semantic segmentation image from specific camera (id and env) and specific object (use the name u set)
        if u don't give the env index, it will return first env camera image
        '''
        if self.assets[object]["semantic_id"] is None:
            raise AssertionError(f"You don't set the semantic id of {object}, U should set it in set_box/mesh_asset")


        segmentation_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_idx], self.camera[id]["handle"], gymapi.IMAGE_SEGMENTATION)
        segmentation = gymtorch.wrap_tensor(segmentation_tensor).cpu().numpy()        
        segmentation[(segmentation != self.assets[object]["semantic_id"])] = 0

        if store:
            cv2.imwrite(f"./{self.camera[id]['mode']}_{id}_{object}_segmentation.png", segmentation)

        return segmentation
    
    def get_state(self):
        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # Rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states).to(self.device)

        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states).to(self.device)
        dof_pos = dof_states[:, 0].view(self.num_envs, 9, 1).to(self.device)
        dof_vel = dof_states[:, 1].view(self.num_envs, 9, 1).to(self.device)

        return rb_states, dof_pos, dof_vel

    def step(self, action=None):
        if action is not None:
            for idx, env_i in enumerate(self.envs):
                env_i.step(action[idx])

        return self._evolve_step()

    def _evolve_step(self):
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Step rendering
        self.gym.step_graphics(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)

        return self.get_state()

    def kill(self):
        self.gym.destroy_viewer(self.viewer)
        for env in self.envs:
            self.gym.destroy_env(env)
        self.gym.destroy_sim(self.sim)

class Mover():
    def __init__(self, env:environment):
        self.env = env
    
    def return_initial_pose(self):
        '''
        Make robot return to the  initial pose
        '''
        self.IK_move(goal_pos=torch.tensor(self.env.init_pos_list, device=self.env.device), 
                     goal_rot=torch.tensor(self.env.init_rot_list, device=self.env.device), 
                     closed=False, T = 200)

    def get_grasping_pose(self, name)->torch.Tensor:
        '''
        return position and rotation (quaternion)
        '''
        with open("./grasping_pose/USBStick_2.pickle", "rb") as f:
            grasp = pickle.load(f)

        z_mat = euler_xyz_to_matrix(0, 0, np.pi/2)
        grasp = torch.tensor(grasp, dtype = torch.float32, device=self.env.device) @ z_mat.repeat(100, 1, 1).to(self.env.device) # 100 for sample 100 grasping

        grasping_id = 10
        t = H_2_Transform(grasp[grasping_id, ...])

        grasp_position = [t.p.x, t.p.y, t.p.z]
        grasp_pos = []
        rb_states, dof_pos, _ = self.env.step()
        for env in range(self.env.num_envs):
            self.env.gym.refresh_rigid_body_state_tensor(self.env.sim)
            
            pos_tmp = rb_states[self.env.obj_idxs[name][env], :3].tolist()
            grasp_pos.append([grasp_position[0] + pos_tmp[0],
                            grasp_position[1] + pos_tmp[1],
                            grasp_position[2] + pos_tmp[2]]) 

        grasp_pos = torch.tensor(grasp_pos).to(self.env.device)
        grasp_rot = torch.tensor([t.r.x, t.r.y, t.r.z, t.r.w]).repeat(self.env.num_envs, 1).to(self.env.device)
        return grasp_pos, grasp_rot
    
    def get_object_predicted_pose_from_camera_id(self,
                                                 mesh: trimesh.base.Trimesh,
                                                 object_name: str,
                                                 camera_id: int = 0,
                                                 env_idx: int = 0,
                                                 debug: int = 4,):
        if object_name not in self.env.assets:
            raise AssertionError(f"You set {object_name} asset, U should set it by set_box/mesh_asset")
        
        debug_dir = f"./FoundationPose/debug/{object_name}"
        os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
        
        depth, rgb, segmentation, intrinsic, view_matrix, t = self.env.get_camera_img(id = camera_id, env_idx=env_idx, store = False)
        view_matrix[:3, :3] = view_matrix[:3, :3] @ R.from_euler("XYZ", np.array([np.pi, 0, 0])).as_matrix()
        view_matrix[:3, 3] = np.array([t.p.x, t.p.y, t.p.z])
        view_matrix[3, :3] = np.array([0., 0., 0.])
        segmentation = self.env.get_specific_object_segmentation(id = camera_id, object=object_name, env_idx=env_idx, store=False)

        depth = (-1) * depth 
        depth[(depth < 0.001) | (depth >= 0.75)] = 0 
        segmentation = segmentation.astype(bool) 
        intrinsic = intrinsic.cpu().numpy().astype(np.float64) 

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
        
        pred_pose = est.register(K=intrinsic, rgb=rgb, depth=depth, ob_mask=segmentation, iteration=5)

        if debug >= 3:
            center_pose = pred_pose@np.linalg.inv(to_origin)
            color = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_NEAREST)
            vis = draw_posed_3d_box(intrinsic, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=intrinsic, thickness=3, transparency=0, is_input_rgb=True)

            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/bbox.png', vis)

            rb_states, dof_pos, _ = self.env.step()
            H_obj_p = rb_states[self.env.obj_idxs[object_name][env_idx], :3]
            H_obj_p[2] = H_obj_p[2] + 0.1
            H_obj = pq_to_H(p=H_obj_p, q=rb_states[self.env.obj_idxs[object_name][env_idx], 3:7]).cpu().numpy()
            gt_in_cam = np.linalg.inv(view_matrix) @ H_obj

            pcd = o3d.io.read_point_cloud(f'{debug_dir}/scene_raw.ply')
            gt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            gt.transform(gt_in_cam)
            pred.transform(pred_pose)
            # o3d.visualization.draw_geometries([pcd, gt, pred])

            pcd_gt = gt.sample_points_uniformly(number_of_points=30000)
            pcd_pred = pred.sample_points_uniformly(number_of_points=30000)
            combined_pcd = pcd + pcd_gt + pcd_pred
            o3d.io.write_point_cloud(f'{debug_dir}/pose.ply', combined_pcd)
        
        pose_in_world = view_matrix @ pred_pose
        # pose_in_world = pred_pose
        return pose_in_world

    def get_predicted_hole_pose(self, camera_id, env_idx = 0, visualize = False):
        '''
        Get the predicted hole pose from specific camera (only work in 1 env)

        return:
            hole position
            hole quaternion
        '''        
        if not self.env.camera:
            raise Exception("Need to set camera first !!")
        depth, rgb, segmentation, intrinsic, view_matrix, t = self.env.get_camera_img(id = camera_id, env_idx = env_idx, store = visualize)
        depth = -depth
        view_matrix[:3, :3] = view_matrix[:3, :3] @ R.from_euler("XYZ", np.array([np.pi, 0, 0])).as_matrix()
        view_matrix[:3, 3] = np.array([t.p.x, t.p.y, t.p.z]) 
        view_matrix[3, :3] = np.array([0, 0, 0])

        predictor = CoarseMover(model_path=f'/kpts/{self.env.args.object}', model_name='pointnet2_kpts',
                               checkpoint_name='best_model.pth', use_cpu=False, out_channel=9)
        H = predictor.predict_kpts_pose(depth=depth, factor=1, K=intrinsic.cpu().numpy(), view_matrix=view_matrix, fine_grain=False, visualize=visualize)        
        H = torch.tensor(H, device = self.env.device).to(torch.float32)

        pq = H_2_Transform(H)

        return pq.p, pq.r

    # def get_predicted_hole_pose(self, camera_id):
    #     '''
    #     STILL NEED TO IMPROVE !!! (about the rotation accuracy, also not sure the performance in real-world..)
    #     Get the predicted hole pose from specific camera (only work in 1 env)

    #     return:
    #         position: gymapi.Vec3
    #         rotation: gymapi.Quat
    #     '''
        
    #     if not self.env.camera:
    #         raise Exception("Need to set camera first !!")

    #     depth, rgb, intrinsic, view_matrix, t = self.env.get_camera_img(id = camera_id, env_idx = 0)
    #     depth = -depth
    #     view_matrix[:3, :3] = view_matrix[:3, :3] @ R.from_euler("XYZ", np.array([np.pi, 0, 0])).as_matrix()
    #     view_matrix[:3, 3] = np.array([t.p.x, t.p.y, t.p.z]) 
    #     view_matrix[3, :3] = np.array([0, 0, 0])

    #     predictor = CoarseMover(model_path='/kpts/2024-05-29_03-56', model_name='pointnet2_kpts',
    #                            checkpoint_name='best_model_e_100.pth', use_cpu=False, out_channel=9)
    #     H = predictor.predict_kpts_pose(depth=depth, factor=1, K=intrinsic.cpu().numpy(), view_matrix=view_matrix, fine_grain=False, visualize=False)        
    #     H = torch.tensor(H, device = self.env.device).to(torch.float32)

    #     pq = H_2_Transform(H)

    #     return pq.p, pq.r

    def control_ik(self, dpose): 
        damping = 0.05
        _jacobian = self.env.gym.acquire_jacobian_tensor(self.env.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        j_eef = jacobian[:, self.env.franka_link_dict["panda_hand"] - 1, :, :7].to(self.env.device)

        # solve damped least squares
        j_eef_T = torch.transpose(j_eef, 1, 2).to(self.env.device)
        lmbda = torch.eye(6, device=self.env.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(self.env.num_envs, 7)
        return u
    
    def IK_move(self,
                goal_pos:torch.Tensor,
                goal_rot:torch.Tensor,
                closed: bool,
                T = 200):
        '''
        Currently only support franka
        '''
        rb_states, dof_pos, dof_vel = self.env.step()  
        pose_action = torch.zeros_like(dof_pos).squeeze(-1).to(self.env.device)

        if closed:
            grip_acts = torch.Tensor([[0., 0.]] * self.env.num_envs)
        else:
            grip_acts = torch.Tensor([[0.04, 0.04]] * self.env.num_envs)

        for _ in range(T):
            rb_states, dof_pos, _ = self.env.step() 
            hand_pos = rb_states[self.env.obj_idxs['hand'], :3]
            hand_rot = rb_states[self.env.obj_idxs['hand'], 3:7]

            # compute position and orientation error
            pos_err = goal_pos - hand_pos
            orn_err = orientation_error(goal_rot, hand_rot)

            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

            # Deploy control based on type
            pose_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + self.control_ik(dpose)
            pose_action[:, 7:9] = grip_acts

            self.env.gym.set_dof_position_target_tensor(self.env.sim, gymtorch.unwrap_tensor(pose_action))

    def generate_curobo_trajectory(self,
                                   goal_pos:list,
                                   goal_rot:list,
                                   exclude_obj_from_environment:list = None,
                                   debug:bool = False):
        '''
        goal_rot: in quaternion
        '''
        print("Generating curobo configuration......", end="")
                        
        #start state
        _dof_states = self.env.gym.acquire_dof_state_tensor(self.env.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states).to(self.env.device)
        start_position = dof_states[:, 0].view(self.env.num_envs, 1, 9).to(self.env.device)
        start_position = start_position[:, :, :7].tolist() # joint 1~7 (exclude finger)

        mesh = {}
        cuboid = {}
        if exclude_obj_from_environment is None:
            exclude_obj_from_environment = []
        for obj_name in self.env.assets:
            if obj_name in exclude_obj_from_environment: continue
            
            obj = self.env.assets[obj_name]
            obj_idx = self.env.obj_idxs[obj_name]
            rb_states, _, _ = self.env.step() 
            if "scale" in obj.keys(): # mesh
                tmp_pose = rb_states[obj_idx, ...]
                tmp_pose = tmp_pose[:, [0, 1, 2, 6, 3, 4, 5]] # [x, y, z, rx, ry, rz, rw] -> [x, y, z, rw, rx, ry, rz]

                mesh[obj_name] = {
                    "pose": tmp_pose.tolist(),
                    "file_path": obj["urdf_collisionMesh_path"],
                    "scale": obj["scale"]
                }
            else: # cuboid
                tmp_pose = rb_states[obj_idx, ...]
                tmp_pose = tmp_pose[:, [0, 1, 2, 6, 3, 4, 5]] # [x, y, z, rx, ry, rz, rw] -> [x, y, z, rw, rx, ry, rz]
                
                cuboid[obj_name] = {
                    "pose": tmp_pose.tolist(),
                    "dims": [obj["dims"].x, obj["dims"].y, obj["dims"].z]
                }

        curobo_config = { # pose: (num_env, -1)             
            "num_envs": self.env.num_envs,
            "robot_type": self.env.robot_type,
            "robot_position": [self.env.robot_pose.p.x, self.env.robot_pose.p.y, self.env.robot_pose.p.z], # 1 dimension     
            "start_state": start_position,
            "goal_position": goal_pos,
            "goal_quaternion":  goal_rot,
        } 
        curobo_config["cuboid"] = cuboid
        curobo_config["mesh"] = mesh

        world_representation = False
        show_estimate_spheres = False
        if debug:
            curobo_config_path = "./planning/yaml/curobo_config.yaml" 
            with open(curobo_config_path, 'w') as f:
                yaml.dump(curobo_config, f)
                print("DONE")
            
            world_representation = True
            show_estimate_spheres = True

        # generate trajectory
        config = copy.deepcopy(curobo_config)
        Plan = cuRobo_planning(curobo_config=config) 
        
        print("============================ planning =================================")
        start_time = time.time()
        trajectory = Plan.planning(attach_obj_name="usb", world_representation = world_representation, show_estimate_spheres = show_estimate_spheres)
        end_time = time.time()
        
        print("total time: ", end_time - start_time) 
        print("========================== planning done ==============================\n") 
        
        return trajectory 
        

    def curobo_move(self, 
                    goal_pos:list, 
                    goal_rot:list,
                    closed: bool, 
                    exclude_obj_from_environment:list =  None,
                    debug:bool = False):
        '''
        CAUTION: curobo quaternion is W/X/Y/Z, while isaac is X/Y/Z/W
        goal_rot: in quaternion 
        '''
        collision_free_traj = self.generate_curobo_trajectory(goal_pos=goal_pos, goal_rot=goal_rot,
                                                              exclude_obj_from_environment=exclude_obj_from_environment, debug=debug)
                
        max_timesteps, collision_free_traj = same_traj_dim(collision_free_traj)                
        collision_free_traj = torch.tensor(collision_free_traj).to(self.env.device)
        
        rb_states, dof_pos, dof_vel = self.env.step() 
        pose_action = torch.zeros_like(dof_pos).squeeze(-1).to(self.env.device)

        if closed:
            grip_acts = torch.tensor([[[0., 0.]]] * self.env.num_envs).view(self.env.num_envs, 2).to(self.env.device) # (num_envs, timesteps, joints) 
        else:
            grip_acts = torch.tensor([[[0.04, 0.04]]] * self.env.num_envs).view(self.env.num_envs, 2).to(self.env.device)

        current_idx = 0
        while True: 
            self.env.step() 
            pose_action = torch.cat((collision_free_traj[:, current_idx, :], grip_acts), 1).to(self.env.device)    
            self.env.gym.set_dof_position_target_tensor(self.env.sim, gymtorch.unwrap_tensor(pose_action))    
            
            _dof_states = self.env.gym.acquire_dof_state_tensor(self.env.sim)
            dof_states = gymtorch.wrap_tensor(_dof_states).to(self.env.device)
            robot_js = dof_states[:, 0].view(self.env.num_envs, 1, 9).to(self.env.device)              
            joint_diff = torch.norm((collision_free_traj[:, current_idx, :] - robot_js[:, :, :7]), dim=-1).unsqueeze(-1).to(self.env.device)
            if current_idx == max_timesteps - 1: current_idx = current_idx
            elif (joint_diff < 0.05).all(): current_idx += 1 

            if current_idx == max_timesteps - 1 and (joint_diff < 0.001).all(): break 

    def RRT_planning(self, start, end, ob, obvertex, draw):
        '''
        Given environment then planning a trajectory by RRT
        '''
        # common setting
        goal_offset = 0.1 # upper to the goal_pose.z
        collision_error = 0.01
        BoxToHand_distance = 0 # robot hand is collision ball center
        CollisionScale = 0.11 + collision_error # radius
        goal_epsilon = 0.002
        dmax = 0.05

        find_error = 0.005
        xmin = self.env.assets["table"]["obj_pos"].x - self.env.assets["table"]["dims"].x * 0.5 - goal_offset - find_error
        xmax = self.env.assets["table"]["obj_pos"].x + self.env.assets["table"]["dims"].x * 0.5 + goal_offset + find_error
        ymin = self.env.assets["table"]["obj_pos"].y - self.env.assets["table"]["dims"].y * 0.5 - goal_offset - find_error
        ymax = self.env.assets["table"]["obj_pos"].y + self.env.assets["table"]["dims"].y * 0.5 + goal_offset + find_error
        zmin = self.env.assets["table"]["obj_pos"].x + find_error * 1.5
        zmax = 1
        RangeBound = [xmin, xmax, ymin, ymax, zmin, zmax]

        planning = RRT.rrt(start,dmax,ob,end,goal_epsilon,RangeBound,BoxToHand_distance,CollisionScale,obvertex)
        path = planning.FindPath(nmax = 10000, slice_path_check = False, step = 10, Draw = draw)
            
        # path = torch.tensor([path]).to(self.env.device)
        # print(path)

        return path

    def RRT_move(self,
                 goal_pos:list,
                 goal_rot:list,
                 closed:bool,
                 exclude_obj_from_environment:list =  None,
                 draw = False):
        '''
        draw: if drawing the path (only work in 1 env)
        '''
        if exclude_obj_from_environment is None:
            exclude_obj_from_environment = []

        # planning trajectory
        paths = []
        for env in range(self.env.num_envs):
            # setting environment
            rb_states, _, _ = self.env.step()
            obs = []
            vertexes = []
            for obj_name in self.env.assets:
                if obj_name in exclude_obj_from_environment or obj_name == 'table': continue

                tmp_dims = self.env.assets[obj_name]["dims"]
                tmp_pos = rb_states[self.env.obj_idxs[obj_name][env], :3].tolist()
                tmp_rot = rb_states[self.env.obj_idxs[obj_name][env], 3:7].tolist()
                ob, vertex = Set_box_oob(tmp_pos, tmp_dims, tmp_rot)
                obs.append(ob)
                vertexes.append(vertex)
            
            # starting point
            hand_start_pos = rb_states[self.env.obj_idxs["hand"], :3].tolist()

            # planning
            path = self.RRT_planning(hand_start_pos[env], goal_pos[env], obs, vertexes, draw=draw)
            paths.append(path)
        max_timestep, paths = same_traj_dim(paths)
        paths = torch.tensor(paths).to(self.env.device)

        # execute trajectory
        goal_rot = torch.tensor(goal_rot, device=self.env.device)
        current_idx = torch.full([self.env.num_envs], 0, dtype=torch.int64).to(self.env.device)
        while True:
            rb_states, _, _ = self.env.step()

            current_node = []
            for env_idx in range(self.env.num_envs):
                current_idx_tmp = current_idx[env_idx]
                current_node.append(paths[env_idx, current_idx_tmp].tolist())
            
                if (current_idx_tmp != len(paths[env_idx]) - 1):  
                    next_idx = current_idx[env_idx] + 1
                    ToNextNode = paths[env_idx, current_idx_tmp] - rb_states[self.env.obj_idxs["hand"][env_idx], :3]
                    ToNextNode_dis = torch.norm(ToNextNode, dim = -1)
                    
                    if(ToNextNode_dis < 0.02):
                        current_node[env_idx] = paths[env_idx, next_idx].tolist()
                        current_idx[env_idx] += 1
            current_node = torch.tensor(current_node).to(self.env.device)
            self.IK_move(goal_pos=current_node, goal_rot=goal_rot, closed=closed, T=100)

            To_end_dis = torch.norm((rb_states[self.env.obj_idxs["hand"], :3] - paths[:, max_timestep - 1]), dim = -1)
            if (current_idx == max_timestep - 1).all() and (To_end_dis < 0.02).all(): break