# Standard Library
# Third Party
import numpy as np
import torch
import yaml
import copy
import os

import time

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig, Cuboid, Mesh
from curobo.geom.sphere_fit import SphereFitType

class cuRobo_planning:
    def __init__(self, curobo_config):        
        # get curobo configuration
        if isinstance(curobo_config, str): # path
            with open(curobo_config, "r") as f:
                config = yaml.safe_load(f)
        elif isinstance(curobo_config, dict):
            config = curobo_config
        
        self.num_envs = config["num_envs"]
        self.robot_type = config["robot_type"]
        self.robot_position = config["robot_position"]
        self.cuboid = config["cuboid"]
        self.mesh = config["mesh"]
        self.start_state = config["start_state"]
        self.goal_position = config["goal_position"]
        self.goal_quaternion = config["goal_quaternion"]
        
        # organize robot file name & put object position related to origin point
        self.organize_curobo_config()
    
    def compute_position_for_robot_on_ori(self, query):
        # [position + pose] or [position], self.robot_position:[position]        
        if (len(query) > 3): # [position + pose]
            position = [query[0] - self.robot_position[0], query[1] - self.robot_position[1], query[2] - self.robot_position[2]]
            quaternion = query[3::]
            
            return position + quaternion

        else: # only [position]
            position = [query[0] - self.robot_position[0], query[1] - self.robot_position[1], query[2] - self.robot_position[2]]
        
        return position  
        
    def organize_curobo_config(self):       
        # get robot file name
        if self.robot_type == "franka":
            self.robot_file = "franka.yml"
        
        # Robot is at the origin point in cuRobo, but it may not be in the same position in IsaacGym 
        for key, value in self.mesh.items():
            for env in range(self.num_envs):
                self.mesh[key]["pose"][env] = self.compute_position_for_robot_on_ori(self.mesh[key]["pose"][env])
        
        for key, value in self.cuboid.items():
            for env in range(self.num_envs):
                self.cuboid[key]["pose"][env] = self.compute_position_for_robot_on_ori(self.cuboid[key]["pose"][env])
        
        for env in range(self.num_envs):
            self.goal_position[env] = self.compute_position_for_robot_on_ori(self.goal_position[env])          

    def planning(self,
                 attach_obj_name = None,
                 world_representation = False,
                 show_estimate_spheres = False):        
                       
        t1 = time.time()
        results = []
        for env in range(self.num_envs):
            cuboid = []
            for key, value in self.cuboid.items():
                tmp = Cuboid(
                    name = key,
                    pose = self.cuboid[key]["pose"][env],
                    dims = self.cuboid[key]["dims"],
                    )            
                cuboid.append(tmp)
            
            mesh = []
            for key, value in self.mesh.items():
                tmp = Mesh(
                    name = key,
                    pose = self.mesh[key]["pose"][env],
                    file_path = self.mesh[key]["file_path"],
                    scale = self.mesh[key]["scale"],
                    )            
                mesh.append(tmp)
            
            world_config = WorldConfig(
                cuboid = cuboid,
                mesh = mesh,
                )
            
            start_state = self.start_state[env]
            goal_position = self.goal_position[env]
            goal_quaternion = self.goal_quaternion[env]            
            
            if attach_obj_name is not None:
                obj_pose = self.mesh[attach_obj_name]["pose"][env]
                list_obstacle = [
                    Mesh(
                        name = attach_obj_name,
                        pose = obj_pose,
                        file_path = self.mesh[attach_obj_name]["file_path"],
                        scale = self.mesh[attach_obj_name]["scale"],
                    )
                ]                
                t2 = time.time()
                print("Setup time: ", t2 - t1)
                result = self.motion_gener_single(world_config, start_state, goal_position, goal_quaternion,
                                                  list_obstacle, show_estimate_spheres=show_estimate_spheres)
            else:
                result = self.motion_gener_single(world_config, start_state, goal_position, goal_quaternion)
            
            results.append(result)
        
        if world_representation:
            if self.num_envs == 1:
                world_representation = world_representation
                self.world_representation()
        
            else:
                assert "world representation only in 1 env"
        
        return results
    
    def motion_gener_single(self, world_config, start_state, goal_position, goal_quaternion,
                            list_obstacle = None, show_estimate_spheres = False):  
        
        tensor_args = TensorDeviceType()
        
        robot_file = self.robot_file           
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_file,
            world_config,
            tensor_args,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            use_cuda_graph=True,
            num_trajopt_seeds=12,
            num_graph_seeds=1,
            num_ik_seeds=30,
        )
        
        motion_gen = MotionGen(motion_gen_config)
        
        Start_State = JointState.from_position(
                position=tensor_args.to_device(start_state),
                joint_names=[   "panda_link1",
                                "panda_link2",
                                "panda_link3",
                                "panda_link4",
                                "panda_link5",
                                "panda_link6",
                                "panda_link7"],
            )
        
        if list_obstacle is not None:
            cu_js = JointState(
            position=tensor_args.to_device(start_state),
            joint_names=[   "panda_link1",
                            "panda_link2",
                            "panda_link3",
                            "panda_link4",
                            "panda_link5",
                            "panda_link6",
                            "panda_link7"],
            ) 
            
            t1 = time.time()
            motion_gen.attach_external_objects_to_robot(
                cu_js,
                list_obstacle,
                sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
                surface_sphere_radius = 0.0075,
                show_estimate_spheres=show_estimate_spheres
            )
            t2 = time.time()
            print("attach time: ", t2 - t1)
                
        max_attempts = 10
        plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=2,
            max_attempts=max_attempts,
            enable_finetune_trajopt=True,
            parallel_finetune=True,
        )
        
        ik_goal = Pose(
                    position=tensor_args.to_device(goal_position),
                    quaternion=tensor_args.to_device(goal_quaternion),
                )    
        
        t1 = time.time()
        result = motion_gen.plan_single(Start_State, ik_goal, plan_config)
        t2 = time.time()
        print("planning time: ", t2 - t1)
       
        print("Trajectory Generated: ", result.success.item())
        if result.success.item():
            traj = result.optimized_plan.position # optimized plan
            
            return traj.tolist()
        
        else:
            return start_state
    
    def world_representation(self):
        cuboid = []
        for key, value in self.cuboid.items():
            tmp = Cuboid(
                name = key,
                pose = self.cuboid[key]["pose"][0],
                dims = self.cuboid[key]["dims"],
                )
            
            cuboid.append(tmp)
            
        mesh = []
        for key, value in self.mesh.items():
            tmp = Mesh(
                name = key,
                pose = self.mesh[key]["pose"][0],
                file_path = self.mesh[key]["file_path"],
                scale = self.mesh[key]["scale"],
            )
            
            mesh.append(tmp)
            
        world_config = WorldConfig(
                cuboid = cuboid,
                mesh = mesh,
            )    
            
        world_file_name = "debug_world_mesh.obj"
        world_file_path = os.path.join(self.mesh[key]["file_path"], "..", world_file_name)
        
        # world_config.save_world_as_mesh(world_file_path)
            
        cuboid_world = WorldConfig.create_obb_world(world_config)
        cuboid_world.save_world_as_mesh(world_file_path)

if __name__ == "__main__":    
    Plan = cuRobo_planning('./planning/yaml/curobo_config.yaml') # same as curobo_congif_path in simulate_curobo.
    
    print("============================ planning =================================")
    start_time = time.time()
    Plan.planning(attach_obj_name="usb", world_representation = False, show_estimate_spheres = False)
    # Plan.planning(attach_obj=False, attach_obj_name=None, world_representation=False)
    end_time = time.time()
    
    print("total time: ", end_time - start_time) 
    print("========================== planning done ==============================\n") 
