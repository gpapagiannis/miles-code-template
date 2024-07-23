#!/usr/bin/env python3

import time
from franka_ros_controller.frankarosjointcontroller import FrankaROSJointController
from closed_loop.utils import *
import pickle as pkl
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
from torchvision.transforms import CenterCrop, Resize
from utils import *
from dino_vit_features import find_sim
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp

ASSETS_DIR = "/path/to/assets"
SIMILARITY_THRESHOLD = .94

class DataCollector:
    def __init__(self,
                 task_name='test',
                 number_of_samples=2000,
                 close_gripper_at_start=True,
                 open_gripper_at_start=True,
                 reset_task=True,
                 data_collection_recording_rate=10,
                 validate_demo=False,
                 resume_data_collection=False,
                 collect_last_inch=False,
                 shorten_demo=False,
                 nearest_neighbour_data_collection=False
                 ):

        self.task_name = task_name
        self._resume_data_collection = resume_data_collection
        self._collect_last_inch = collect_last_inch

        make_dir(os.path.join(ASSETS_DIR, 'tasks'))
        make_dir(os.path.join(ASSETS_DIR, 'tasks', task_name))
        make_dir(os.path.join(ASSETS_DIR, 'tasks', task_name, 'data'))
        make_dir(os.path.join(ASSETS_DIR, 'tasks', task_name, 'data', 'closed_loop'))
        make_dir(os.path.join(ASSETS_DIR, 'tasks', task_name, 'data', 'last_inch'))
        make_dir(os.path.join(ASSETS_DIR, 'tasks', task_name, 'data', 'NN'))
        make_dir(os.path.join(ASSETS_DIR, 'tasks', task_name, 'demonstration'))
        self.franka = FrankaROSJointController(control_frequency=250, use_camera=True, reset_on_init=False, camera_name='wrist_camera')
        self._recording_rate = data_collection_recording_rate
        self.nearest_neighbour_data_collection = nearest_neighbour_data_collection
        self.data_folder_name = 'closed_loop'
        if nearest_neighbour_data_collection:
            self.data_folder_name = 'NN'
        if open_gripper_at_start:
            input("Press enter to open gripper")
            self.franka.open_gripper()
        if close_gripper_at_start:
            input("Press enter to close gripper")
            if self.franka.is_gripper_open():
                print("closing gripper")
                self.franka.grasp()
                input("Gripper clsed")
        demonstration_dir = "{}/tasks/{}/demonstration".format(ASSETS_DIR, task_name)
        demonstration_path = "{}/tasks/{}/demonstration/recorded_demo.pkl".format(ASSETS_DIR, task_name)
        demonstration_data_dir = "{}/tasks/{}/data/{}/".format(ASSETS_DIR, task_name, self.data_folder_name)
        starting_pose_path = "{}/tasks/{}/demonstration/starting_pose.pkl".format(ASSETS_DIR, task_name)
        self.demonstration_dir = demonstration_dir
        if os.path.isfile(demonstration_path) and not reset_task:
            self.demonstration = pkl.load(open(demonstration_path, 'rb'))
            self.starting_pose = pkl.load(open(starting_pose_path, 'rb'))
            self.gripper_states = pkl.load(open("{}/gripper_states.pkl".format(demonstration_dir), 'rb'))
            if not validate_demo:
                self.demonstration_state_actions = torch.load("{}/demonstration_data.pt".format(
                    demonstration_data_dir))  # Load the state - actions pairs observed in the demonstration
                self.demonstration_state_actions = {
                    'data_imgs': self.demonstration_state_actions['data_imgs'],
                    'data_forces': self.demonstration_state_actions['data_forces'],
                    # 'data_actions': self.demonstration_in_base,
                    'in_trajectory_identifier': self.demonstration_state_actions['in_trajectory_identifier'],
                    'is_on_waypoint': self.demonstration_state_actions['is_on_waypoint']}
            print("Number of demonstration waypoints: {}".format(len(self.demonstration)))
            if shorten_demo:
                demonstration_accelerated = []
                gripper_states_accelerated = []
                c = 0
                input("Shorten")
                for d in self.demonstration:
                    if c % 4 == 0:
                        demonstration_accelerated.append(d)
                        gripper_states_accelerated.append(self.gripper_states[c])
                    c+=1
                print(len(demonstration_accelerated), len(self.demonstration))
                pkl.dump(demonstration_accelerated,
                         open("{}/tasks/{}/demonstration/recorded_demo.pkl".format(ASSETS_DIR, task_name), 'wb'))
                pkl.dump(gripper_states_accelerated, open("{}/tasks/{}/demonstration/gripper_states.pkl".format(ASSETS_DIR, task_name), 'wb'))
        else:
            self.set_starting_pose()
            time.sleep(1)
            self.starting_pose = pkl.load(open(starting_pose_path, 'rb'))
            self.franka.record_demonstration(control_freq=self._recording_rate, demonstration_path=demonstration_dir)
            self.franka.go_to_pose(self.starting_pose)
            time.sleep(1)
            self.demonstration = pkl.load(open(demonstration_path, 'rb'))

        self.demonstration_in_base = copy.deepcopy(self.convert_demo_trajectory_to_base_frame())
        self.number_of_samples = number_of_samples
        self._data_imgs = torch.zeros(self.number_of_samples, 20, 3, 64, 64)
        self._data_forces = torch.zeros(self.number_of_samples, 20, 6)
        self._data_actions = torch.zeros(self.number_of_samples, 20, 4, 4)
        self._in_trajectory_identifier = torch.zeros(self.number_of_samples, 20, 1)
        self._number_data_collected = 0
        self._last_identifier = 0
        # Load saved data using torch.load
        if validate_demo:
            self.validate_trajectory(path=demonstration_dir)
    def validate_trajectory(self, path):
        self.franka.go_to_pose(self.starting_pose)
        self.franka.replay_demonstration(replay_rate=self._recording_rate,
                                                     task_name=self.task_name,
                                                     demonstration_path=path,
                                                     assets_path=ASSETS_DIR,
                                                     save=True,
                                                     frame='base')
        demonstration_dir = "{}/tasks/{}/demonstration".format(ASSETS_DIR, 'open_box_main_good2')
        self.franka.replay_kinesthetic_demonstration(control_freq=self._recording_rate, demonstration_path=demonstration_dir)
    def save_data(self):
        if self._collect_last_inch:
            path = "{}/tasks/{}/data/last_inch/recorded_data.pt".format(ASSETS_DIR, self.task_name)
        else:
            path = "{}/tasks/{}/data/coarse/recorded_data_coarse.pt".format(ASSETS_DIR, self.task_name)

        data = {"data_imgs": self._data_imgs,
                "data_forces": self._data_forces,
                "data_actions": self._data_actions,
                "in_trajectory_identifier": self._in_trajectory_identifier,
                "number_data_collected": self._number_data_collected,
                "last_identifier": self._last_identifier}
        torch.save(data, path)

    def set_starting_pose(self):
        self.franka.guide_mode(duration=5, show_FLIR=True)
        time.sleep(1)
        bottleneck = self.franka.get_eef_pose()
        self.franka.go_to_pose(bottleneck)
        print("Vertical starting pose set.")
        pkl.dump(bottleneck, open("{}/tasks/{}/demonstration/starting_pose.pkl".format(ASSETS_DIR, self.task_name), 'wb'))

    def convert_demo_trajectory_to_base_frame(self):
        demo_in_base = []
        for i, pose in enumerate(self.demonstration):
            demo_in_base.append(self.starting_pose @ pose)
        return demo_in_base

    def get_random_transformation(self, linear_range=None, angular_range=None, coarse=False, last_inch=False):
        if linear_range is None:
            linear_range = [-.02, .02]
        if angular_range is None:
            angular_range = [-2, 2]
        if coarse:
            if not last_inch:
                linear_range = [-.1, .1]
            angular_range = [-30, 30]
        angular_range[0] = np.deg2rad(angular_range[0])
        angular_range[1] = np.deg2rad(angular_range[1])
        linear = np.random.uniform(low=linear_range[0], high=linear_range[1], size=3)
        angular = np.random.uniform(low=angular_range[0], high=angular_range[1], size=3)
        if coarse:
            angular[0] = .0
            angular[1] = .0
            if not last_inch:
                linear[2] = .0
        transformation = np.eye(4)
        transformation[:3, 3] = linear
        transformation[:3, :3] = self.franka.se3_transforms.euler2rot("xyz", angular)
        return transformation

    def process_img(self, img):
        center_crop = CenterCrop(480)
        img_resize = Resize((128, 128))
        if img.shape[0] == img.shape[1]:
            img = img_resize(torch.tensor(img).permute(2, 0, 1) / 255)
        else:
            img = center_crop(torch.tensor(img).permute(2, 0, 1) / 255)
            img = img_resize(img)
        return img

    def compute_dino_similarity(self, demo_img):
        current_img = self.franka.camera.get_rgb()  # .permute(1, 2, 0).cpu().numpy() * 255  # Querying before processing to ensure data is in sync
        current_img = self.process_img(current_img).permute(1, 2, 0).cpu().numpy() * 255
        current_img = current_img.astype(np.uint8)
        demo_img = demo_img.astype(np.uint8)
        sim, image1_batch, image2_batch, extractor, similarities_patch_map, similarities = find_sim.sim(current_img, demo_img)
        image1 = (image1_batch[0].permute(1, 2, 0).cpu().numpy() * extractor.std) + extractor.mean
        image2 = (image2_batch[0].permute(1, 2, 0).cpu().numpy() * extractor.std) + extractor.mean
        if sim < SIMILARITY_THRESHOLD:
            fig, ax = plt.subplots(2, 2)
            ax[0, 0].imshow(image1)
            ax[0, 0].set_title("Image 1")
            ax[0, 1].imshow(image2)
            ax[0, 1].set_title("Image 2")
            ax[1, 0].imshow(.5 * image1 + .5 * image2)
            ax[1, 0].set_title("Overlayed")
            sim_im = ax[1, 1].imshow(similarities_patch_map[:, :, 0].cpu().numpy(), cmap='gray')
            ax[1, 1].set_title("Cosine similarity: {:.2f}".format(torch.mean(similarities).item()))
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(sim_im, cax=cbar_ax)
            # Remove grid around plots
            for i in range(2):
                for j in range(2):
                    ax[i, j].grid(False)
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
            plt.show()
        return sim

    def collect_data(self):
        def go_to_starting_pose_ori():
            # this is just to avoid any damages for tasks where held object may get stuck
            current_eef = self.franka.get_eef_pose()
            current_eef[:3, :3] = self.starting_pose[:3, :3]
            self.franka.go_to_pose(current_eef, timeout=3)

        input("Press enter to move to starting pose to start data collection")
        self.franka.go_to_pose(self.starting_pose)
        data_collected = 0
        demo_length = len(self.demonstration_in_base)
        demo_index = 0
        trajectories_collected = 0
        trajectories_per_way_point = 5
        environment_requires_reset = False
        distance_errors = np.array([.0, .0, .0])
        rotation_errors = np.array([.0, .0, .0])
        if self._resume_data_collection:
            path = "{}/tasks/{}/data/{}/data_collection_details.pt".format(ASSETS_DIR, self.task_name, self.data_folder_name)
            data_collection_details = torch.load(path)
            trajectories_collected = data_collection_details["number_of_trajectories_collected"]
            trajectories_per_way_point = data_collection_details["number_of_trajectories_per_way_point"]
            demo_index = int(np.floor(trajectories_collected / trajectories_per_way_point))
            remaining_trajectories_for_current_index = trajectories_collected % trajectories_per_way_point
            trajectories_per_way_point -= remaining_trajectories_for_current_index
            print(bcolors.UNDERLINE + bcolors.BOLD +
                  "[RESUMING DATA COLLECTION] Trajectories collected: {} | Demo index: {} | Trajectories remaining for current index: {}".format(
                      trajectories_collected, demo_index, trajectories_per_way_point) + bcolors.ENDC)
        while demo_index < demo_length and not environment_requires_reset:
            number_of_waypoint_data_collected = 0
            self._number_data_collected = data_collected
            self._last_identifier = demo_index
            if demo_index > 0:
                """
                Before moving to the next waypoint to collect data, first move to the previous waypoint to make sure
                that the next waypoint is always reachable."""
                go_to_starting_pose_ori()
                self.franka.go_to_pose(self.starting_pose)
                self.franka.replay_demonstration(replay_rate=self._recording_rate,
                                                             demonstration_path=self.demonstration_dir,
                                                             task_name=self.task_name,
                                                             until_index=demo_index + 1,
                                                             frame='base')
                self.franka.go_to_pose(self.demonstration_in_base[demo_index], timeout=3)
                self.franka.reset_async_tracking()
                time.sleep(.1)


            """
            Sample the current waypoint demo."""
            current_demo_pose = self.demonstration_in_base[demo_index]
            """
            Move to current waypoint for which data will be collected."""
            self.franka.go_to_pose(current_demo_pose)
            if self.gripper_states[demo_index] == 1:
                if self.franka.gripper_state == 0:
                    print("closing gripper")
                    self.franka.grasp()
            elif self.gripper_states[demo_index] == 0:
                if self.franka.gripper_state == 1:
                    self.franka.open_gripper()
                    print("opening gripper")
            time.sleep(.1)
            pose_reached = copy.deepcopy(self.franka.get_eef_pose())
            demo_img = self.franka.camera.get_rgb()  # Querying before processing to ensure data is in sync
            demo_img = self.process_img(demo_img).permute(1, 2, 0).cpu().numpy() * 255
            input("Continue to next waypoint?")
            print(bcolors.OKBLUE + "[UPDATE] Moved to new waypoint: {}".format(demo_index))
            traj_num = 0
            while traj_num < trajectories_per_way_point:
                t_images = []
                t_forces = []
                t_identifiers = []
                t_actions = []
                t_is_on_waypoint = []
                gripper_states = []
                loop_time = time.time()
                if data_collected >= self.number_of_samples:
                    break

                similarity = self.compute_dino_similarity(demo_img)
                print(bcolors.OKBLUE + "[UPDATE] Similarity: {}".format(similarity) + bcolors.ENDC)
                if similarity < SIMILARITY_THRESHOLD and np.max(distance_errors) < .0010000001 and np.max(rotation_errors) < .7:
                    inp = input("Stop data collection? (y/n)")
                    if inp == "y":
                        environment_requires_reset = True
                        print(bcolors.FAIL + "[STOPPING DATA COLLECTION] Environment disturbance detected." + bcolors.ENDC)
                        break

                """
                Sample a new random pose and go to it to collect augmentation trajectory. """
                transformation = self.get_random_transformation()
                print(bcolors.OKBLUE + "[UPDATE] Transformation: {}".format(transformation[:3, 3]) + bcolors.ENDC)
                print("Transformation euler: {}".format(self.franka.se3_transforms.rot2euler("xyz", transformation[:3, :3], degrees=True)))
                new_pose = current_demo_pose @ transformation
                tracked_poses = False
                self.franka.go_to_pose_in_base_async(new_pose,
                                                     tracking_rate=250,
                                                     max_linear_speed=.0002,
                                                     max_angular_speed=.002,
                                                     stop_when_target_pose_reached=True,
                                                     make_final_correction=False,
                                                     timeout=1.5,
                                                     duration=.5,
                                                     log=False)

                time.sleep(.1)
                # traj_poses = []
                while not tracked_poses:
                    while self.franka.is_tracking_poses_async:
                        # traj_poses.append(self.franka.get_eef_pose())
                        force_felt = self.franka.get_eef_wrench()

                        # manually specified force thresholds for safety
                        if np.abs(force_felt[0]) > 80 \
                                or np.abs(force_felt[1]) > 80 \
                                or np.abs(force_felt[2]) > 80 \
                                or force_felt[2] < -80 \
                                or np.abs(force_felt[3]) > 45 \
                                or np.abs(force_felt[4]) > 45 \
                                or np.abs(force_felt[5]) > 45:
                            print("Force threshold exceeded. Skipping sample", force_felt)
                            self.franka.reset_async_tracking()
                            time.sleep(.1)
                    tracked_poses = True
                time.sleep(.1)
                """
                Go back to current_demo_pose to record the corrective
                trajectory."""
                self.franka.reset_async_tracking()
                self.franka.go_to_pose_in_base_async(current_demo_pose,
                                                     tracking_rate=250,
                                                     max_linear_speed=.0001,
                                                     max_angular_speed=.001,
                                                     stop_when_target_pose_reached=True,
                                                     make_final_correction=False,
                                                     timeout=3.,
                                                     duration=.5,
                                                     log=False)
                tracked_poses = False
                while not tracked_poses:
                    while self.franka.is_tracking_poses_async:
                        stime = time.time()
                        force_felt = self.franka.get_eef_wrench()
                        t_forces.append(copy.deepcopy(force_felt))
                        c_pose = self.franka.get_eef_pose()
                        t_actions.append(copy.deepcopy(c_pose))
                        temp_img = self.franka.camera.get_rgb()  # Querying before processing to ensure data is in sync
                        t_images.append(self.process_img(temp_img).cpu().numpy())
                        t_identifiers.append([demo_index])
                        gripper_states.append(self.gripper_states[demo_index])
                        if np.abs(pose_reached[0, 3] - c_pose[0, 3]) < .001 or np.abs(
                                pose_reached[1, 3] - c_pose[1, 3]) < .001 or np.abs(
                            pose_reached[2, 3] - c_pose[2, 3]) < .001:
                            t_is_on_waypoint.append([1])
                        else:
                            t_is_on_waypoint.append([0])

                        if np.abs(force_felt[0]) > 80 \
                                or np.abs(force_felt[1]) > 80 \
                                or np.abs(force_felt[2]) > 80 \
                                or force_felt[2] < -80 \
                                or np.abs(force_felt[3]) > 45 \
                                or np.abs(force_felt[4]) > 45 \
                                or np.abs(force_felt[5]) > 45:
                            print("Force threshold exceeded. Skipping sample", force_felt)
                            self.franka.reset_async_tracking()
                            time.sleep(.1)
                        data_collected += 1
                        number_of_waypoint_data_collected += 1
                        sleep_time = np.max((1 / self._recording_rate - (time.time() - stime), 0))
                        time.sleep(sleep_time)  # Sync loop to self._recording_rate Hz as recording
                    tracked_poses = True
                print(bcolors.WARNING + "[ERROR] linear: {}, angular: {}".format(
                    self.franka.get_eef_pose()[:3, 3] - current_demo_pose[:3, 3],
                    self.franka.se3_transforms.rot2euler("xyz", self.franka.get_eef_pose()[:3, :3], degrees=True) -
                    self.franka.se3_transforms.rot2euler("xyz", current_demo_pose[:3, :3], degrees=True)))

                achieved_pose = self.franka.get_eef_pose()

                print(bcolors.WARNING + "[ERROR ACHIEVED POSE] linear: {}, angular: {}".format(
                    achieved_pose[:3, 3] - pose_reached[:3, 3],
                    self.franka.se3_transforms.rot2euler("xyz",achieved_pose[:3, :3], degrees=True) -
                    self.franka.se3_transforms.rot2euler("xyz", pose_reached[:3, :3], degrees=True)))

                achieved_rot = self.franka.se3_transforms.rot2euler("xyz", achieved_pose[:3, :3], degrees=True)
                reached_rot = self.franka.se3_transforms.rot2euler("xyz", pose_reached[:3, :3], degrees=True)
                distance_errors = np.array([np.abs(achieved_pose[0, 3] - pose_reached[0, 3]), np.abs(
                    achieved_pose[1, 3] - pose_reached[1, 3]), np.abs(
                    achieved_pose[2, 3] - pose_reached[2, 3])])
                rotation_errors = np.array([np.abs(achieved_rot[0] - reached_rot[0]), np.abs(
                    achieved_rot[1] - reached_rot[1]), np.abs(
                    achieved_rot[2] - reached_rot[2])])
                
                # adjust the reachability constraints as needed for your robot
                if (np.abs(achieved_pose[0, 3] - pose_reached[0, 3]) > .001 or np.abs(
                        achieved_pose[1, 3] - pose_reached[1, 3]) > .001 or np.abs(
                    achieved_pose[2, 3] - pose_reached[2, 3]) > .001 or np.abs(achieved_rot[0] - reached_rot[0]) > .5
                        or np.abs(achieved_rot[1] - reached_rot[1]) > .5 or np.abs(achieved_rot[2] - reached_rot[2]) > .5):
                    print(
                        "[REACHABILITY] Could not accurately return back to waypoint. Skipping sample. Returning back to starting pose and will move to waypoint again.")
                    # go_to_starting_pose_ori()
                    self.franka.go_to_pose(self.starting_pose, timeout=4)
                    self.franka.replay_demonstration(replay_rate=self._recording_rate,
                                                                 demonstration_path=self.demonstration_dir,
                                                                 task_name=self.task_name,
                                                                 until_index=demo_index + 1,
                                                                 frame='base')
                    self.franka.go_to_pose(self.demonstration_in_base[demo_index], timeout=3)
                    self.franka.reset_async_tracking()
                    t_images = []
                    t_forces = []
                    t_identifiers = []
                    t_actions = []
                    t_is_on_waypoint = []
                    gripper_states = []
                    continue
                # input("Next trajectory")
                t_forces = np.array(t_forces)
                t_images = np.array(t_images)
                t_identifiers = np.array(t_identifiers)
                t_actions = np.array(t_actions)
                t_is_on_waypoint = np.array(t_is_on_waypoint)
                gripper_states = np.array(gripper_states)

                forces_torch = torch.from_numpy(t_forces).float()
                images_torch = torch.from_numpy(t_images).float()
                identifiers_torch = torch.from_numpy(t_identifiers).float()
                actions_torch = torch.from_numpy(t_actions).float()
                is_on_waypoint_torch = torch.from_numpy(t_is_on_waypoint).float()
                gripper_states_torch = torch.from_numpy(gripper_states).float()
                # Save the above data
                path = "{}/tasks/{}/data/{}/trajectory_sample_{}.pt".format(ASSETS_DIR, self.task_name, self.data_folder_name,
                                                                                     trajectories_collected)

                data = {"data_imgs": images_torch,
                        "data_forces": forces_torch,
                        "data_actions": actions_torch,
                        "in_trajectory_identifier": identifiers_torch,
                        "gripper_state": gripper_states_torch,
                        "is_on_waypoint": is_on_waypoint_torch}
                torch.save(data, path)

                trajectories_collected += 1
                traj_num += 1
                data_collection_details = {"number_of_trajectories_collected": trajectories_collected,
                                           "number_of_trajectories_per_way_point": trajectories_per_way_point,
                                           "demo_index": demo_index}
                path_2 = "{}/tasks/{}/data/{}/data_collection_details.pt".format(ASSETS_DIR, self.task_name, self.data_folder_name)
                torch.save(data_collection_details, path_2)

                print("Collected {} samples. Trajectories for this waypoint: {}. Demo index: {}. Time take: {}".format(
                    data_collected, traj_num, demo_index,
                    time.time() - loop_time))

            demo_index += 1  # After collecting all the trajectories for the current waypoint, move to the next.
            # if self._resume_data_collection:
            trajectories_per_way_point = 5
        go_to_starting_pose_ori()
        self.franka.go_to_pose(self.starting_pose)

    
    def collect_data_alternative_method(self): 
        # this method tracks the impedance controller's equilibrium points directly instead of the eef pose as the other function
        # with a well tuned impedance controller both methods should perform identically

        input("Press enter to move to starting pose to start data collection")
        self.franka.go_to_pose(self.starting_pose)
        data_collected = 0
        demo_length = len(self.demonstration_in_base)
        demo_index = 0
        trajectories_collected = 0
        trajectories_per_way_point = 10
        environment_requires_reset = False
        distance_errors = np.array([.0, .0, .0])
        rotation_errors = np.array([.0, .0, .0])
        if self._resume_data_collection:
            path = "{}/tasks/{}/data/closed_loop/data_collection_details.pt".format(ASSETS_DIR, self.task_name)
            data_collection_details = torch.load(path)
            trajectories_collected = data_collection_details["number_of_trajectories_collected"]
            trajectories_per_way_point = 10#data_collection_details["number_of_trajectories_per_way_point"]
            demo_index = int(np.floor(trajectories_collected / trajectories_per_way_point))
            remaining_trajectories_for_current_index = trajectories_collected % trajectories_per_way_point
            trajectories_per_way_point -= remaining_trajectories_for_current_index
            print(bcolors.UNDERLINE + bcolors.BOLD +
                  "[RESUMING DATA COLLECTION] Trajectories collected: {} | Demo index: {} | Trajectories remaining for current index: {}".format(
                      trajectories_collected, demo_index, trajectories_per_way_point) + bcolors.ENDC)

        while demo_index < demo_length and not environment_requires_reset:
            number_of_waypoint_data_collected = 0
            self._number_data_collected = data_collected
            self._last_identifier = demo_index
            if demo_index > 0:
                """
                Before moving to the next waypoint to collect data, first move to the previous waypoint to make sure
                that the next waypoint is always reachable."""
                self.franka.go_to_pose(self.starting_pose)
                self.franka.replay_demonstration(replay_rate=self._recording_rate,
                                                             demonstration_path=self.demonstration_dir,
                                                             task_name=self.task_name,
                                                             until_index=demo_index + 1,
                                                             frame='base')
                self.franka.go_to_pose(self.demonstration_in_base[demo_index])
                self.franka.reset_async_tracking()

                time.sleep(5.1)


            """
            Sample the current waypoint demo."""
            current_demo_pose = self.demonstration_in_base[demo_index]
            """
            Move to current waypoint for which data will be collected."""
            self.franka.go_to_pose(current_demo_pose)

            if self.gripper_states[demo_index] == 1:
                if self.franka.gripper_state == 0:
                    print("closing gripper")
                    self.franka.grasp()
            elif self.gripper_states[demo_index] == 0:
                if self.franka.gripper_state == 1:
                    self.franka.open_gripper()
                    print("opening gripper")
            time.sleep(.1)
            pose_reached = copy.deepcopy(self.franka.get_eef_pose())
            demo_img = self.franka.camera.get_rgb()  # Querying before processing to ensure data is in sync
            demo_img = self.process_img(demo_img).permute(1, 2, 0).cpu().numpy() * 255
            input("Continue to next waypoint?")
            print(bcolors.OKBLUE + "[UPDATE] Moved to new waypoint: {}".format(demo_index))
            traj_num = 0
            while traj_num < trajectories_per_way_point:
                t_images = []
                t_forces = []
                t_identifiers = []
                t_actions = []
                t_is_on_waypoint = []
                gripper_states = []
                loop_time = time.time()
                if data_collected >= self.number_of_samples:
                    break

                similarity = self.compute_dino_similarity(demo_img)
                print(bcolors.OKBLUE + "[UPDATE] Similarity: {}".format(similarity) + bcolors.ENDC)
                if similarity < SIMILARITY_THRESHOLD and np.max(distance_errors) < .0010000001 and np.max(rotation_errors) < .7:
                    inp = input("Stop data collection? (y/n)")
                    if inp == "y":
                        environment_requires_reset = True
                        print(bcolors.FAIL + "[STOPPING DATA COLLECTION] Environment disturbance detected." + bcolors.ENDC)
                        break

                """
                Sample a new random pose and go to it to collect data. """
                transformation = self.get_random_transformation()
                new_pose = current_demo_pose @ transformation
                tracked_poses = False
                current_joints = self.franka.get_joints_state() + np.random.normal(0, .001, 7)
                def get_trajectory_to_pose(pose, from_pose=None):
                    stime = time.time()

                    current_pose = from_pose
                    traj_weights = [t / 20 for t in range(1, 21)]
                    traj_weights[-1] = 1  # To avoid numerical errors
                    traj_weights = np.asarray(traj_weights)
                    x = current_pose[0, 3] * (1 - traj_weights) + pose[0, 3] * traj_weights
                    y = current_pose[1, 3] * (1 - traj_weights) + pose[1, 3] * traj_weights
                    z = current_pose[2, 3] * (1 - traj_weights) + pose[2, 3] * traj_weights
                    orientations = Rotation.from_matrix([current_pose[:3, :3], pose[:3, :3]])
                    slerp = Slerp([0, 1], orientations)
                    interp_ori = slerp(traj_weights)

                    return {"x": x, "y": y, "z": z, "ori": interp_ori, "num_of_steps": traj_weights.shape[0]}

                trajectory = get_trajectory_to_pose(new_pose, from_pose=current_demo_pose)
                n_poses = 0
                for t in range(len(trajectory['x'])):


                    temp_pose = np.eye(4)
                    temp_pose[0, 3] = trajectory['x'][t]
                    temp_pose[1, 3] = trajectory["y"][t]
                    temp_pose[2, 3] = trajectory["z"][t]
                    temp_pose[:3, :3] = trajectory["ori"][t].as_matrix()
                    self.franka.go_to_pose_in_base_async(temp_pose,
                                                     tracking_rate=250,
                                                     max_linear_speed=.0001,
                                                     max_angular_speed=.001,
                                                     stop_when_target_pose_reached=False,
                                                     make_final_correction=False,
                                                     timeout=2.5,
                                                     duration=.5,
                                                     log=False)
                    time.sleep(.1)
                self.franka.reset_async_tracking()
                time.sleep(.1)
                while not tracked_poses:
                    while self.franka.is_tracking_poses_async:
                        force_felt = self.franka.get_eef_wrench()
                        if np.abs(force_felt[0]) > 50 \
                                or np.abs(force_felt[1]) > 50 \
                                or np.abs(force_felt[2]) > 50 \
                                or np.abs(force_felt[3]) > 20 \
                                or np.abs(force_felt[4]) > 20 \
                                or np.abs(force_felt[5]) > 20:
                            print("Force threshold exceeded. Skipping sample")
                            self.franka.reset_async_tracking()
                            time.sleep(.1)
                    tracked_poses = True
                time.sleep(.1)

                """
                Go back to current_demo_pose to record the corrective
                trajectory."""
                self.franka.reset_async_tracking()

                tracked_poses = False
                # for p_index in range(len(traj_poses) - 2, -1, -1):
                while not tracked_poses:
                    for t in range(len(trajectory['x']) - 1, -4, -1):
                        if t < 0:
                            temp_pose = current_demo_pose
                        else:
                            temp_pose = np.eye(4)
                            temp_pose[0, 3] = trajectory['x'][t]
                            temp_pose[1, 3] = trajectory["y"][t]
                            temp_pose[2, 3] = trajectory["z"][t]
                            temp_pose[:3, :3] = trajectory["ori"][t].as_matrix()
                        self.franka.go_to_pose_in_base_async(temp_pose,  # traj_poses[len(traj_poses) - 1],
                                                             tracking_rate=250,
                                                             max_linear_speed=.0001,
                                                             max_angular_speed=.001,
                                                             stop_when_target_pose_reached=False,
                                                             make_final_correction=False,
                                                             timeout=3.,
                                                             duration=.5,
                                                             log=False)

                        # Waiting until pose tracking begins
                        stime = time.time()

                        force_felt = self.franka.get_eef_wrench()
                        t_forces.append(copy.deepcopy(force_felt))
                        c_pose = self.franka.get_eef_pose()
                        t_actions.append(copy.deepcopy(temp_pose))
                        temp_img = self.franka.camera.get_rgb()  # Querying before processing to ensure data is in sync
                        t_images.append(self.process_img(temp_img).cpu().numpy())
                        # t_images.append(temp_img.cpu().numpy())
                        t_identifiers.append([demo_index])
                        gripper_states.append(self.gripper_states[demo_index])
                        if np.abs(pose_reached[0, 3] - c_pose[0, 3]) < .001 or np.abs(
                                pose_reached[1, 3] - c_pose[1, 3]) < .001 or np.abs(
                            pose_reached[2, 3] - c_pose[2, 3]) < .001:
                            t_is_on_waypoint.append([1])
                        else:
                            t_is_on_waypoint.append([0])

                        if np.abs(force_felt[0]) > 50 \
                                or np.abs(force_felt[1]) > 50 \
                                or np.abs(force_felt[2]) > 50 \
                                or np.abs(force_felt[3]) > 20 \
                                or np.abs(force_felt[4]) > 20 \
                                or np.abs(force_felt[5]) > 20:
                            print("Force threshold exceeded. Skipping sample")
                            self.franka.reset_async_tracking()
                            time.sleep(.1)
                        data_collected += 1
                        number_of_waypoint_data_collected += 1
                        sleep_time = np.max((1 / self._recording_rate - (time.time() - stime), 0))
                        time.sleep(sleep_time)  # Sync loop to self._recording_rate Hz as recording
                    tracked_poses = True
                # time.sleep(2)
                print(bcolors.WARNING + "[ERROR] linear: {}, angular: {}".format(
                    self.franka.get_eef_pose()[:3, 3] - current_demo_pose[:3, 3],
                    self.franka.se3_transforms.rot2euler("xyz", self.franka.get_eef_pose()[:3, :3], degrees=True) -
                    self.franka.se3_transforms.rot2euler("xyz", current_demo_pose[:3, :3], degrees=True)))

                achieved_pose = self.franka.get_eef_pose()

                print(bcolors.WARNING + "[ERROR ACHIEVED POSE] linear: {}, angular: {}".format(
                    achieved_pose[:3, 3] - pose_reached[:3, 3],
                    self.franka.se3_transforms.rot2euler("xyz",achieved_pose[:3, :3], degrees=True) -
                    self.franka.se3_transforms.rot2euler("xyz", pose_reached[:3, :3], degrees=True)))

                achieved_rot = self.franka.se3_transforms.rot2euler("xyz", achieved_pose[:3, :3], degrees=True)
                reached_rot = self.franka.se3_transforms.rot2euler("xyz", pose_reached[:3, :3], degrees=True)
                distance_errors = np.array([np.abs(achieved_pose[0, 3] - pose_reached[0, 3]), np.abs(
                    achieved_pose[1, 3] - pose_reached[1, 3]), np.abs(
                    achieved_pose[2, 3] - pose_reached[2, 3])])
                rotation_errors = np.array([np.abs(achieved_rot[0] - reached_rot[0]), np.abs(
                    achieved_rot[1] - reached_rot[1]), np.abs(
                    achieved_rot[2] - reached_rot[2])])
                if (np.abs(achieved_pose[0, 3] - pose_reached[0, 3]) > .001 or np.abs(
                        achieved_pose[1, 3] - pose_reached[1, 3]) > .001 or np.abs(
                    achieved_pose[2, 3] - pose_reached[2, 3]) > .00 or np.abs(achieved_rot[0] - reached_rot[0]) > 1
                        or np.abs(achieved_rot[1] - reached_rot[1]) > 1 or np.abs(achieved_rot[2] - reached_rot[2]) > 1):
                    print(
                        "Could not accurately return back to waypoint. Skipping sample. Returning back to starting pose and will move to waypoint again.")
                    self.franka.go_to_pose(self.starting_pose)
                    self.franka.replay_demonstration(replay_rate=self._recording_rate,
                                                                 demonstration_path=self.demonstration_dir,
                                                                 task_name=self.task_name,
                                                                 until_index=demo_index + 1,
                                                                 frame='base')
                    self.franka.go_to_pose(self.demonstration_in_base[demo_index])
                    self.franka.reset_async_tracking()
                    t_images = []
                    t_forces = []
                    t_identifiers = []
                    t_actions = []
                    t_is_on_waypoint = []
                    gripper_states = []
                    continue
                # input("Next trajectory")
                t_forces = np.array(t_forces)
                t_images = np.array(t_images)
                t_identifiers = np.array(t_identifiers)
                t_actions = np.array(t_actions)
                t_is_on_waypoint = np.array(t_is_on_waypoint)
                gripper_states = np.array(gripper_states)

                forces_torch = torch.from_numpy(t_forces).float()
                images_torch = torch.from_numpy(t_images).float()
                identifiers_torch = torch.from_numpy(t_identifiers).float()
                actions_torch = torch.from_numpy(t_actions).float()
                is_on_waypoint_torch = torch.from_numpy(t_is_on_waypoint).float()
                gripper_states_torch = torch.from_numpy(gripper_states).float()
                # Save the above data
                path = "{}/tasks/{}/data/closed_loop/trajectory_sample_{}.pt".format(ASSETS_DIR, self.task_name,
                                                                                     trajectories_collected)

                data = {"data_imgs": images_torch,
                        "data_forces": forces_torch,
                        "data_actions": actions_torch,
                        "in_trajectory_identifier": identifiers_torch,
                        "gripper_state": gripper_states_torch,
                        "is_on_waypoint": is_on_waypoint_torch}
                torch.save(data, path)

                trajectories_collected += 1
                traj_num += 1
                data_collection_details = {"number_of_trajectories_collected": trajectories_collected,
                                           "number_of_trajectories_per_way_point": trajectories_per_way_point,
                                           "demo_index": demo_index}
                path_2 = "{}/tasks/{}/data/closed_loop/data_collection_details.pt".format(ASSETS_DIR, self.task_name)
                torch.save(data_collection_details, path_2)

                print("Collected {} samples. Trajectories for this waypoint: {}. Demo index: {}. Time take: {}".format(
                    data_collected, traj_num, demo_index,
                    time.time() - loop_time))
            demo_index += 1  # After collecting all the trajectories for the current waypoint, move to the next.
            trajectories_per_way_point = 10
        self.franka.go_to_pose(self.starting_pose)
    

if __name__ == '__main__':
    data_collector = DataCollector(task_name="test",
                                   data_collection_recording_rate=10,
                                   reset_task=True,
                                   validate_demo=True,
                                   resume_data_collection=False,
                                   collect_last_inch=False,
                                   close_gripper_at_start=False ,
                                   open_gripper_at_start=True  ,
                                   nearest_neighbour_data_collection=False,
                                   shorten_demo=False)
    data_collector.collect_data()
    data_collector.save_data()
