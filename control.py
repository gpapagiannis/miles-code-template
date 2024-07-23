#!/usr/bin/env python3
import random
import time
from franka_ros_controller.frankarosjointcontroller import FrankaROSJointController
from closed_loop.utils import *
import pickle as pkl
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
from torchvision.transforms import CenterCrop, Resize
from models import LSTMPolicy
import franka_controller.se3_tools as se3
from policy_trainer import PolicyTrainer, InteractionDataset
import matplotlib.pyplot as plt
import cv2


TASKS = ["key_insert_twist", "plug_in_socket", "usb_insertion", "insert_power_cable", "bread_in_toaster", "open_lid", "screwdiver", "bin_generalization"]
TASK_NAME_DEMO = TASKS[4]
ASSETS_DIR = "/path/to/assets/assets"
DEFAULT_K_GAINS = [600.0, 600.0, 600.0, 600.0, 150.0, 100.0, 20.0]
DEFAULT_K_GAINS_SAFETY = [30.0, 30.0, 30.0, 30.0, 12.0, 7.0, 2.0]
DAMPING_RATIO = 1.
ABLATION=False
ABLATION_SPLIT=1.0
MAX_LIN_DEMO = .0005
MAX_ANG_DEMO = .005
CONTROL_RATE_DEMO = 1
TIMEOUT_FOR_REPLAY = 20
RATE_AFTER_SEQ = 0.01

class ClosedLoopControl:
    def __init__(self,
                 task_name='test',
                 close_gripper_at_start=True,
                 open_gripper_at_start=True,
                 data_collection_recording_rate=10,
                 validate_demo=False,
                 last_inch=False,
                 action_horizon=6,
                 coarse_approach=False,
                 template_matching=False,
                 eval_training_sample=False,
                 reset_on_init=True,
                 denormalize=True,
                 vision_only=False,
                 force_only=False,
                 short_sequences=False,
                 random_starts=False,
                 nearest_neighbor=False,
                 lstm_nearest_neighbor=False,
                 use_LSTM=True,
                 ablation=False,
                 c2f=False,
                 reset_gripper=False,
                 ablation_split=1.0,
                 ):

        self.action_horizon = action_horizon
        self.coarse_approach = coarse_approach
        self.close_gripper_at_start = close_gripper_at_start
        self.open_gripper_at_start = open_gripper_at_start
        self.denormalize = denormalize
        self.force_only = force_only
        self.vision_only = vision_only
        self.short_sequences = short_sequences
        self.random_starts = random_starts
        self.nearest_neighbor = nearest_neighbor
        self.lstm_nearest_neighbor = lstm_nearest_neighbor
        self.average_execution_time = 0
        self.use_LSTM = use_LSTM
        self.ablation = ablation
        self.ablation_split = ablation_split
        self.c2f = c2f
        if self.ablation:
            self.action_horizon = 5
            self.vision_only = False
            self.force_only = False
            self.use_LSTM = True
            assert self.ablation_split < 1.0
        # set up cuda
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = "cpu"
        self.task_name = task_name
        self.last_inch = last_inch

        self.franka = FrankaROSJointController(control_frequency=250,
                                               use_camera=True,
                                               record_camera_images=True,
                                               reset_on_init=reset_on_init,
                                               camera_name='wrist_camera')

        self._recording_rate = data_collection_recording_rate
        if reset_gripper:
            self.franka.home_gripper();input(); self.franka.grasp()
        if open_gripper_at_start:
            input("Press enter to open gripper")
            self.franka.open_gripper()
        if close_gripper_at_start:
            if self.franka.is_gripper_open():
                input("Press enter to close gripper")

                self.franka.grasp()

        demonstration_dir = "{}/tasks/{}/demonstration".format(ASSETS_DIR, task_name)
        demonstration_path = "{}/tasks/{}/demonstration/recorded_demo.pkl".format(ASSETS_DIR, task_name)
        starting_pose_path = "{}/tasks/{}/demonstration/starting_pose.pkl".format(ASSETS_DIR, task_name)

        self.demonstration = pkl.load(open(demonstration_path, 'rb'))
        self.starting_pose = pkl.load(open(starting_pose_path, 'rb'))
        self.demonstration_in_base = copy.deepcopy(self.convert_demo_trajectory_to_base_frame())
        data_folder = "closed_loop"
        if lstm_nearest_neighbor:
            data_folder = "NN"
        self.train_dataset = InteractionDataset(
            path="{}/tasks/{}/data/{}/".format(ASSETS_DIR, self.task_name, data_folder),
            demonstration_path="{}/tasks/{}/demonstration".format(ASSETS_DIR, self.task_name),
            horizon=action_horizon,
            action_multiplier=6,
            subsample_frequency=1,
            mix_trajectories=False,
            normalize_actions=False)

        if True:

            self.collected_data_details = torch.load(
                "{}/tasks/{}/data/{}/data_collection_details.pt".format(ASSETS_DIR, task_name, data_folder))

            self.last_demo_index_collected =self.collected_data_details['demo_index']
            if self.c2f:
                self.last_demo_index_collected = 0
            print("Last demo index collected: {}".format(self.collected_data_details['demo_index']))

            self.control_model = LSTMPolicy(action_dim=3 * self.action_horizon, with_gaussian_output=False,
                                            vision_only=self.vision_only, force_only=self.force_only,
                                            train_classifier=False).to(self.device)
            self.control_model_ori = LSTMPolicy(action_dim=3 * self.action_horizon, with_gaussian_output=False,
                                                vision_only=self.vision_only, force_only=self.force_only,
                                                train_classifier=False).to(self.device)


            extension = ""
            if self.vision_only:
                extension += "_vision_only"
            elif self.force_only:
                extension += "_force_only"

            if short_sequences:
                extension += "_short_sequences"
            if random_starts:
                extension += "_random_starts"
            # extension2 = ""
            if c2f:
                extension += "_C2F_NN"
            if nearest_neighbor:
                extension += "_NN"
            if lstm_nearest_neighbor:
                extension += "_lstm_NN"

            lstm_extension=""
            if not self.use_LSTM:
                lstm_extension = "no_"

            if not self.ablation:
                self.control_model.load_state_dict(
                    torch.load("{}/tasks/{}/models/closed_loop/policy_{}lstm_seq_{}_lin{}.pt".format(ASSETS_DIR, task_name, lstm_extension, action_horizon, extension)))

                self.control_model_ori.load_state_dict(
                    torch.load("{}/tasks/{}/models/closed_loop/policy_{}lstm_seq_{}_ori{}.pt".format(ASSETS_DIR, task_name, lstm_extension, action_horizon, extension))) # if it can't find path it's because you may have changed something in this line
                print("LSTM path:  ", "{}/tasks/{}/models/closed_loop/policy_{}lstm_seq_{}_lin{}.pt".format(ASSETS_DIR, task_name, lstm_extension, action_horizon, extension))
            else:
                print("------------LOADING NETWORKS FOR ABLATION-------------------")
                self.control_model.load_state_dict(
                    torch.load("{}/tasks/{}/models/closed_loop/{}/policy_{}lstm_seq__lin_split_{}_ablation.pt".format(ASSETS_DIR,
                                                                                                                        task_name,
                                                                                                                        int(100 * self.ablation_split),
                                                                                                                        action_horizon,
                                                                                                                        self.ablation_split)))

                self.control_model_ori.load_state_dict(
                    torch.load("{}/tasks/{}/models/closed_loop/{}/policy_{}lstm_seq__ori_split_{}_ablation.pt".format(ASSETS_DIR,
                                                                                                                           task_name,
                                                                                                                           int(100 * self.ablation_split),
                                                                                                                           action_horizon,
                                                                                                                           self.ablation_split)))

            if not self.nearest_neighbor or not self.lstm_nearest_neighbor:
                extension = ""
            if not self.use_LSTM:
                extension = "_no_LSTM"
            if self.c2f:
                extension = "_C2F"

            if not self.ablation:
                norm_constants_lin = torch.load(
                    "{}/tasks/{}/data/{}/normalization_constants_lin_{}{}.pt".format(ASSETS_DIR, task_name, data_folder,
                                                                                            action_horizon, extension))
                print("Norm path",  "{}/tasks/{}/data/{}/normalization_constants_lin_{}{}.pt".format(ASSETS_DIR, task_name, data_folder,
                                                                                            action_horizon, extension))
                input()
            else:
                norm_constants_lin = torch.load(
                    "{}/tasks/{}/models/{}/{}/normalization_constants_lin_{}_split_{}_ablation.pt".format(ASSETS_DIR,
                                                                                                          task_name,
                                                                                                          data_folder,
                                                                                                          int(100 * self.ablation_split),
                                                                                                            action_horizon,
                                                                                                          int(100 * self.ablation_split)))
            self.min_norm_consts_lin, self.max_norm_consts_lin = norm_constants_lin['min_norm'], norm_constants_lin[
                'max_norm']
            if not self.ablation:
                norm_constants_ang = torch.load("{}/tasks/{}/data/{}/normalization_constants_ang_{}{}.pt".format(ASSETS_DIR, task_name, data_folder,
                                                                                        action_horizon, extension))
            else:
                norm_constants_ang = torch.load(
                    "{}/tasks/{}/models/{}/{}/normalization_constants_ang_{}_split_{}_ablation.pt".format(ASSETS_DIR, task_name, data_folder,
                                                                                                          int(100 * self.ablation_split),
                                                                                                action_horizon, int(100 * self.ablation_split)))
            self.min_norm_consts_ang, self.max_norm_consts_ang = norm_constants_ang['min_norm'], norm_constants_ang[
                'max_norm']
            self.control_model.eval()
            self.control_model_ori.eval()
            self.target_rnd.eval()
            self.predictor_rnd.eval()

           
            if validate_demo:
                self.validate_trajectory(path=demonstration_dir)

    def denormalize_actions(self, data, denormalize_ori=False):

        """Denormalize the data in the range of [-1,+1] based on the min and max normalization constants."""
        if not self.denormalize:# or denormalize_ori:
            return data
        if denormalize_ori:
            min = self.min_norm_consts_ang
            max = self.max_norm_consts_ang

        else:
            # Denormalize every linear action separately
            min = self.min_norm_consts_lin
            max = self.max_norm_consts_lin
        data[0::3] = (data[0::3] + 1) * (max[0] - min[0]) / 2 + min[0]
        data[1::3] = (data[1::3] + 1) * (max[1] - min[1]) / 2 + min[1]
        data[2::3] = (data[2::3] + 1) * (max[2] - min[2]) / 2 + min[2]
        return data

    def validate_trajectory(self, path):
        self.franka.go_to_pose_in_base_kdl(self.starting_pose)
        self.franka.replay_demonstration(replay_rate=self._recording_rate,
                                                            is_teleoperated=True,
                                                            in_joint_space=False,
                                                            task_name=self.task_name,
                                                            demonstration_path=path)

        input()

    def convert_demo_trajectory_to_base_frame(self):
        demo_in_base = []
        for i, pose in enumerate(self.demonstration):
            demo_in_base.append(self.starting_pose @ pose)
        return demo_in_base

    def process_img(self, img):
        center_crop = CenterCrop(480)
        img_resize = Resize((128, 128))
        if img.shape[0] == img.shape[1]:
            img = img_resize(torch.tensor(img).permute(2, 0, 1) / 255)
        else:
            img = center_crop(torch.tensor(img).permute(2, 0, 1) / 255)
            img = img_resize(img)
        return img

    def get_random_transformation(self, linear_range=None, angular_range=None, coarse=False):

        if coarse:
            range_l = .1
            rannge_o = 30
        else:
            range_l = .02
            rannge_o = 2
        if linear_range is None:
            linear_range = [-range_l, range_l]
        if angular_range is None:
            angular_range = [-rannge_o, rannge_o]

        angular_range[0] = np.deg2rad(angular_range[0])
        angular_range[1] = np.deg2rad(angular_range[1])

        linear = np.random.uniform(low=linear_range[0], high=linear_range[1], size=3)
        angular = np.random.uniform(low=angular_range[0], high=angular_range[1], size=3)
        #
        if self.coarse_approach:
            angular[0] = .0
            angular[1] = .0

        if coarse:
            linear[2] = .0
            angular[0] = .0
            angular[1] = .0


        transformation = np.eye(4)
        transformation[:3, 3] = linear
        transformation[:3, :3] = self.franka.se3_transforms.euler2rot("xyz", angular, degrees=False)
        print("Transformation linear: {}".format(linear))
        print("Transformation orientation: {}".format(self.franka.se3_transforms.rot2euler("xyz", transformation[:3, :3], degrees=True)), "Angular range: {}".format(angular_range))

        return transformation

    def pos_eval_metrics(self, pred, target):
        pos_error = torch.abs(torch.add(pred[:, :, :3 * self.action_horizon], -target[:, :, :3 * self.action_horizon]))
        print("pos_error", pos_error)
        x_error = None
        for i in range(self.action_horizon):
            if x_error is None:
                x_error = pos_error[:, :, 3 * i]
            else:
                x_error = torch.add(x_error, pos_error[:, :, 3 * i])

        x_error = torch.div(x_error, self.action_horizon)

        y_error = None
        for i in range(self.action_horizon):
            if y_error is None:
                y_error = pos_error[:, :, 3 * i + 1]
            else:
                y_error = torch.add(y_error, pos_error[:, :, 3 * i + 1])

        y_error = torch.div(y_error, self.action_horizon)

        z_error = None
        for i in range(self.action_horizon):
            if z_error is None:
                z_error = pos_error[:, :, 3 * i + 2]
            else:
                z_error = torch.add(z_error, pos_error[:, :, 3 * i + 2])

        z_error = torch.div(z_error, self.action_horizon)
        return torch.mean(x_error), torch.mean(y_error), torch.mean(z_error)
    def sample_dataset_initial_pose(self, idx=None):
        if idx is None:
            sample = self.train_dataset.__getitem__(np.random.randint(0, 200))
        else:
            sample = self.train_dataset.__getitem__(idx)
        prop = sample['actions_unprocessed']
        return prop[0].cpu().numpy(), sample['actions'], sample
    
    def perform_task(self, idx=None, auto_reset=False, load_poses_file=False):
        self.perform_task_closed_loop(idx=idx, auto_reset=auto_reset, load_poses_file=load_poses_file)

    def evaluate_training_samples(self, index=0):
        trainer = PolicyTrainer(task_name="test",
                                action_horizon=1,
                                predict_only_displacement=False,
                                subsample_frequency=3,
                                with_gaussian_output=False,
                                train_classifier=False,
                                mix_trajectories=False,
                                train_ori=True,
                                normalize_actions=False)
        sample = trainer.train_dataset.__getitem__(index)
        return sample
    
    @torch.no_grad()
    def perform_task_closed_loop(self, idx=None,  auto_reset=False, load_poses_file=False):

        
        # probably this will follow DINOBot's reach to bottleneck
        input("Press enter to start control")

        identifier = 0
        hidden_state = None
        hidden_state_ori = None
        submm_actions_predicted = 0
        consecutive_resets = 0
        init_pose = copy.deepcopy(self.franka.get_eef_pose())
        stime = time.time()

        for timestep in range(200):
            torch.cuda.empty_cache()
            current_eef_pose = copy.deepcopy(self.franka.get_eef_pose())
            current_img = self.franka.camera.get_rgb(as_tensor=True).to(self.device)
            current_img = self.process_img(current_img)
            current_force = torch.from_numpy(self.franka.get_eef_wrench()).float().to(self.device)
            action_pred, hidden_state = self.control_model.forward_step(current_img.unsqueeze(0).unsqueeze(0),
                                                                   current_force.unsqueeze(0).unsqueeze(0),
                                                                   hidden_state)

            action_ori, hidden_state_ori = self.control_model_ori.forward_step(current_img.unsqueeze(0).unsqueeze(0),
                                                                               current_force.unsqueeze(0).unsqueeze(0),
                                                                               hidden_state_ori)

            action_pred[0, 0, :] = self.denormalize_actions(action_pred[0, 0, :3 * self.action_horizon])
            action_ori[0, 0, :3 * self.action_horizon] = self.denormalize_actions(
                action_ori[0, 0, :3 * self.action_horizon],
                denormalize_ori=True)

            action = torch.cat((action_pred, action_ori), dim=2)
            action = action.detach().cpu().numpy()[0]
            actions_in_eef = []
            for k in range(self.action_horizon):
                action_in_eef_0 = np.eye(4)
                action_in_eef_0[:3, 3] = action[0, k * 3:k * 3 + 3]
                action_in_eef_0[:3, :3] = se3.euler2rot("xyz", action[0,
                                                               3 * self.action_horizon + k * 3:3 * self.action_horizon + k * 3 + 3],
                                                        degrees=False)
                action_in_eef_0 = np.asarray(action_in_eef_0)
                actions_in_eef.append(action_in_eef_0)

            MAX_LIN = MAX_LIN_DEMO
            MAX_ANG = MAX_ANG_DEMO
            print("-------------------")
            last_action = None
            for idx, action in enumerate(actions_in_eef):
                next_pose = current_eef_pose @ action
                last_action = action
                print(action[:3, 3], se3.rot2euler("xyz", action[:3, :3], degrees=True), submm_actions_predicted)
                if np.abs(action[0, 3]) < 0.001 and np.abs(action[1, 3]) < 0.001 and np.abs(action[2, 3]) < 0.001:
                    submm_actions_predicted += 1
                elif submm_actions_predicted < 15:
                    submm_actions_predicted = 0
                self.franka.go_to_pose_in_base_async(next_pose,
                                                     max_linear_speed=MAX_LIN,
                                                     max_angular_speed=MAX_ANG,
                                                     stop_when_target_pose_reached=False,
                                                     make_final_correction=False,
                                                     timeout=3.1,
                                                     duration=1,
                                                     log=False)

                sleep_time = np.max((1 / self._recording_rate - (time.time() - stime), 0))

                current_img = self.franka.camera.get_rgb(as_tensor=True).to(self.device)
                current_force = torch.from_numpy(self.franka.get_eef_wrench()).float().to(self.device)
                current_img = self.process_img(current_img)
                ood_image = copy.deepcopy(current_img)
                force_felt = self.franka.get_eef_wrench()
                time.sleep(CONTROL_RATE_DEMO/self._recording_rate)

                if np.abs(force_felt[0]) > 50 \
                        or np.abs(force_felt[1]) > 50 \
                        or np.abs(force_felt[2]) > 50 \
                        or np.abs(force_felt[3]) > 20 \
                        or np.abs(force_felt[4]) > 20 \
                        or np.abs(force_felt[5]) > 20:
                    print("Force exceeded exiting experiment")
                    break
                if not self.short_sequences:
                    _, hidden_state = self.control_model.forward_step(current_img.unsqueeze(0).unsqueeze(0),
                                                                      current_force.unsqueeze(0).unsqueeze(0),
                                                                      hidden_state)
                    _, hidden_state_ori = self.control_model.forward_step(current_img.unsqueeze(0).unsqueeze(0),
                                                                          current_force.unsqueeze(0).unsqueeze(0),
                                                                          hidden_state_ori)
         
            time.sleep(RATE_AFTER_SEQ)
            if submm_actions_predicted >= 9:#if network converged replay
                print("Submm actions predicted: {}".format(submm_actions_predicted))
                break
            force_felt = self.franka.get_eef_wrench()
            if np.abs(force_felt[0]) > 50 \
                    or np.abs(force_felt[1]) > 50 \
                    or np.abs(force_felt[2]) > 50 \
                    or np.abs(force_felt[3]) > 20 \
                    or np.abs(force_felt[4]) > 20 \
                    or np.abs(force_felt[5]) > 20:
                print("Force exceeded exiting experiment")
                break
            if time.time() - stime > 1 * TIMEOUT_FOR_REPLAY:
                break
 

        self.franka.reset_async_tracking()
        demonstration_dir = "{}/tasks/{}/demonstration".format(ASSETS_DIR, self.task_name)
        self.franka.replay_demonstration(replay_rate=self._recording_rate,
                                                     task_name=self.task_name,
                                                     from_index=self.last_demo_index_collected,
                                                     init_pose=None,
                                                     demonstration_path=demonstration_dir)

            
  

if __name__ == '__main__':
    closed_loop_control = ClosedLoopControl(task_name=TASK_NAME_DEMO,
                                            validate_demo=False,
                                            last_inch=False,
                                            vision_only=False, 
                                            force_only=False,
                                            action_horizon=5,
                                            eval_training_sample=False,
                                            coarse_approach=False,
                                            open_gripper_at_start=True,
                                            template_matching=False,
                                            close_gripper_at_start=False,
                                            reset_gripper=False ,
                                            denormalize=True,
                                            reset_on_init=True,
                                            short_sequences=False,
                                            random_starts=False,
                                            nearest_neighbor=False,
                                            lstm_nearest_neighbor=False,
                                            use_LSTM=True,
                                            ablation=False,
                                            c2f=False,
                                            ablation_split=1,
                                            data_collection_recording_rate=10)

    closed_loop_control.perform_task()
