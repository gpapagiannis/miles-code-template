* This folder contains the code template to run MILES. MILES relies on an impedance controller and as a result the values used, e.g., gains, damping, control rate, force thresholds, etc... are hardware dependent. They were tuned for a Franka Emika robot, but from my experience running this on two different Frankas even if the robots are the same it is likely that some tuning is neccessary.

* You will note that there is a class called FrankaROSJointController. This class is not included, as it is robot hardware specific, but should be straighforward to implement. It is used to control the robot. The important part of this class is a function named: go_to_pose_in_base_async. That function moves the robot asynchronously while the rest of the script collects data. There are a few parts in the code specific to our class robotics controller, but it should be clear what they do.

* You will also note that there are variables, that likely do not seem relevant to MILES (and are not used). This is because the original code was significantly longer as it included code for baselines and other studies we ran before coming up with the final method. I have tried to clean almost all of this up. 

* There is a file called find_sim.py included which should be put in the folder of the repository https://github.com/ShirAmir/dino-vit-features?tab=readme-ov-file . It is used for the disturbance detection using the DINO network. 

* You will likely also need to adapt the code that communicates with the wrist camera of the robot.

* Overall, I would recommend reading the code provided in here along with the pseudocode in the appendix of the paper section C. MILES is very simple to implement, the tricky part should be tuning it to ones hardware. 



