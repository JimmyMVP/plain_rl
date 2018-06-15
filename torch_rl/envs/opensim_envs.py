import math
import numpy as np
import os
from osim.env.utils.mygym import convert_to_gym
import gym
import opensim
import random
import osim

## OpenSim interface
# The amin purpose of this class is to provide wrap all 
# the necessery elements of OpenSim in one place
# The actual RL environment then only needs to:
# - open a model
# - actuate
# - integrate
# - read the high level description of the state
# The objective, stop condition, and other gym-related
# methods are enclosed in the OsimEnv class
class OsimModel(object):
    # Initialize simulation
    stepsize = 0.05

    model = None
    state = None
    state0 = None
    joints = []
    bodies = []
    brain = None
    verbose = False
    istep = 0
    
    state_desc_istep = None
    prev_state_desc = None
    state_desc = None
    integrator_accuracy = None

    maxforces = []
    curforces = []

    def __init__(self, model_path, visualize, integrator_accuracy = 5e-5):
        self.integrator_accuracy = integrator_accuracy
        self.model = opensim.Model(model_path)
        self.model.initSystem()
        self.brain = opensim.PrescribedController()

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.markerSet = self.model.getMarkerSet()
        self.contactGeometrySet = self.model.getContactGeometrySet()

        if self.verbose:
            self.list_elements()

        # Add actuators as constant functions. Then, during simulations
        # we will change levels of constants.
        # One actuartor per each muscle
        for j in range(self.muscleSet.getSize()):
            func = opensim.Constant(1.0)
            self.brain.addActuator(self.muscleSet.get(j))
            self.brain.prescribeControlForActuator(j, func)

            self.maxforces.append(self.muscleSet.get(j).getMaxIsometricForce())
            self.curforces.append(1.0)

        self.noutput = self.muscleSet.getSize()
            
        self.model.addController(self.brain)
        self.model.initSystem()

    def list_elements(self):
        print("JOINTS")
        for i in range(self.jointSet.getSize()):
            print(i,self.jointSet.get(i).getName())
        print("\nBODIES")
        for i in range(self.bodySet.getSize()):
            print(i,self.bodySet.get(i).getName())
        print("\nMUSCLES")
        for i in range(self.muscleSet.getSize()):
            print(i,self.muscleSet.get(i).getName())
        print("\nFORCES")
        for i in range(self.forceSet.getSize()):
            print(i,self.forceSet.get(i).getName())
        print("\nMARKERS")
        for i in range(self.markerSet.getSize()):
            print(i,self.markerSet.get(i).getName())
        print("")

    def actuate(self, action):
        if np.any(np.isnan(action)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")

        # TODO: Check if actions within [0,1]
        self.last_action = action
            
        brain = opensim.PrescribedController.safeDownCast(self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = opensim.Constant.safeDownCast(functionSet.get(j))
            func.setValue( float(action[j]) )

    """
    Directly modifies activations in the current state.
    """
    def set_activations(self, activations):
        if np.any(np.isnan(activations)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")
        for j in range(self.muscleSet.getSize()):
            self.muscleSet.get(j).setActivation(self.state, activations[j])

    """
    Get activations in the given state.
    """
    def get_activations(self):
        return [self.muscleSet.get(j).getActivation(self.state) for j in range(self.muscleSet.getSize())]

    def compute_state_desc(self):
        self.model.realizeAcceleration(self.state)

        res = {}

        ## Joints
        res["joint_pos"] = {}
        res["joint_vel"] = {}
        res["joint_acc"] = {}
        for i in range(self.jointSet.getSize()):
            joint = self.jointSet.get(i)
            name = joint.getName()
            res["joint_pos"][name] = [joint.get_coordinates(i).getValue(self.state) for i in range(joint.numCoordinates())]
            res["joint_vel"][name] = [joint.get_coordinates(i).getSpeedValue(self.state) for i in range(joint.numCoordinates())]
            res["joint_acc"][name] = [joint.get_coordinates(i).getAccelerationValue(self.state) for i in range(joint.numCoordinates())]

        ## Bodies
        res["body_pos"] = {}
        res["body_vel"] = {}
        res["body_acc"] = {}
        res["body_pos_rot"] = {}
        res["body_vel_rot"] = {}
        res["body_acc_rot"] = {}
        for i in range(self.bodySet.getSize()):
            body = self.bodySet.get(i)
            name = body.getName()
            res["body_pos"][name] = [body.getTransformInGround(self.state).p()[i] for i in range(3)]
            res["body_vel"][name] = [body.getVelocityInGround(self.state).get(1).get(i) for i in range(3)]
            res["body_acc"][name] = [body.getAccelerationInGround(self.state).get(1).get(i) for i in range(3)]
            
            res["body_pos_rot"][name] = [body.getTransformInGround(self.state).R().convertRotationToBodyFixedXYZ().get(i) for i in range(3)]
            res["body_vel_rot"][name] = [body.getVelocityInGround(self.state).get(0).get(i) for i in range(3)]
            res["body_acc_rot"][name] = [body.getAccelerationInGround(self.state).get(0).get(i) for i in range(3)]

        ## Forces
        res["forces"] = {}
        for i in range(self.forceSet.getSize()):
            force = self.forceSet.get(i)
            name = force.getName()
            values = force.getRecordValues(self.state)
            res["forces"][name] = [values.get(i) for i in range(values.size())]

        ## Muscles
        res["muscles"] = {}
        for i in range(self.muscleSet.getSize()):
            muscle = self.muscleSet.get(i)
            name = muscle.getName()
            res["muscles"][name] = {}
            res["muscles"][name]["activation"] = muscle.getActivation(self.state)
            res["muscles"][name]["fiber_length"] = muscle.getFiberLength(self.state)
            res["muscles"][name]["fiber_velocity"] = muscle.getFiberVelocity(self.state)
            res["muscles"][name]["fiber_force"] = muscle.getFiberForce(self.state)
            # We can get more properties from here http://myosin.sourceforge.net/2125/classOpenSim_1_1Muscle.html 
        
        ## Markers
        res["markers"] = {}
        for i in range(self.markerSet.getSize()):
            marker = self.markerSet.get(i)
            name = marker.getName()
            res["markers"][name] = {}
            res["markers"][name]["pos"] = [marker.getLocationInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["vel"] = [marker.getVelocityInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["acc"] = [marker.getAccelerationInGround(self.state)[i] for i in range(3)]

        ## Other
        res["misc"] = {}
        res["misc"]["mass_center_pos"] = [self.model.calcMassCenterPosition(self.state)[i] for i in range(2)]
        res["misc"]["mass_center_vel"] = [self.model.calcMassCenterVelocity(self.state)[i] for i in range(2)]
        res["misc"]["mass_center_acc"] = [self.model.calcMassCenterAcceleration(self.state)[i] for i in range(2)]

        return res

    def get_state_desc(self):
        if self.state_desc_istep != self.istep:
            self.prev_state_desc = self.state_desc
            self.state_desc = self.compute_state_desc()
            self.state_desc_istep = self.istep
        return self.state_desc

    def set_strength(self, strength):
        self.curforces = strength
        for i in range(len(self.curforces)):
            self.muscleSet.get(i).setMaxIsometricForce(self.curforces[i] * self.maxforces[i])

    def get_body(self, name):
        return self.bodySet.get(name)

    def get_joint(self, name):
        return self.jointSet.get(name)

    def get_muscle(self, name):
        return self.muscleSet.get(name)

    def get_marker(self, name):
        return self.markerSet.get(name)

    def get_contact_geometry(self, name):
        return self.contactGeometrySet.get(name)

    def get_force(self, name):
        return self.forceSet.get(name)

    def get_action_space_size(self):
        return self.noutput

    def set_integrator_accuracy(self, integrator_accuracy):
        self.integrator_accuracy = integrator_accuracy

    def reset_manager(self):
        self.manager = opensim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def reset(self):
        self.state = self.model.initializeState()
        self.state.setTime(0)
        self.istep = 0

        self.reset_manager()

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state
        self.reset_manager()

    def integrate(self):
        # Define the new endtime of the simulation
        self.istep = self.istep + 1

        # Integrate till the new endtime
        try:
            self.state = self.manager.integrate(self.stepsize * self.istep)
        except Exception as e:
            print (e)


class Spec(object):
    def __init__(self, *args, **kwargs):
        self.id = 0
        self.timestep_limit = 1000

## OpenAI interface
# The amin purpose of this class is to provide wrap all 
# the functions of OpenAI gym. It is still an abstract
# class but closer to OpenSim. The actual classes of
# environments inherit from this one and:
# - select the model file
# - define the rewards and stopping conditions
# - define an obsernvation as a function of state
class OsimEnv(gym.Env):
    action_space = None
    observation_space = None
    osim_model = None
    istep = 0
    verbose = False

    visualize = False
    spec = None
    time_limit = 1e10

    prev_state_desc = None

    model_path = None # os.path.join(os.path.dirname(__file__), '../models/MODEL_NAME.osim')    

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : None
    }

    def reward(self):
        raise NotImplementedError

    def is_done(self):
        return False

    def __init__(self, visualize = True, integrator_accuracy = 5e-5):
        self.osim_model = OsimModel(self.model_path, visualize, integrator_accuracy = integrator_accuracy)

        # Create specs, action and observation spaces mocks for compatibility with OpenAI gym
        self.spec = Spec()
        self.spec.timestep_limit = self.time_limit

        if not self.action_space:
            self.action_space = ( [0.0] * self.osim_model.get_action_space_size(), [1.0] * self.osim_model.get_action_space_size() )
        if not self.observation_space:
#            self.observation_space = ( [-math.pi*100] * self.get_observation_space_size(), [math.pi*100] * self.get_observation_space_s
            self.observation_space = ( [0] * self.get_observation_space_size(), [0] * self.get_observation_space_size() )
        self.action_space = convert_to_gym(self.action_space)
        self.observation_space = convert_to_gym(self.observation_space)

    def get_state_desc(self):
        return self.osim_model.get_state_desc()

    def get_prev_state_desc(self):
        return self.prev_state_desc

    def get_observation(self):
        # This one will normally be overwrtitten by the environments
        # In particular, for the gym we want a vector and not a dictionary
        return self.osim_model.get_state_desc()

    def get_observation_space_size(self):
        return 0

    def get_action_space_size(self):
        return self.osim_model.get_action_space_size()

    def reset(self, project = True):
        self.osim_model.reset()
        
        if not project:
            return self.get_state_desc()
        return self.get_observation()

    def step(self, action, project = True):
        self.prev_state_desc = self.get_state_desc()        
        self.osim_model.actuate(action)
        self.osim_model.integrate()

        if project:
            obs = self.get_observation()
        else:
            obs = self.get_state_desc()
            
        return [ obs, self.reward(), self.is_done() or (self.osim_model.istep >= self.spec.timestep_limit), {} ]

    def render(self, mode='human', close=False):
        return

class L2RunEnv(OsimEnv):
    model_path = os.path.join(os.path.dirname(__file__), '../models/gait9dof18musc.osim')    
    time_limit = 1000

    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.7

    ## Values in the observation vector
    # y, vx, vy, ax, ay, rz, vrz, arz of pelvis (8 values)
    # x, y, vx, vy, ax, ay, rz, vrz, arz of head, torso, toes_l, toes_r, talus_l, talus_r (9*6 values)
    # rz, vrz, arz of ankle_l, ankle_r, back, hip_l, hip_r, knee_l, knee_r (7*3 values)
    # activation, fiber_len, fiber_vel for all muscles (3*18)
    # x, y, vx, vy, ax, ay ofg center of mass (6)
    # 8 + 9*6 + 8*3 + 3*18 + 6 = 146
    def get_observation(self):
        state_desc = self.get_state_desc()

        # Augmented environment from the L2R challenge
        res = []
        pelvis = None

        for body_part in ["pelvis", "head","torso","toes_l","toes_r","talus_l","talus_r"]:
            cur = []
            cur += state_desc["body_pos"][body_part][0:2]
            cur += state_desc["body_vel"][body_part][0:2]
            cur += state_desc["body_acc"][body_part][0:2]
            cur += state_desc["body_pos_rot"][body_part][2:]
            cur += state_desc["body_vel_rot"][body_part][2:]
            cur += state_desc["body_acc_rot"][body_part][2:]
            if body_part == "pelvis":
                pelvis = cur
                res += cur[1:]
            else:
                cur_upd = cur
                cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
                cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6,7)]
                res += cur

        for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in state_desc["muscles"].keys():
            res += [state_desc["muscles"][muscle]["activation"]]
            res += [state_desc["muscles"][muscle]["fiber_length"]]
            res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
        res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

        return res

    def get_observation_space_size(self):
        return 143

    def reward(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        return state_desc["joint_pos"]["ground_pelvis"][1] - prev_state_desc["joint_pos"]["ground_pelvis"][1]

class Run3DEnv(OsimEnv):
    model_path = os.path.join(os.path.dirname(__file__), '../models/gait14dof22musc_20170320.osim')    
    time_limit = 300

    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.7

    ## Values in the observation vector
    # y, vx, vy, ax, ay, rz, vrz, arz of pelvis (8 values)
    # x, y, vx, vy, ax, ay, rz, vrz, arz of head, torso, toes_l, toes_r, talus_l, talus_r (9*6 values)
    # rz, vrz, arz of ankle_l, ankle_r, back, hip_l, hip_r, knee_l, knee_r (7*3 values)
    # activation, fiber_len, fiber_vel for all muscles (3*18)
    # x, y, vx, vy, ax, ay ofg center of mass (6)
    # 8 + 9*6 + 8*3 + 3*18 + 6 = 146
    def get_observation(self):
        state_desc = self.get_state_desc()

        # Augmented environment from the L2R challenge
        res = []
        pelvis = None

        for body_part in ["pelvis", "head","torso","toes_l","toes_r","talus_l","talus_r"]:
            cur = []
            cur += state_desc["body_pos"][body_part][0:2]
            cur += state_desc["body_vel"][body_part][0:2]
            cur += state_desc["body_acc"][body_part][0:2]
            cur += state_desc["body_pos_rot"][body_part][2:]
            cur += state_desc["body_vel_rot"][body_part][2:]
            cur += state_desc["body_acc_rot"][body_part][2:]
            if body_part == "pelvis":
                pelvis = cur
                res += cur[1:]
            else:
                cur_upd = cur
                cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
                cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6,7)]
                res += cur

        for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in state_desc["muscles"].keys():
            res += [state_desc["muscles"][muscle]["activation"]]
            res += [state_desc["muscles"][muscle]["fiber_length"]]
            res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
        res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

        return res

    def get_observation_space_size(self):
        return 167

    def reward(self):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        return state_desc["joint_pos"]["ground_pelvis"][3] - prev_state_desc["joint_pos"]["ground_pelvis"][3]

import osim

class Arm2DEnv(OsimEnv):
    model_path = os.path.abspath(os.path.join(os.path.dirname(osim.__file__), 'models/arm2dof6musc.osim'))    
    time_limit = 200
    target_x = 0
    target_y = 0

    def get_observation(self):
        """
            The observation vector is [target_x, target_y, shoulder_pos, shoulder_vel, shoulder_acc
                                        elbow_pos, elbow_vel, elbow_acc, muscle_activation*6,
                                        styloid_radius*2]

            Remove 3 and 6 to remove velocity
            Remove 4 and 7 to remove acceleration

        """
        state_desc = self.get_state_desc()

        res = [self.target_x, self.target_y]

        # for body_part in ["r_humerus", "r_ulna_radius_hand"]:
        #     res += state_desc["body_pos"][body_part][0:2]
        #     res += state_desc["body_vel"][body_part][0:2]
        #     res += state_desc["body_acc"][body_part][0:2]
        #     res += state_desc["body_pos_rot"][body_part][2:]
        #     res += state_desc["body_vel_rot"][body_part][2:]
        #     res += state_desc["body_acc_rot"][body_part][2:]


        for joint in ["r_shoulder","r_elbow",]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in state_desc["muscles"].keys():
            res += [state_desc["muscles"][muscle]["activation"]]
            # res += [state_desc["muscles"][muscle]["fiber_length"]]
            # res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        res += state_desc["markers"]["r_radius_styloid"]["pos"][:2]

        return np.asarray(res, dtype=np.float32)

    def get_observation_space_size(self):
        return 16 #46

    def generate_new_target(self):
        theta = random.uniform(math.pi*9/8, math.pi*12/8)
        radius = random.uniform(0.5, 0.65)
        self.target_x = math.cos(theta) * radius 
        self.target_y = math.sin(theta) * radius

        state = self.osim_model.get_state()

#        self.target_joint.getCoordinate(0).setValue(state, self.target_x, False)
        self.target_joint.getCoordinate(1).setValue(state, self.target_x, False)

        self.target_joint.getCoordinate(2).setLocked(state, False)
        self.target_joint.getCoordinate(2).setValue(state, self.target_y, False)
        self.target_joint.getCoordinate(2).setLocked(state, True)
        
        
    def reset(self):
        obs = super(Arm2DEnv, self).reset()
        if not self.target_generated or not self.one_target:
            self.generate_new_target()
        self.target_generated = True
        self.osim_model.reset_manager()
        self.start_position = obs[[2,5]]
        self.start_dist = np.abs(obs[-2] - self.target_x), np.abs(obs[-1] - self.target_y)
        return obs

    def __init__(self, *args,one_target=False, max_speed=5., kin_coef=0., vel_prof_coef=0., acc_coef=0.,**kwargs):
        super(Arm2DEnv, self).__init__(*args, **kwargs)
        blockos = opensim.Body('target', 0.0001 , opensim.Vec3(0), opensim.Inertia(1,1,.0001,0,0,0) );
        self.target_joint = opensim.PlanarJoint('target-joint',
                                  self.osim_model.model.getGround(), # PhysicalFrame
                                  opensim.Vec3(0, 0, 0),
                                  opensim.Vec3(0, 0, 0),
                                  blockos, # PhysicalFrame
                                  opensim.Vec3(0, 0, -0.25),
                                  opensim.Vec3(0, 0, 0))

        geometry = opensim.Ellipsoid(0.02, 0.02, 0.02);
        geometry.setColor(opensim.Green);
        blockos.attachGeometry(geometry)
        self.one_target = one_target
        self.target_generated = False
        self.osim_model.model.addJoint(self.target_joint)
        self.osim_model.model.addBody(blockos)
        self.osim_model.model.initSystem()
        self.kin_coef = kin_coef
        self.vel_prof_coef = vel_prof_coef
        self.max_speed = max_speed
    def velocity_profile_reward(self, obs):

        vel1, vel2 = obs[3], obs[6]
        dist1, dist2 = np.abs(obs[-2] - self.target_x), np.abs(obs[-1] - self.target_y)

        #Peek when dist = 1/2 start_dist
        vel_target1 = self.max_speed - (dist1-self.start_dist[0]/2.)**2
        vel_target2 = self.max_speed - (dist2-self.start_dist[1]/2.)**2

        
        #Squared loss between vel_target and vel
        return -self.vel_prof_coef  * ((vel_target1-vel1)**2 + (vel_target2 -vel2)**2)/2.

    def kinetic_energy_reward(self, obs):
        kinetic_energy = (obs[3]**2 + obs[6]**2)/2.
        return  - self.kin_coef * kinetic_energy

    def reward(self):
        state_desc = self.get_state_desc()
        distance_penalty = (state_desc["markers"]["r_radius_styloid"]["pos"][0] - self.target_x)**2 + (state_desc["markers"]["r_radius_styloid"]["pos"][1] - self.target_y)**2
        obs = self.get_observation()

        # print(state_desc["markers"]["r_radius_styloid"]["pos"])
        # print((self.target_x, self.target_y))
        return 1.- distance_penalty + self.kinetic_energy_reward(obs) +  self.velocity_profile_reward(obs)

import gym
import numpy as np
from .wrappers import ShrinkEnvWrapper


class Standardise(gym.Wrapper):

        def __init__(self, env, *args, **kwargs):
            super(Standardise, self).__init__(env, *args, **kwargs)


        def step(self, act):
            o, r, d, i = self.env.step(act)
            return np.asarray(o), float(r)-1. , bool(d), i

        def reset(self):
            o = self.env.reset()
            return np.asarray(o)


def make_osim(env_name):
    kin_coef = 0.
    max_speed = 6.
    prof_coef = 0.
    if 'KineticEnergy' in env_name:
        kin_coef = 0.3
    if 'VelocityProfile' in env_name:
        prof_coef = 0.6
        max_speed = 6.
    if 'Arm2D' in env_name:
        if 'OneGoal' in env_name:
            env =  Standardise(Arm2DEnv(max_speed=max_speed, kin_coef=kin_coef,
                vel_prof_coef=prof_coef, one_target=True, visualize=False))
        else:
            env =  Standardise(Arm2DEnv(max_speed=max_speed, kin_coef=kin_coef,
                vel_prof_coef=prof_coef, visualize=False))

    elif 'Arm3D' in env_name:
        env =  Standardise(Arm3DEnv(visualize=False))

    keep_indices_mask = np.ones(env.observation_space.shape[0], dtype=bool)
    if 'NoVel' in env_name:
        vel_indices = [3,6]
        keep_indices_mask[vel_indices] = False
    if 'NoAcc' in env_name:
        acc_indices = [4,7]
        keep_indices_mask[acc_indices] = False
    if 'NoMuscles' in env_name:
        muscle_indices = np.arange(7,13)
        keep_indices_mask[muscle_indices] = False
    if 'NoStyloid' in env_name:
        styloid_indices = np.arange(13,15)
        keep_indices_mask[styloid_indices] = False


    if not np.all(keep_indices_mask):
        env = ShrinkEnvWrapper(env, keep_indices_mask)

    return env


if __name__ == '__main__':
    import numpy as np
    import random
    env_name = 'OsimArm2DOneGoal'
    env1 = make_osim(env_name)
    env2 = make_osim(env_name + 'NoVel')
    env3 = make_osim(env_name + 'NoAcc')
    env4 = make_osim(env_name + 'NoAccNoVel')
    env5 = make_osim(env_name + 'NoMuscles')
    random.seed(111)
    obs1 = env1.reset()
    random.seed(111)
    obs2 = env2.reset()
    random.seed(111)
    obs3 = env3.reset()
    random.seed(111)
    obs4 = env4.reset()
    random.seed(111)
    obs5 = env5.reset()

    print("Checking if targets same...", env1.env.target_x, env2.env.env.target_x, env3.env.env.target_x)
    assert env1.env.target_x == env2.env.env.target_x == env3.env.env.target_x, "Targets should be same"
    assert env1.observation_space.shape[0] == obs2.shape[0]+2, "2 observations should be removed with NoVel"
    assert env1.observation_space.shape[0] == obs3.shape[0]+2, "2 observations should be removed with NoAcc"
    assert env1.observation_space.shape[0] == obs4.shape[0]+4, "4 observations should be removed with NoVelNoAcc"
    assert env1.observation_space.shape[0] == obs5.shape[0]+6, "6 observations should be removed with NoMuscles"

    print("All cool!")
