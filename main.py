import os
import sys
import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import gym
import random
import tensorflow as tf
from tf2rl.envs.utils import is_discrete, get_act_dim
from sac import SAC
from tf2rl.algos.td3 import TD3
from trainer import Trainer
from pureBB import BBTrainer
import wandb
import pickle
import shutil
import glob

# sys.path.insert(1, os.path.join(sys.path[0], '../..'))
env_dir = "Cheetah-Gym"
sys.path.insert(1, env_dir)

OLD = False

if not OLD:
	from dog import Dog
else:
	from old_dog import Dog

DEBUG = False  # REMEMBER TO TURN IT OFF BEFORE YOU GOING TO SLEEP !!!
BO = True
REAL_TIME = False
ENV_VER = 3
# FOR SINE/ROSE/LINE GAIT IN THE SIMULATOR
DOMAIN_RANGE = [[0.01, 0.1], [0, 0.1], [0, 0.1], [1, 5], [0.04, 0.1], [0.04, 0.1], [0.04, 0.1], [0.04, 0.1], [0, 0.1]]  # [[0.01, 0.1], [0, 0.1], [0, 0.1], [1, 5]] # [[0.01, 0.1], [0, 0.1], [0, 0.1], [1, 5], [0, 0.2], [0, 0.2]]
INI_GUESS = [0.035, 0, 0, 1, 0.1, 0.1, 0.1, 0.1, 0.02] # [0.015, 0, 0, 6, 0.1, 0.1, 0.1, 0.1] # [0.0896, 0.0603, 0.0645, 4.3990] # [0.035, 0, 0, 2.85714, 0.1, 0.1]  # [0.0896, 0.0603, 0.0645, 4.3990] # [0.0712, 0.0777, 0.1463, 2.5790] #[0.036, 0.01, 0.02, 4] #[0.06000, 0.05, 0.12, 2] # [0.0323, 0.0642, 0.0994, 4.5330] #SAC_20210727036  # 
# Note that these may be overwrote



# FOR TRIANGLE GAIT ON REAL ROBOT
# DOMAIN_RANGE = [[0.01, 0.03], [0, 0.1], [0, 0.1], [9, 11], [0.04, 0.1], [0.04, 0.1], [0.04, 0.1], [0.04, 0.1], [0, 0.02]]
# INI_GUESS = [0.022, 0, 0, 10, 0.1*0.8, 0.06*0.8, 0.1*0.8, 0.1*0.8, 0.011]

OPTIMISATION_MASK = "100100001"
INI_STEP_GUESS = [INI_GUESS[0]*INI_GUESS[3], INI_GUESS[0]]
PARAM_UPDATE_INTERVAL = 10
OPTIMISER_WARMUP = int(3e3)
FITTING_MODE = "training"
OPTIMISER = "TBPSA"
# ROTATION = False
ACTION_MODE="residual"  # SAC_20210726010 is a good example for whole RL control
WANDB = True
RESIDUAL_MULTIPLIER = 0.2
ACTION_MULTIPLIER = 0.4
LEG_MULTIPLIER = 0.1 # 0.2
A_RANGE = (0.1, 0.5)
B_RANGE = (float("-inf"), float("inf"))
ARM_PD = True
FAST_CONTROL = True
STATE_MODE = ["body_arm_p" , "body_arm_leg_full_p", "body_arm_leg_full", "h_body_arm"][0]
LEG_ACTION = ["none", "parallel_offset", "hips_offset", "knees_offset", "hips_knees_offset"][0]
GAIT = ["sine", "rose", "triangle", "line"][0]

# SIMULATION FRIENDLY CONFIGERATION
DYN_CONFIG0 = {'lateralFriction_toe': 1, # 0.6447185797960826, 
			   'lateralFriction_shank': 0.737, #0.6447185797960826  *  (0.351/0.512),
			   'contactStiffness': 4173, #2157.4863390669952, 
			   'contactDamping': 122, #32.46233575737161, 
			   'linearDamping': 0.03111169082496665, 
			   'angularDamping': 0.04396695866661371, 
			   'jointDamping': 0.03118494025640309, 
			   }

# CLOSER TO REAL WORLD
DYN_CONFIG_HARD3 = {'lateralFriction_toe': 0.7, #1, # 0.6447185797960826, 
					'lateralFriction_shank': 0.5, #0.6447185797960826  *  (0.351/0.512),
					'contactStiffness': 2592, #2729, # 4173, #2157.4863390669952, 1615?
					'contactDamping': 450, #414, #160, # 122, #32.46233575737161,    150?
					'linearDamping': 0.03111169082496665, 
					'angularDamping': 0.04396695866661371, 
					'jointDamping': 0.03118494025640309, 
					"max_force": 12, #10,
					"mass_body": 9.5 #10.5
					}

DYN_CONFIG = [DYN_CONFIG0, DYN_CONFIG_HARD3][0]

PARAM_OPT_FOR_TESTING = None



if __name__ == '__main__':
	parser = Trainer.get_argument()
	parser = SAC.get_argument(parser)
	parser = TD3.get_argument(parser)
	parser.add_argument('--DEBUG', action="store_true", default=False)
	# args = parser.parse_args()
	# DEBUG = args.DEBUG
	parser.add_argument('--action-mode', type=str, default=ACTION_MODE)
	parser.add_argument('--action-multiplier', type=float, default=ACTION_MULTIPLIER)
	parser.add_argument('--arm-pd', action="store_true", default=ARM_PD)
	parser.add_argument('--dynamics-setting', type=str, default="easy")  # easy / hard / real (hard+small range)
	parser.add_argument('--disable-wandb', action="store_true", default=False)
	parser.add_argument('--eval-using-online-param', action="store_true", default=False)
	parser.add_argument('--external-force', type=float, default=0)
	parser.add_argument('--eval-using-online-param', action="store_true", default=False)
	parser.add_argument('--fitting-mode', type=str, default=FITTING_MODE)
	parser.add_argument('--fast-error-update', action="store_true", default=True)
	parser.add_argument('--gait', type=str, default=GAIT)
	parser.add_argument('--leg-bootstrapping', action="store_true", default=False)  # be careful. developing!
	parser.add_argument('--leg-offset-range', type=float, default=0.4)
	parser.add_argument('--leg-multiplier', type=float, default=LEG_MULTIPLIER)
	parser.add_argument('--leg-action-mode', type=str, default=LEG_ACTION)
	parser.add_argument('--num-history-observation', type=int, default=0)
	parser.add_argument('--num-experiment', type=int, default=1)
	parser.add_argument('--note', type=str, default="")
	parser.add_argument('--optimiser', type=str, default=OPTIMISER)
	parser.add_argument('--optimiser-warmup', type=int, default=OPTIMISER_WARMUP if not DEBUG else 500)
	parser.add_argument('--optimisation-mask', type=str, default=OPTIMISATION_MASK)
	parser.add_argument('--only-randomise-dyn', action="store_true", default=False)
	parser.add_argument('--profile', action="store_true", default=False)
	parser.add_argument('--param-update-interval', type=int, default=PARAM_UPDATE_INTERVAL)
	parser.add_argument('--progressing', action="store_true", default=False)
	parser.add_argument('--policy', type=str, default="SAC")
	parser.add_argument('--rotation', action="store_true", default=False)
	parser.add_argument('--residual-multiplier', type=float, default=RESIDUAL_MULTIPLIER)
	parser.add_argument('--residual-with-optimisation', action="store_true", default=False)
	parser.add_argument('--reset-mode', type=str, default="stand")
	parser.add_argument('--robot-k', type=float, default=0.69)
	parser.add_argument('--randomise', type=float, default=0)
	parser.add_argument('--randomise-eval', type=float, default=0)
	parser.add_argument('--random-initial', action="store_true", default=False)
	parser.add_argument('--render', action="store_true", default=False)
	parser.add_argument('--state-mode', type=str, default=STATE_MODE)
	args = parser.parse_args()
	parser.set_defaults(batch_size=256)
	parser.set_defaults(n_warmup=10000 if not DEBUG else 100)
	parser.set_defaults(max_steps=3e6 if not DEBUG else 2000)
	parser.set_defaults(episode_max_steps=2000 if args.progressing else 1000)
	if DEBUG:
		parser.set_defaults(test_interval=1001)  # FOR DEBUGGING
	# parser.set_defaults(model_dir="results/20210707T121014.071938_SAC_") // VERSION 1
	# parser.set_defaults(model_dir="results/20210714T220333.388640_SAC_")  # new action limit, action factor 0.8->0.4
	# parser.set_defaults(model_dir="results/20210715T224434.334049_SAC_")  # new rear legs pattern, the cheated result, Optimal Point: [0.2188, 0.0607]
	# parser.set_defaults(model_dir="results/20210716T112428.073500_SAC_")  # fully trained, ver2
	# parser.set_defaults(model_dir="results/20210723T002346.052897_SAC_")  # fully trained, ver3, Optimal Value: 1085.6391 Optimal Point: [0.1000, 0.0350]
	# parser.set_defaults(model_dir="results/20210723T210835.832958_SAC_")  # fully trained, ver3, Optimal Value: 1584.2914 Optimal Point: [0.1000, 0.0635]
	# parser.set_defaults(model_dir="results/SAC_20210724020F")  # fully trained, ver3, Optimal Point: [0.1000, 0.0250, 0.0873, 0.2276]  Optimal Value: 1129.6312; the control is based on amplitude
	# parser.set_defaults(save_test_movie=True)  # code the render function for mode='rgb_array'  20210723T210835.832958_SAC_  
	args = parser.parse_args()


	# SIMULATION FRIENDLY CONFIGERATION
	DYN_CONFIG0 = {'lateralFriction_toe': 1, # 0.6447185797960826, 
				   'lateralFriction_shank': 0.737, #0.6447185797960826  *  (0.351/0.512),
				   'contactStiffness': 4173, #2157.4863390669952, 
				   'contactDamping': 122, #32.46233575737161, 
				   'linearDamping': 0.03111169082496665, 
				   'angularDamping': 0.04396695866661371, 
				   'jointDamping': 0.03118494025640309, 
				   }

	# CLOSER TO REAL WORLD
	DYN_CONFIG_HARD3 = {'lateralFriction_toe': 0.7, #1, # 0.6447185797960826, 
						'lateralFriction_shank': 0.5, #0.6447185797960826  *  (0.351/0.512),
						'contactStiffness': 2592, #2729, # 4173, #2157.4863390669952, 1615?
						'contactDamping': 450, #414, #160, # 122, #32.46233575737161,    150?
						'linearDamping': 0.03111169082496665, 
						'angularDamping': 0.04396695866661371, 
						'jointDamping': 0.03118494025640309, 
						"max_force": 12, #10,
						"mass_body": 9.5 #10.5
						}


	if args.gait == "triangle":  # pay attention that this is for simulation training
		# FOR TRIANGLE GAIT IN THE SIMULATOR
		DOMAIN_RANGE = [[0.01, 0.1], [0, 0.1], [0, 0.1], [5, 20], [0.04, 0.1], [0.04, 0.1], [0.04, 0.1], [0.04, 0.1], [0, 0.1]]
		if not args.random_initial:
			INI_GUESS = [0.022, 0, 0, 10, 0.1, 0.1, 0.1, 0.1, 0.02]
		else:
			INI_GUESS = [np.random.uniform(r[0], r[1]) for r in DOMAIN_RANGE]
		A_RANGE = (0.1, 2)



	if args.dynamics_setting == "easy":
		DYN_CONFIG = DYN_CONFIG0
	elif args.dynamics_setting == "hard":
		DYN_CONFIG = DYN_CONFIG_HARD3
	elif args.dynamics_setting == "real":
		DYN_CONFIG = DYN_CONFIG_HARD3
		if args.gait == "triangle": 
			DOMAIN_RANGE = [[0.01, 0.03], [0, 0.1], [0, 0.1], [9, 11], [0.04, 0.1], [0.04, 0.1], [0.04, 0.1], [0.04, 0.1], [0, 0.02]]
			if not args.random_initial:
				INI_GUESS = [0.022, 0, 0, 10, 0.1*0.8, 0.06*0.8, 0.1*0.8, 0.1*0.8, 0.011]
			else:
				INI_GUESS = [np.random.uniform(r[0], r[1]) for r in DOMAIN_RANGE]
			A_RANGE = (0.1, 2)  # seems useless
		else:
			DOMAIN_RANGE = [[0.01, 0.03], [0, 0.1], [0, 0.1], [1, 3], [0.04, 0.1], [0.04, 0.1], [0.04, 0.1], [0.04, 0.1], [0, 0.02]]  
			if not args.random_initial:
				INI_GUESS = [0.022, 0, 0, 2, 0.1*0.8, 0.1*0.8, 0.1*0.8, 0.1*0.8, 0.011] 
			else:
				INI_GUESS = [np.random.uniform(r[0], r[1]) for r in DOMAIN_RANGE]
			A_RANGE = (0.01, 0.2)  # seems useless

	else:
		print("WHAT THE F**K IS ", args.dynamics_setting, " ???")



	
	if DEBUG:
		WANDB = False

	LEG_OFFSET_RANGE = (-args.leg_offset_range, args.leg_offset_range)


	ROTATION = args.rotation
	FITTING_MODE = args.fitting_mode
	OPTIMISER = args.optimiser
	ACTION_MODE = args.action_mode
	OPTIMISER_WARMUP = args.optimiser_warmup
	ACTION_MULTIPLIER = args.action_multiplier
	RESIDUAL_MULTIPLIER = args.residual_multiplier
	RENDER = args.render #if not DEBUG else True

	if args.gpu < 0:
		tf.config.experimental.set_visible_devices([], 'GPU')
	else:
		if tf.config.experimental.list_physical_devices('GPU'):
			for cur_device in tf.config.experimental.list_physical_devices("GPU"):
				print(cur_device)
				tf.config.experimental.set_memory_growth(cur_device, enable=True)


	if args.action_mode == "residual" and not args.residual_with_optimisation:
		BO = False

	if args.evaluate or args.model_dir is not None:
		BO = False
		REAL_TIME = True
		# RENDER = True
		if os.path.isfile(os.path.join(args.model_dir, 'env_config.pkl')):
			with open(os.path.join(args.model_dir, 'env_config.pkl'), 'rb') as f:
				env_args = pickle.load(f)
				env_args["render"] = True
				
		elif os.path.isfile(os.path.join(args.model_dir, os.pardir, 'env_config.pkl')):
			with open(os.path.join(args.model_dir, os.pardir, 'env_config.pkl'), 'rb') as f:
				env_args = pickle.load(f)
				env_args["render"] = True
		else:
			# FOR THE PREVIOUS EXPERIMENT THAT PARAMETERS ARE NOT SAVED
			env_args = {"render": True, "fix_body": False, "real_time": True, "immortal": False, 
						"version": 3, "normalised_abduct": False, "mode": "stand", "action_mode": "residual", "residual_multiplier": 0.2, 
						"action_multiplier": 0.4, "A_range": [float("-inf"), 0.5], "B_range": [float("-inf"), float("inf")], "arm_pd_control": False, "fast_error_update": False,
						"state_mode": "body_arm_p", "leg_action_mode": "parameter", "leg_offset_multiplier": 0, "ini_step_param": [0.1, 0.01]
						}
			
		assert "custom_dynamics" in env_args  # NEW DYNAMICS SETTING FRAMWORK

		if os.path.isfile(os.path.join(args.model_dir, 'env_dynamics_config.pkl')):
			with open(os.path.join(args.model_dir, 'env_dynamics_config.pkl'), 'rb') as f:
				DYN_CONFIG = pickle.load(f)
		elif os.path.isfile(os.path.join(args.model_dir, os.pardir, 'env_dynamics_config.pkl')):
			with open(os.path.join(args.model_dir, os.pardir, 'env_dynamics_config.pkl'), 'rb') as f:
				DYN_CONFIG = pickle.load(f)
		else:
			# DYN_CONFIG = {'lateralFriction_toe': 0.6447185797960826, 
			#   'lateralFriction_shank': 0.6447185797960826  *  (0.351/0.512),
			#   'contactStiffness': 2157.4863390669952, 
			#   'contactDamping': 32.46233575737161, 
			#   'linearDamping': 0.03111169082496665, 
			#   'angularDamping': 0.04396695866661371, 
			#   'jointDamping': 0.03118494025640309, 
			#   # 'w_y_offset': 0.0021590823485152914
			#   }

			DYN_CONFIG = {'lateralFriction_toe': 1, # 0.6447185797960826, 
						  'lateralFriction_shank': 0.737, #0.6447185797960826  *  (0.351/0.512),
						  'contactStiffness': 4173, #2157.4863390669952, 
						  'contactDamping': 122, #32.46233575737161, 
						  'linearDamping': 0.03111169082496665, 
						  'angularDamping': 0.04396695866661371, 
						  'jointDamping': 0.03118494025640309, 
						  }

	else:
		max_steps = 1000 if not args.progressing else 2000
		env_args = {"render": (RENDER or args.evaluate), "fix_body": False, "real_time": REAL_TIME, "immortal": False, 
					"version": ENV_VER, "normalised_abduct": True, "action_mode": args.action_mode, "residual_multiplier": args.residual_multiplier, 
					"action_multiplier": args.action_multiplier, "A_range": A_RANGE, "B_range": B_RANGE, "arm_pd_control": args.arm_pd, "fast_error_update": args.fast_error_update,
					"state_mode": args.state_mode, "leg_action_mode": args.leg_action_mode, "leg_offset_multiplier": args.leg_multiplier, "ini_step_param": INI_STEP_GUESS,
					"param_opt": INI_GUESS, "gait": args.gait,
					"num_history_observation": args.num_history_observation, "randomise": args.randomise, "custom_dynamics": DYN_CONFIG, "max_steps": max_steps,
					"progressing" : args.progressing, "custom_robot": {"k": args.robot_k}, "mode": args.reset_mode, "leg_offset_range": LEG_OFFSET_RANGE,
					"external_force": args.external_force, "leg_bootstrapping": args.leg_bootstrapping, "only_randomise_dyn": args.only_randomise_dyn
					}



	# env_args = {"render": True, "fix_body": False, "real_time": False, "immortal": False, 
	# 			"version": 3, "normalised_abduct": True, "mode": "stand", "debug_tuner_enable" : False, "action_mode":"residual", 
	# 			"state_mode":"body_arm_leg_full", "leg_action_mode":"hips_offset",
	# 			"tuner_enable": False, "action_tuner_enable": False, "A_range": (0.1, 0.5), "B_range": (0.01, 0.1), "gait": "triangle",
	# 			"arm_pd_control": True, "fast_error_update": True, "leg_offset_multiplier" : 0.1,
	# 			"custom_dynamics": DYN_CONFIG,
	# 			"mode": "standup", "randomise": True
	# 			}

	# assert env_args["gait"] == "triangle"


	env = Dog(**env_args)
	if not args.evaluate:
		env_args["render"] = False
		if args.randomise_eval:
			env_args["randomise"] = args.randomise_eval
		test_env = Dog(**env_args)
	else:
		test_env = env

	if OLD:
		env.set_dynamics(**DYN_CONFIG)
		test_env.set_dynamics(**DYN_CONFIG)

	if args.policy =="SAC":
		policy = SAC(
			state_shape=env.observation_space.shape,
			action_dim=env.action_space.high.size,
			gpu=0,
			memory_capacity=args.memory_capacity,
			max_action=env.action_space.high[0],
			batch_size=args.batch_size,
			n_warmup=args.n_warmup,
			alpha=args.alpha,
			auto_alpha=args.auto_alpha,
			actor_units=(256, 256) if args.num_history_observation==0 else (256, 256, 256),
			critic_units=(256, 256) if args.num_history_observation==0 else (256, 256, 256))
	elif args.policy =="TD3":
		policy = TD3(
			state_shape=env.observation_space.shape,
			action_dim=env.action_space.high.size,
			gpu=0,
			memory_capacity=args.memory_capacity,
			max_action=env.action_space.high[0],
			batch_size=args.batch_size,
			n_warmup=args.n_warmup,
			# alpha=args.alpha,
			# auto_alpha=args.auto_alpha,
			actor_units=(256, 256) if args.num_history_observation==0 else (256, 256, 256),
			critic_units=(256, 256) if args.num_history_observation==0 else (256, 256, 256))
	elif args.policy =="none":
		pass
	else:
		print("WHAT THE F**K IS ", args.policy, " ???")

	if args.optimiser == "none":
		BO = False
		INI_GUESS = [0]*9

	for exp_i in range(args.num_experiment):

		print("")
		print("========================================================")
		print("                 START EXPERIENT ", exp_i)
		print("========================================================")
		print("")

		if not args.policy == "none":

			trainer = Trainer(policy, env, args, test_env=test_env, 
							  bo = BO,
							  param_domain = DOMAIN_RANGE,  # if the length is 4, they stand for [A, Kp, Kd, B factor]  v2: [B, Kp, Kd, A factor]     # param_domain = [[0.3, 0.6], [0.02, 0.1]],
							  param_opt = INI_GUESS, # [0.1000, 0.0635], # [0.1188, 0.0607], # [0.12, 0.0389], #[0.2, 0.0389],
							  param_update_interval_epi = args.param_update_interval,
							  warmup_epi = args.optimiser_warmup if not DEBUG else 3,
							  fitting = FITTING_MODE,
							  optimiser = args.optimiser,
							  rotation = args.rotation,
							  debug = DEBUG,
							  profiler_enable = args.profile,
							  optimisation_mask = args.optimisation_mask,
							  param_opt_testing = PARAM_OPT_FOR_TESTING,
							  eval_using_online_param = args.eval_using_online_param,
							  DEBUG = DEBUG)

		else:

			trainer = BBTrainer(env, args, test_env=test_env, param_opt=INI_GUESS, param_domain=DOMAIN_RANGE, optimisation_mask=args.optimisation_mask, optimiser=args.optimiser, max_steps=3e6,
								eval_using_online_param=args.eval_using_online_param)

		if not args.evaluate:
			with open(os.path.join(trainer._output_dir, "env_config.pkl"), 'wb') as f:
				pickle.dump(env_args, f, pickle.HIGHEST_PROTOCOL)
			with open(os.path.join(trainer._output_dir, "env_dynamics_config.pkl"), 'wb') as f:
				pickle.dump(DYN_CONFIG, f, pickle.HIGHEST_PROTOCOL)

			##########  SAVE SRC FILES
			if not os.path.isdir(os.path.join(trainer._output_dir, "src")) :
				os.mkdir(os.path.join(trainer._output_dir, "src"))
			# shutil.copy(os.path.abspath(__file__), os.path.join(trainer._output_dir, "src")) 
			# shutil.copy("tf2rl/experiments/trainer.py", os.path.join(trainer._output_dir, "src")) 
			src_files = glob.glob("*.py")
			for src_file in src_files:
				shutil.copy(src_file, os.path.join(trainer._output_dir, "src"))

			shutil.copytree(env_dir, os.path.join(trainer._output_dir, "src", "environment"), dirs_exist_ok=True)

			# if not os.path.isdir(os.path.join(trainer._output_dir, "src", "environment")) :
			# 	os.mkdir(os.path.join(trainer._output_dir, "src", "environment"))
			# shutil.copy(os.path.abspath(__file__), os.path.join(trainer._output_dir, "src")) 


		config = {}
		config["experiment/log_directory"] = trainer._output_dir
		config["experiment/env_version"] = ENV_VER
		# config["experiment/action_mode"] = ACTION_MODE
		for conf in DYN_CONFIG:
			config["dynamics/"+conf] = DYN_CONFIG[conf]
		parser_dict = vars(args)
		# del parser_dict['fitting_mode']
		# del parser_dict['optimiser']
		# del parser_dict['rotation']
		# del parser_dict['action_mode']
		# del parser_dict['optimiser_warmup']
		for conf in parser_dict:
			config["training/"+conf] = vars(args)[conf]
		config["optimisation/turn_on"] = BO
		config["optimisation/domain_range"] = DOMAIN_RANGE
		config["optimisation/initial_guess"] = INI_GUESS
		config["optimisation/param_update_interval"] = PARAM_UPDATE_INTERVAL
		config["optimisation/optimisation_mask"] = args.optimisation_mask
		# config["optimisation/optimiser_warmup"] = OPTIMISER_WARMUP
		# config["optimisation/fitting_mode"] = FITTING_MODE
		# config["optimisation/optimiser"] = OPTIMISER
		# config["optimisation/rotation"] = ROTATION

		for conf in env_args:
			if not conf == "experiment_info_str":
				config["environment/"+conf] = env_args[conf]
		# config["training/A_range"] = A_RANGE
		# config["training/B_range"] = B_RANGE

		name = trainer._policy.policy_name
		name = name + "+" + OPTIMISER if ACTION_MODE=="partial" else name
		if ACTION_MODE == "partial":
			name = name + "+PD"  if len(DOMAIN_RANGE)==4 else name
			name = name + "[T]" if FITTING_MODE == "training" else name + "[E]" 
			name = name + "[R]" if ROTATION else name
		elif ACTION_MODE == "whole":
			name = name + "+CP"
		elif ACTION_MODE == "residual":
			name = "Residual" + name if not args.policy == "none" else name
			if BO:
				name = name + "+" + OPTIMISER
			else:
				name = trainer._policy.policy_name

		

		if args.progressing:
			name = name  + "[Prog]"

		if args.gait == "rose":
			name = name  + "[Rose]"
		elif args.gait == "triangle":
			name = name  + "[Trig]"
		elif args.gait == "sine":
			name = name  + "[Sine]"
		elif args.gait == "line":
			name = name  + "[Line]"

		if args.randomise:
			name = name  + "[Rand]"
		if args.randomise_eval:
			name = name  + "[REval]"
		if args.dynamics_setting == "hard":
			name = name  + "[Hard]"
		elif args.dynamics_setting == "real":
			name = name  + "[Real]"

		if args.optimiser_warmup < 0:
			name = name  + "[BBF]"


		if OLD:
			name = name  + "[Old]"



		env.note = name
		test_env.note = name
		experiment_info = ""
		for k in config:
			experiment_info += k + " : " + str(config[k]) + "\n"
		print(experiment_info)
		env.experiment_info_str = experiment_info
		test_env.experiment_info_str = experiment_info

		if not os.path.isdir(os.path.join(trainer._output_dir, os.pardir, ".wandb")) :
			os.mkdir(os.path.join(trainer._output_dir, os.pardir, ".wandb"))
		wandb_dir = os.path.join(trainer._output_dir, os.pardir, ".wandb", os.path.basename(trainer._output_dir))
		if not os.path.isdir(wandb_dir) :
			os.mkdir(wandb_dir)
		wandb_dir = os.path.abspath(wandb_dir)
		wandb.init(config=config, project="Standing Cheetah", name=name, dir=wandb_dir, mode="disabled" if args.evaluate or not WANDB or args.disable_wandb else "online", notes=args.note)

		if args.evaluate or args.model_dir is not None:
			trainer.evaluate_policy_continuously()
		else:
			trainer()

		wandb.finish()

	test_env._p.disconnect()
	env._p.disconnect()


