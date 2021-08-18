import os
import os.path
import time
import logging
import argparse
# import pickle
import dill as pickle
import numpy as np
import tensorflow as tf
from gym.spaces import Box

from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.envs.normalizer import EmpiricalNormalizer

# from dragonfly.exd import domains
# from dragonfly.exd.experiment_caller import CPFunctionCaller, EuclideanFunctionCaller
# from dragonfly.opt import random_optimiser, cp_ga_optimiser, gp_bandit
import nevergrad as ng

import wandb

# if tf.config.experimental.list_physical_devices('GPU'):
# 	for cur_device in tf.config.experimental.list_physical_devices("GPU"):
# 		print(cur_device)
# 		tf.config.experimental.set_memory_growth(cur_device, enable=True)



class Trainer:
	def __init__(
			self,
			policy,
			env,
			args,
			test_env=None,
			param_update_interval=0,
			param_update_interval_epi=10,
			param_opt = [],
			bo=True,
			param_domain = [[0.3, 0.5], [0.03,0.05]],
			save_model=False,
			warmup_epi=0,
			rotation=False,
			fitting="evaluation",  # "training" / "evaluation",
			optimiser="BO",
			debug=False,
			profiler_enable=False,
			optimisation_mask="000000000",
			param_opt_testing=None,
			eval_using_online_param=False,
			DEBUG = False):
		if isinstance(args, dict):
			_args = args
			args = policy.__class__.get_argument(Trainer.get_argument())
			args = args.parse_args([])
			for k, v in _args.items():
				if hasattr(args, k):
					setattr(args, k, v)
				else:
					raise ValueError(f"{k} is invalid parameter.")

		self._set_from_args(args)
		self._policy = policy
		self._env = env
		self._test_env = self._env if test_env is None else test_env
		if self._normalize_obs:
			assert isinstance(env.observation_space, Box)
			self._obs_normalizer = EmpiricalNormalizer(
				shape=env.observation_space.shape)

		# prepare log directory
		if not os.path.isdir(self._logdir) :
			os.mkdir(self._logdir)
		run_i = 0
		while (os.path.exists(os.path.join(self._logdir, "{}_{}{:03d}".format(self._policy.policy_name, int(time.strftime("%Y%m%d", time.localtime())), run_i))) 
			or os.path.exists(os.path.join(self._logdir, "{}_{}{:03d}F".format(self._policy.policy_name, int(time.strftime("%Y%m%d", time.localtime())), run_i)))
			or os.path.exists(os.path.join(self._logdir, "{}_{}{:03d}T".format(self._policy.policy_name, int(time.strftime("%Y%m%d", time.localtime())), run_i)))):
			run_i += 1

		self._output_dir = os.path.join(self._logdir, "{}_{}{:03d}".format(self._policy.policy_name, int(time.strftime("%Y%m%d", time.localtime())), run_i))
		if debug:
			self._output_dir += "T"
		os.mkdir(self._output_dir)
		os.mkdir(os.path.join(self._output_dir, "best_models"))


		# self._output_dir = prepare_output_dir(
		#     args=args, user_specified_dir=self._logdir,
		#     suffix="{}_{}".format(self._policy.policy_name, args.dir_suffix))
		self.logger = initialize_logger(
			logging_level=logging.getLevelName(args.logging_level),
			output_dir=self._output_dir)

		self.save_model = save_model
		self.bo = bo
		self.action_mode = self._env.action_mode
		self.fitting = fitting
		self.rotation = rotation
		self.optimiser_name = optimiser
		self.domains = param_domain
		self.debug = debug
		self.profiler_enable = profiler_enable
		self.optimisation_mask = optimisation_mask
		assert len(self.optimisation_mask) == 9
		self.param_opt = param_opt

		self.param_opt_masked = [self.param_opt[i] for i in range(len(self.optimisation_mask)) if self.optimisation_mask[i] == '1']
		self.domains_masked = [self.domains[i] for i in range(len(self.optimisation_mask)) if self.optimisation_mask[i] == '1']

		self.param_opt_testing = param_opt_testing
		self.eval_using_online_param = eval_using_online_param

		self.DEBUG = DEBUG

		print("optimisation_mask: ", self.optimisation_mask)
		print("param_opt_masked: ", self.param_opt_masked)
		print("domains_masked: ", np.array(self.domains_masked))


		if self.action_mode == "whole":
			self.logger.info("ACTION MODE: WHOLE BODY CONTROL WITH REINFORCEMENT LEARNING")
			self.bo = False
			assert env.action_space.shape[0] == 6
		elif self.action_mode == "patial":
			self.logger.info("ACTION MODE: PATIAL BODY CONTROL WITH BAYESIAN OPTIMISATION")
			assert env.action_space.shape[0] == 4
			if not args.evaluate:
				assert bo == True
		elif self.action_mode == "residual":
			pass
			# assert env.action_space.shape[0] == 6

		if args.evaluate:
			assert args.model_dir is not None
			assert bo == False
			assert param_opt != []
		self._set_check_point(args.model_dir)

		# prepare TensorBoard output
		self.writer = tf.summary.create_file_writer(self._output_dir)
		self.writer.set_as_default()

		self.param_update_interval = param_update_interval
		self.param_update_interval_epi = param_update_interval_epi
		
		self.bo_lock = True
		self.rl_lock = False
		self.warmup_epi = warmup_epi
		self.best_x, self.best_y = None, float('-inf')
		self.best_evaluation = float('-inf')

		# if self.bo:
		# 	self.param_opt = param_opt
		# 	if self.optimiser_name == "BO":
		# 		domain = domains.EuclideanDomain(param_domain)
		# 		self.func_caller = EuclideanFunctionCaller(None, domain)

		# 	# options = argparse.Namespace(
		# 	# rand_exp_sampling_replace = True
		# 	# )
		# 	# self.optimiser = gp_bandit.EuclideanGPBandit(self.func_caller, ask_tell_mode=True, options=options)
		# 	# self.optimiser.initialise()
		# 	if not self.bo_lock == True:
		# 		pass
		# 		# param_opt = self.optimiser.ask()
		# 		# if type(param_opt[1]) == dict:  # return type of nevergrad
		# 		#     self.param_opt = [i for i in param_opt[0]]
		# 		# else:
		# 		#     self.param_opt = param_opt
		# 	else:
		# 		assert self.warmup_epi != 0 or self.rotation == True
		# 		self.param_opt = self.param_opt

		# 	assert len(self.param_opt) == len(param_domain)
		# else:
		# 	self.param_opt = param_opt

		self.step_param = [0, 0]
		self.returns = []

		
	def _set_check_point(self, model_dir):
		# Save and restore model
		self._checkpoint = tf.train.Checkpoint(policy=self._policy)
		self.checkpoint_manager = tf.train.CheckpointManager(
			self._checkpoint, directory=self._output_dir, max_to_keep=5)
		self.best_checkpoint_manager = tf.train.CheckpointManager(
			self._checkpoint, directory=os.path.join(self._output_dir, "best_models"), max_to_keep=2)

		if model_dir is not None:
			assert os.path.isdir(model_dir)
			self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
			self._checkpoint.restore(self._latest_path_ckpt)
			self.logger.info("Restored {}".format(self._latest_path_ckpt))
			model_save_dir = os.path.join(model_dir, "actor_model_dis")
			if not os.path.isdir(model_save_dir) and model_dir[-1] == 'F':
				os.mkdir(model_save_dir)
				s_dim = self._env.observation_space.shape[0]  # this assumes that the config has been restored to the environment
				self._policy.save_actor(model_save_dir, s_dim=s_dim) 

	def __call__(self):
		if self._evaluate:
			self.evaluate_policy_continuously()

		if self.profiler_enable:
			tf.profiler.experimental.start(self._output_dir)
			profiling = True


		total_steps = 0
		self.global_total_steps = 0
		opt_steps = 0
		tf.summary.experimental.set_step(total_steps)
		episode_steps = 0
		episode_return = 0
		episode_start_time = time.perf_counter()
		n_episode = 0

		replay_buffer = get_replay_buffer(
			self._policy, self._env, self._use_prioritized_rb,
			self._use_nstep_rb, self._n_step)

		obs = self._env.reset()

		p_error, d_error = 0, 0

		while total_steps < self._max_steps:

			if self.profiler_enable:
				if not profiling:
					tf.profiler.experimental.start(self._output_dir)
					profiling = True

			if total_steps < self._policy.n_warmup:
				action = self._env.action_space.sample()
			else:
				action = self._policy.get_action(obs)

			# if self.action_mode == "patial" or self.action_mode == "residual":
			# 	if len(list(self.param_opt)) == 2:
			# 		self.step_param = self.param_opt
			# 	elif len(list(self.param_opt)) == 4:
			# 		assert len(self.step_param) == 2
			# 		pd_ver = 2
			# 		if pd_ver == 1:
			# 			self.step_param[0] = max(self.param_opt[0] + self.param_opt[1] * p_error + self.param_opt[2] * d_error, 0.05)
			# 			self.step_param[1] = self.param_opt[0] * self.param_opt[3]
			# 		if pd_ver == 2:
			# 			self.step_param[1] = max(self.param_opt[0] + self.param_opt[1] * p_error + self.param_opt[2] * d_error, 0.01)
			# 			self.step_param[0] = self.param_opt[1] * self.param_opt[3]
			# 	next_obs, reward, done, info = self._env.step(action, param=self.step_param)
			# elif self.action_mode == "whole":
			# 	next_obs, reward, done, info = self._env.step(action)

			if self.bo:
				param_opt_to_env = self.param_opt
			else:
				param_opt_to_env = None

			# timer12 = time.time()
			next_obs, reward, done, info = self._env.step(action, param_opt=param_opt_to_env)
			# print(f"STEPPING TAKES {time.time() - timer12} SEC")

			p_error, d_error = info["p_error"], info["d_error"]

			if self._show_progress:
				self._env.render()
			episode_steps += 1
			episode_return += reward
			total_steps += 1
			self.global_total_steps += 1
			tf.summary.experimental.set_step(total_steps)

			done_flag = done
			if (hasattr(self._env, "_max_episode_steps") and
				episode_steps == self._env._max_episode_steps):
				done_flag = False
			replay_buffer.add(obs=obs, act=action,
							  next_obs=next_obs, rew=reward, done=done_flag)
			obs = next_obs

			if done or episode_steps == self._episode_max_steps:

				timer1 = time.time()

				replay_buffer.on_episode_end()
				obs = self._env.reset()

				n_episode += 1
				fps = episode_steps / (time.perf_counter() - episode_start_time)
				self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
					n_episode, total_steps, episode_steps, episode_return, fps))
				tf.summary.scalar(name="Common/training_return", data=episode_return)
				tf.summary.scalar(name="Common/training_episode_length", data=episode_steps)
				wandb.log({"Training Return": episode_return, "Episode Steps": episode_steps, "FPS": fps}, step=self.global_total_steps)
				# wandb.log({"Training_Return_Global": episode_return, "Episode_Steps_Global": episode_steps, "FPS_Global": fps}, step=global_total_steps)

				if self.bo:
					self.returns.append(episode_return)

				episode_steps = 0
				episode_return = 0
				episode_start_time = time.perf_counter()

				if n_episode % self.param_update_interval_epi == 0 and self.bo and self.param_update_interval_epi != 0 and not self.bo_lock:
					# if len(self.returns) != 1:
					#     fit = np.percentile(self.returns[1:], 90)
					# else:
					#     fit = np.percentile(self.returns, 90)

					if self.rotation == False:
						if self.fitting == "training":
							fit = np.mean(self.returns)
						elif self.fitting == "evaluation":
							fit, _ = self.evaluate_policy(total_steps, record_step=True)
						else:
							raise ValueError('FITTING MODE NOT FOUND')


						if fit > self.best_y:
							self.best_x, self.best_y = [i for i in self.param_opt], fit
						# if len(self.param_opt) == 2:
						# 	self.logger.info("x: [{0:.4f}, {1:.4f}]  y: {2:.4f} Optimal Value: {3:.4f} Optimal Point: [{4:.4f}, {5:.4f}]".format(self.param_opt[0], self.param_opt[1], fit, self.best_y, self.best_x[0], self.best_x[1]) )
						# if len(self.param_opt) == 4:
						# 	# print(self.best_x)
						# 	self.logger.info("X: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]  Y: {:.4f} ".format(self.param_opt[0], self.param_opt[1], self.param_opt[2], self.param_opt[3], fit) )
						# 	self.logger.info("Optimal Point: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]  Optimal Value: {:.4f}".format(self.best_x[0], self.best_x[1], self.best_x[2], self.best_x[3], self.best_y) )

						self.logger.info("X: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]  Y: {:.4f} ".format(self.param_opt[0], self.param_opt[1], self.param_opt[2], self.param_opt[3], self.param_opt[4], self.param_opt[5], self.param_opt[6], self.param_opt[7], self.param_opt[8], fit) )
						self.logger.info("Optimal Point: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]  Optimal Value: {:.4f}".format(self.best_x[0], self.best_x[1], self.best_x[2], self.best_x[3], self.best_x[4], self.best_x[5], self.best_x[6], self.best_x[7], self.best_x[8], self.best_y) )

						if hasattr(self, 'ng_param_opt'):
							self.optimiser.tell(self.ng_param_opt, -fit)
							self.ng_param_opt = self.optimiser.ask()
							self.param_opt_masked = list(self.ng_param_opt.value[0])
							j = 0
							for i in range(len(self.optimisation_mask)):
								if self.optimisation_mask[i] == '1':
									self.param_opt[i] = self.param_opt_masked[j]
									j+=1

						else:
							self.optimiser.tell([(self.param_opt, fit)])
							self.param_opt = self.optimiser.ask()
						self.returns = []
						opt_steps += 1

						if self.DEBUG:
							print("[DEBUG] PARAM FROM OPTIMISER: ", self.param_opt)

						for i, x in enumerate(self.param_opt):
							wandb.log({f"x_{i}": x}, step=self.global_total_steps)
						for i, x in enumerate(self.best_x):
							wandb.log({f"Optimal x_{i}": x}, step=self.global_total_steps)
						wandb.log({f"Fitting": fit, "Optimal Fitting":self.best_y}, step=self.global_total_steps)

					else:
						# print("PARAMETERS OPTIMISATION IS RUNNING INDIVIDUALLY!")
						if self.best_x == None:
							assert self.param_opt == self.param_opt
							fit, _ = self.evaluate_policy(total_steps, record_step=True)
							self.best_x = [i for i in self.param_opt]
							self.best_y = fit

						if hasattr(self, 'ng_param_opt'):
							self.param_opt = list(self.ng_param_opt.value[0])
						else:
							self.param_opt = self.df_param_opt


						for _ in range(self.param_update_interval_epi):

							fit, _ = self.evaluate_policy(total_steps, record_step=True)

							if fit > self.best_y:
								self.best_x, self.best_y = [i for i in self.param_opt], fit
							if len(self.param_opt) == 2:
								self.logger.info("x: [{0:.4f}, {1:.4f}]  y: {2:.4f} Optimal Value: {3:.4f} Optimal Point: [{4:.4f}, {5:.4f}]".format(self.param_opt[0], self.param_opt[1], fit, self.best_y, self.best_x[0], self.best_x[1]) )
							if len(self.param_opt) == 4:
								# print(self.best_x)
								self.logger.info("X: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]  Y: {:.4f} ".format(self.param_opt[0], self.param_opt[1], self.param_opt[2], self.param_opt[3], fit) )
								self.logger.info("Optimal Point: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]  Optimal Value: {:.4f}".format(self.best_x[0], self.best_x[1], self.best_x[2], self.best_x[3], self.best_y) )
							if hasattr(self, 'ng_param_opt'):
								self.optimiser.tell(self.ng_param_opt, -fit)
								self.ng_param_opt = self.optimiser.ask()
								self.param_opt = list(self.ng_param_opt.value[0])
							else:
								self.optimiser.tell([(self.param_opt, fit)])
								self.param_opt = self.optimiser.ask()

							opt_steps += 1

							for i, x in enumerate(self.param_opt):
								wandb.log({f"x_{i}": x}, step=self.global_total_steps)
							for i, x in enumerate(self.best_x):
								wandb.log({f"Optimal x_{i}": x}, step=self.global_total_steps)
							wandb.log({f"Fitting": fit, "Optimal Fitting":self.best_y}, step=self.global_total_steps)


						self.param_opt = self.best_x
							

				if self.bo_lock and n_episode == self.warmup_epi and self.bo:
					self.logger.info("PARAMETERS OPTIMISATION BEGINS!")
					# TIME FOR STARTING BO OPTIMISATION
					self.bo_lock = False
					# assert len(self.param_opt) == 4

					self._setup_optimiser()

					if self.rotation:
						self.rl_lock = True

				# print("PROCESSING AFTER AN EPISODE TAKES {:.2f} SEC".format(time.time() - timer1))

				   

			# if (not self.param_update_interval == 0) and total_steps % self.param_update_interval == 0 and self.bo:
			# 	if len(self.returns) != 1:
			# 		fit = np.percentile(self.returns[1:], 90)
			# 	else:
			# 		fit = np.percentile(self.returns, 90)

			# 	if fit > self.best_y:
			# 		self.best_x, self.best_y = self.param_opt, fit
			# 	self.logger.info("[Optimiser] x: [{0:.4f}, {1:.4f}]  y: {2:.4f} Optimal Value: {3:.4f} Optimal Point: [{4:.4f}, {5:.4f}]".format(self.param_opt[0], self.param_opt[1], fit, self.best_y, self.best_x[0], self.best_x[1]) )
			# 	self.optimiser.tell([(self.param_opt, fit)])
			# 	self.returns = []
			# 	self.param_opt = self.optimiser.ask()


			if total_steps < self._policy.n_warmup:

				continue

			if total_steps % self._policy.update_interval == 0:
				samples = replay_buffer.sample(self._policy.batch_size)
				with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
					self._policy.train(
						samples["obs"], samples["act"], samples["next_obs"],
						samples["rew"], np.array(samples["done"], dtype=np.float32),
						None if not self._use_prioritized_rb else samples["weights"])
				if self._use_prioritized_rb:
					td_error = self._policy.compute_td_error(
						samples["obs"], samples["act"], samples["next_obs"],
						samples["rew"], np.array(samples["done"], dtype=np.float32))
					replay_buffer.update_priorities(
						samples["indexes"], np.abs(td_error) + 1e-6)

			

			if total_steps % self._test_interval == 0:
				avg_test_return, avg_test_steps = self.evaluate_policy(total_steps, use_best_x=not self.eval_using_online_param)
				self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
					total_steps, avg_test_return, self._test_episodes))
				tf.summary.scalar(
					name="Common/average_test_return", data=avg_test_return)
				tf.summary.scalar(
					name="Common/average_test_episode_length", data=avg_test_steps)
				tf.summary.scalar(name="Common/fps", data=fps)

				wandb.log({f"Evaluation Return": avg_test_return, "Evaluation Episode Length": avg_test_steps}, step=self.global_total_steps)

				if avg_test_return > self.best_evaluation:
					self.best_evaluation = avg_test_return
					self.best_checkpoint_manager.save()
					with open(os.path.join(self._output_dir, "best_models", "log.txt"), 'a') as log_file:
						log_file.write(f"Total Steps: {total_steps}    Global Total Step: {self.global_total_steps}    Best Evaluation Return: {avg_test_return}    Evaluation Episode Length: {avg_test_steps}    Parameters: {self.best_x if self.best_x is not None else self.param_opt}")
						log_file.write('\n')
					if self.bo:
						np.save(os.path.join(self._output_dir, "best_models", "optimal_parameters"), self.best_x if self.best_x is not None else self.param_opt)


			if total_steps % self._save_model_interval == 0:
				self.checkpoint_manager.save()
				if self.bo:
					np.save(os.path.join(self._output_dir, "optimal_parameters"), self.best_x if self.best_x is not None else self.param_opt)
				if self.profiler_enable:
					if profiling:
						tf.profiler.experimental.stop()
						profiling = False

				
				# with open(os.path.join(self._output_dir, 'param_optimiser.pkl'), 'wb') as outp:
				#     pickle.dump(self.optimiser, outp, pickle.HIGHEST_PROTOCOL)


		tf.summary.flush()
		#tf.profiler.experimental.stop()

		os.rename(self._output_dir, self._output_dir + "F")

	def _setup_optimiser(self):
		# if self.optimiser_name == "BO":
		# 	# this is not supported any more
		# 	assert len(self.param_opt) == 4
		# 	def prior_mean_4d(x):
		# 		y1 = -(x[0] - self.param_opt[0])**2
		# 		y2 = -(x[1] - self.param_opt[1])**2
		# 		y3 = -(x[2] - self.param_opt[2])**2
		# 		y4 = -(x[3] - self.param_opt[3])**2
		# 		return y1 + y2 + y3 + y4 + self.best_y

		# 	options = argparse.Namespace(rand_exp_sampling_replace = True, gp_prior_mean = prior_mean_4d, 
		# 								 progress_load_from_and_save_to=os.path.join(self._output_dir, 'progress.p'), progress_save_every = 10)
		# 	self.optimiser = gp_bandit.EuclideanGPBandit(self.func_caller, ask_tell_mode=True, options=options)
		# 	self.optimiser.initialise()
		# 	self.df_param_opt = self.optimiser.ask()
		# 	self.param_opt = self.df_param_opt

		# elif self.optimiser_name == "TBPSA":

		# 	scalars = [ng.p.Scalar(init=i, lower=r[0], upper=r[1]) for i, r in zip(self.param_opt_masked, self.domains_masked)]
		# 	instrum = ng.p.Instrumentation(*scalars)
		# 	self.optimiser = ng.optimizers.TBPSA(parametrization=instrum, budget=self._max_steps/500/2, num_workers=1)

		# 	self.ng_param_opt = self.optimiser.ask()
		# 	self.param_opt_masked = list(self.ng_param_opt.value[0])
		# 	j = 0
		# 	for i in range(len(self.optimisation_mask)):
		# 		if self.optimisation_mask[i] == '1':
		# 			self.param_opt[i] = self.param_opt_masked[j]
		# 			j+=1


		self.budget = self._max_steps/500/2
		self.num_workers = 1

		scalars = [ng.p.Scalar(init=i, lower=r[0], upper=r[1]) for i, r in zip(self.param_opt_masked, self.domains_masked)]
		instrum = ng.p.Instrumentation(*scalars)
		if self.optimiser_name == "auto":
			self.optimiser = ng.optimizers.NGOpt(parametrization=instrum, budget=self.budget, num_workers=self.num_workers)
			print(self.optimiser._select_optimizer_cls(), "IS USED FOR OPTIMISATION!")
			self.config["optimisation/optimiser"] = str(self.optimizer._select_optimizer_cls())
		elif self.optimiser_name == "CMA":
			self.optimiser = ng.optimizers.CMA(parametrization=instrum, budget=self.budget, num_workers=self.num_workers)
		elif self.optimiser_name == "BO":
			self.optimiser = ng.optimizers.BO(parametrization=instrum, budget=self.budget, num_workers=self.num_workers)
		elif self.optimiser_name == "TBPSA":
			self.optimiser = ng.optimizers.TBPSA(parametrization=instrum, budget=self.budget, num_workers=self.num_workers)
		elif self.optimiser_name == "METACMA":
			self.optimiser = ng.optimizers.MetaModel(parametrization=instrum, budget=self.budget, num_workers=self.num_workers)

		self.ng_param_opt = self.optimiser.ask()
		self.param_opt_masked = list(self.ng_param_opt.value[0])
		j = 0
		for i in range(len(self.optimisation_mask)):
			if self.optimisation_mask[i] == '1':
				self.param_opt[i] = self.param_opt_masked[j]
				j+=1

		# if self.rotation:
		# 	self.param_opt = self.param_opt

	def evaluate_policy_continuously(self):
		"""
		Periodically search the latest checkpoint, and keep evaluating with the latest model until user kills process.
		"""
		if self._model_dir is None:
			self.logger.error("Please specify model directory by passing command line argument `--model-dir`")
			exit(-1)

		if self.param_opt_testing is not None:
			self.best_x = self.param_opt = self.param_opt_testing
		elif os.path.isfile(os.path.join(self._model_dir, "optimal_parameters.npy")):
			self.best_x = self.param_opt = list(np.load(os.path.join(self._model_dir, "optimal_parameters.npy")))
			self.logger.info("Restored {}".format(os.path.join(self._model_dir, "optimal_parameters.npy")))
			print("========================================================")
			print("Restored Optimised Parameters: ")
			print(self.param_opt)
			print("========================================================")

			# self.best_x = [0.0793, 0.0565, 0.0687, 5.4652]  # TEMP  # TODO!!!
		else:
			self.best_x = self.param_opt = self._test_env.param_opt

		# self.evaluate_policy(total_steps=0)
		while True:
			latest_path_ckpt = tf.train.latest_checkpoint(self._model_dir)
			if self._latest_path_ckpt != latest_path_ckpt:
				self._latest_path_ckpt = latest_path_ckpt
				self._checkpoint.restore(self._latest_path_ckpt)
				self.logger.info("Restored {}".format(self._latest_path_ckpt))
			

			self.evaluate_policy(total_steps=0, use_best_x=True)

	def evaluate_policy(self, total_steps, record_step=False, use_best_x=False):
		tf.summary.experimental.set_step(total_steps)
		if self._normalize_obs:
			self._test_env.normalizer.set_params(
				*self._env.normalizer.get_params())
		avg_test_return = 0.
		avg_test_steps = 0
		if self._save_test_path:
			replay_buffer = get_replay_buffer(
				self._policy, self._test_env, size=self._episode_max_steps)


		if self._model_dir is not None:
			param_opt_to_env = self.param_opt
			assert self.param_opt == self.best_x
		elif not self.bo:
			param_opt_to_env = None
		else:
			if use_best_x and self.best_x is not None:
				param_opt_to_env = self.best_x
			else:
				param_opt_to_env = self.param_opt

		for i in range(self._test_episodes):
			episode_return = 0.
			frames = []
			obs = self._test_env.reset()
			# print("[DEBUG] RESETTING")
			avg_test_steps += 1
			# self.step_param = [self.param_opt[0] * self.param_opt[3], self.param_opt[0]]
			p_error = 0
			d_error = 0
			n_step = 0
			# print("OMEGA_PITCH: ", self._test_env.get_full_state()[10], "  OR  ", obs[4])
			for _ in range(self._episode_max_steps):
				action = self._policy.get_action(obs, test=True)
				# print(self._policy.get_action(np.array([0.5]*14), test=True))
				if self.action_mode == "patial" or self.action_mode == "residual":

					# if len(list(param_opt)) == 2:
					# 	self.step_param = param_opt
					# elif len(list(param_opt)) == 4:
					# 	pd_ver = 2
					# 	if pd_ver == 1:
					# 		self.step_param[0] = max(param_opt[0] + param_opt[1] * p_error + param_opt[2] * d_error, 0.05)
					# 		self.step_param[1] = param_opt[0] * param_opt[3]
					# 	elif pd_ver == 2:
					# 		self.step_param[1] = max(param_opt[0] + param_opt[1] * p_error + param_opt[2] * d_error, 0.01)
					# 		self.step_param[0] = param_opt[1] * param_opt[3]
					if self.DEBUG:
						print("[DEBUG] PARAM TO ENV: ", param_opt_to_env)

					next_obs, reward, done_, info = self._test_env.step(action, param_opt=param_opt_to_env)
					# print("P ERRPR: ", info["p_error"])
					# print(self._env.done)
					n_step += 1
				
					p_error = info["p_error"]
					d_error = info["d_error"]
					# print(f"e: {p_error}, e_dot: {d_error}  -->  u: {self.step_param[1]} - {self.param_opt[0]} = {self.step_param[1] - self.param_opt[0]}")

				elif self.action_mode == "whole":
					next_obs, reward, done_, info = self._test_env.step(action)

				avg_test_steps += 1
				if record_step:
					self.global_total_steps += 1
				if self._save_test_path:
					replay_buffer.add(obs=obs, act=action,
									  next_obs=next_obs, rew=reward, done=done_)

				if self._save_test_movie:
					frames.append(self._test_env.render(mode='rgb_array'))
				elif self._show_test_progress:
					self._test_env.render()
				episode_return += reward
				obs = next_obs
				if done_:
					break
			# print("LENGTH OF TEST EPISODE: ", n_step, "  PARAMS: ", self.param_opt)
			prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(
				total_steps, i, episode_return)
			if self._save_test_path:
				save_path(replay_buffer._encode_sample(np.arange(self._episode_max_steps)),
						  os.path.join(self._output_dir, prefix + ".pkl"))
				replay_buffer.clear()
			if self._save_test_movie:
				frames_to_gif(frames, prefix, self._output_dir)
			avg_test_return += episode_return
		if self._show_test_images:
			images = tf.cast(
				tf.expand_dims(np.array(obs).transpose(2, 0, 1), axis=3),
				tf.uint8)
			tf.summary.image('train/input_img', images,)
		return avg_test_return / self._test_episodes, avg_test_steps / self._test_episodes

	def _set_from_args(self, args):
		# experiment settings
		self._max_steps = args.max_steps
		self._episode_max_steps = (args.episode_max_steps
								   if args.episode_max_steps is not None
								   else args.max_steps)
		self._n_experiments = args.n_experiments
		self._show_progress = args.show_progress
		self._save_model_interval = args.save_model_interval
		self._save_summary_interval = args.save_summary_interval
		self._normalize_obs = args.normalize_obs
		self._logdir = args.logdir
		self._model_dir = args.model_dir
		# replay buffer
		self._use_prioritized_rb = args.use_prioritized_rb
		self._use_nstep_rb = args.use_nstep_rb
		self._n_step = args.n_step
		# test settings
		self._evaluate = args.evaluate
		self._test_interval = args.test_interval
		self._show_test_progress = args.show_test_progress
		self._test_episodes = args.test_episodes
		self._save_test_path = args.save_test_path
		self._save_test_movie = args.save_test_movie
		self._show_test_images = args.show_test_images

	@staticmethod
	def get_argument(parser=None):
		if parser is None:
			parser = argparse.ArgumentParser(conflict_handler='resolve')
		# experiment settings
		parser.add_argument('--max-steps', type=int, default=int(1e6),
							help='Maximum number steps to interact with env.')
		parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
							help='Maximum steps in an episode')
		parser.add_argument('--n-experiments', type=int, default=1,
							help='Number of experiments')
		parser.add_argument('--show-progress', action='store_true',
							help='Call `render` in training process')
		parser.add_argument('--save-model-interval', type=int, default=int(1e4),
							help='Interval to save model')
		parser.add_argument('--save-summary-interval', type=int, default=int(1e3),
							help='Interval to save summary')
		parser.add_argument('--model-dir', type=str, default=None,
							help='Directory to restore model')
		parser.add_argument('--dir-suffix', type=str, default='',
							help='Suffix for directory that contains results')
		parser.add_argument('--normalize-obs', action='store_true',
							help='Normalize observation')
		parser.add_argument('--logdir', type=str, default='results',
							help='Output directory')
		# test settings
		parser.add_argument('--evaluate', action='store_true',
							help='Evaluate trained model')
		parser.add_argument('--test-interval', type=int, default=int(1e4),
							help='Interval to evaluate trained model')
		parser.add_argument('--show-test-progress', action='store_true',
							help='Call `render` in evaluation process')
		parser.add_argument('--test-episodes', type=int, default=5,
							help='Number of episodes to evaluate at once')
		parser.add_argument('--save-test-path', action='store_true',
							help='Save trajectories of evaluation')
		parser.add_argument('--show-test-images', action='store_true',
							help='Show input images to neural networks when an episode finishes')
		parser.add_argument('--save-test-movie', action='store_true',
							help='Save rendering results')
		# replay buffer
		parser.add_argument('--use-prioritized-rb', action='store_true',
							help='Flag to use prioritized experience replay')
		parser.add_argument('--use-nstep-rb', action='store_true',
							help='Flag to use nstep experience replay')
		parser.add_argument('--n-step', type=int, default=4,
							help='Number of steps to look over')
		# others
		parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'WARNING'],
							default='INFO', help='Logging level')
		return parser
