import os
import time

import numpy as np
import tensorflow as tf

from cpprb import ReplayBuffer

from trainer import Trainer
from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer, get_default_rb_dict
from tf2rl.misc.discount_cumsum import discount_cumsum
from tf2rl.envs.utils import is_discrete

import nevergrad as ng
import wandb


class OnPolicyTrainer(Trainer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		assert self.warmup_epi >=0 # BBF MODE IS NOT IMPLEMENTED YET FOR ON POLICY

	def __call__(self):
		# Prepare buffer
		self.replay_buffer = get_replay_buffer(
			self._policy, self._env)
		kwargs_local_buf = get_default_rb_dict(
			size=self._policy.horizon, env=self._env)
		kwargs_local_buf["env_dict"]["logp"] = {}
		kwargs_local_buf["env_dict"]["val"] = {}
		if is_discrete(self._env.action_space):
			kwargs_local_buf["env_dict"]["act"]["dtype"] = np.int32
		self.local_buffer = ReplayBuffer(**kwargs_local_buf)

		episode_steps = 0
		episode_return = 0
		episode_start_time = time.time()
		total_steps = np.array(0, dtype=np.int32)
		self.global_total_steps = 0
		opt_steps = 0
		n_episode = 0
		obs = self._env.reset()

		tf.summary.experimental.set_step(total_steps)
		while total_steps < self._max_steps:
			# Collect samples
			for _ in range(self._policy.horizon):
				if self._normalize_obs:
					obs = self._obs_normalizer(obs, update=False)
				act, logp, val = self._policy.get_action_and_val(obs)
				if not is_discrete(self._env.action_space):
					env_act = np.clip(act, self._env.action_space.low, self._env.action_space.high)
				else:
					env_act = act

				if self.bo:
					param_opt_to_env = self.param_opt
				else:
					param_opt_to_env = None

				next_obs, reward, done, _ = self._env.step(env_act, param_opt=param_opt_to_env)
				if self._show_progress:
					self._env.render()

				episode_steps += 1
				total_steps += 1
				self.global_total_steps += 1  # useful for rotation mode
				episode_return += reward

				done_flag = done
				if (hasattr(self._env, "_max_episode_steps") and
					episode_steps == self._env._max_episode_steps):
					done_flag = False
				self.local_buffer.add(
					obs=obs, act=act, next_obs=next_obs,
					rew=reward, done=done_flag, logp=logp, val=val)
				obs = next_obs

				if done or episode_steps == self._episode_max_steps:
					tf.summary.experimental.set_step(total_steps)
					self.finish_horizon()
					obs = self._env.reset()
					n_episode += 1
					fps = episode_steps / (time.time() - episode_start_time)
					self.logger.info(
						"Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
							n_episode, int(total_steps), episode_steps, episode_return, fps))
					tf.summary.scalar(name="Common/training_return", data=episode_return)
					tf.summary.scalar(name="Common/training_episode_length", data=episode_steps)
					tf.summary.scalar(name="Common/fps", data=fps)
					wandb.log({"Training Return": episode_return, "Episode Steps": episode_steps, "FPS": fps}, step=self.global_total_steps)
					
					if self.bo:
						self.returns.append(episode_return)

					episode_steps = 0
					episode_return = 0
					episode_start_time = time.time()


					if n_episode % self.param_update_interval_epi == 0 and self.bo and self.param_update_interval_epi != 0 and not self.bo_lock:

						if self.rotation == False:
							if self.fitting == "training":
								fit = np.mean(self.returns)
							elif self.fitting == "evaluation":
								fit, _ = self.evaluate_policy(total_steps, record_step=True)
							else:
								raise ValueError('FITTING MODE NOT FOUND')


							if fit > self.best_y:
								self.best_x, self.best_y = [i for i in self.param_opt], fit

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


				if total_steps % self._test_interval == 0:
					avg_test_return, avg_test_steps = self.evaluate_policy(total_steps, use_best_x=not self.eval_using_online_param)
					self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
						total_steps, avg_test_return, self._test_episodes))
					tf.summary.scalar(
						name="Common/average_test_return", data=avg_test_return)
					tf.summary.scalar(
						name="Common/average_test_episode_length", data=avg_test_steps)
					self.writer.flush()

					wandb.log({f"Evaluation Return": avg_test_return, "Evaluation Episode Length": avg_test_steps}, step=self.global_total_steps)

					if avg_test_return > self.best_evaluation:
						self.best_evaluation = avg_test_return
						self.param_opt_best_eval = [i for i in self.param_opt]

						self.best_checkpoint_manager.save()
						with open(os.path.join(self._output_dir, "best_models", "log.txt"), 'a') as log_file:
							log_file.write(f"Total Steps: {total_steps}    Global Total Step: {self.global_total_steps}    Best Evaluation Return: {avg_test_return}    Evaluation Episode Length: {avg_test_steps}    Parameters: {self.param_opt}    Best Parameters: {self.best_x if self.best_x is not None else self.param_opt}")
							log_file.write('\n')
						if self.bo:
							np.save(os.path.join(self._output_dir, "best_models", "optimal_parameters"), self.best_x if self.best_x is not None else self.param_opt)
							np.save(os.path.join(self._output_dir, "best_models", "optimal_parameters_online"), self.param_opt)

				if total_steps % self._save_model_interval == 0:
					self.checkpoint_manager.save()
					if self.bo:
						np.save(os.path.join(self._output_dir, "optimal_parameters"), self.best_x if self.best_x is not None else self.param_opt)
					if self.profiler_enable:
						if profiling:
							tf.profiler.experimental.stop()
							profiling = False

			self.finish_horizon(last_val=val)

			tf.summary.experimental.set_step(total_steps)

			# Train actor critic
			if self._policy.normalize_adv:
				samples = self.replay_buffer.get_all_transitions()
				mean_adv = np.mean(samples["adv"])
				std_adv = np.std(samples["adv"])
				# Update normalizer
				if self._normalize_obs:
					self._obs_normalizer.experience(samples["obs"])
			with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
				for _ in range(self._policy.n_epoch):
					samples = self.replay_buffer._encode_sample(
						np.random.permutation(self._policy.horizon))
					if self._normalize_obs:
						samples["obs"] = self._obs_normalizer(samples["obs"], update=False)
					if self._policy.normalize_adv:
						adv = (samples["adv"] - mean_adv) / (std_adv + 1e-8)
					else:
						adv = samples["adv"]
					for idx in range(int(self._policy.horizon / self._policy.batch_size)):
						target = slice(idx * self._policy.batch_size,
									   (idx + 1) * self._policy.batch_size)
						self._policy.train(
							states=samples["obs"][target],
							actions=samples["act"][target],
							advantages=adv[target],
							logp_olds=samples["logp"][target],
							returns=samples["ret"][target])

		tf.summary.flush()

		model_save_dir = os.path.join(self._output_dir, "DAM")
		if not os.path.isdir(model_save_dir):
			os.mkdir(model_save_dir)
			s_dim = self._env.observation_space.shape[0]
			self._policy.save_actor(model_save_dir, s_dim=s_dim) 
		if not os.path.isfile(os.path.join(model_save_dir, "param_opt.conf")):
			with open(os.path.join(model_save_dir, "param_opt.conf"),"a") as f:
				f.write(self._test_env.gait)
				f.write("\n")
				for p in self.param_opt:
					f.write(str(p))
					f.write("\n")
				f.write(f"The actor model is generated from {self._output_dir}")

		if os.path.isdir(os.path.join(self._output_dir, "best_models")):
			self._set_check_point(model_dir = os.path.join(self._output_dir, "best_models"))
			if not os.path.isfile(os.path.join(self._output_dir, "best_models", "DAM", "param_opt.conf")):
				with open(os.path.join(self._output_dir, "best_models", "DAM", "param_opt.conf"),"a") as f:
					f.write(self._test_env.gait)
					f.write("\n")
					for p in self.param_opt_best_eval:
						f.write(str(p))
						f.write("\n")
					f.write(f"The actor model is generated from {self._output_dir}")
					
		os.rename(self._output_dir, self._output_dir + "F")

	def finish_horizon(self, last_val=0):
		self.local_buffer.on_episode_end()
		samples = self.local_buffer._encode_sample(
			np.arange(self.local_buffer.get_stored_size()))
		rews = np.append(samples["rew"], last_val)
		vals = np.append(samples["val"], last_val)

		# GAE-Lambda advantage calculation
		deltas = rews[:-1] + self._policy.discount * vals[1:] - vals[:-1]
		if self._policy.enable_gae:
			advs = discount_cumsum(deltas, self._policy.discount * self._policy.lam)
		else:
			advs = deltas

		# Rewards-to-go, to be targets for the value function
		rets = discount_cumsum(rews, self._policy.discount)[:-1]
		self.replay_buffer.add(
			obs=samples["obs"], act=samples["act"], done=samples["done"],
			ret=rets, adv=advs, logp=np.squeeze(samples["logp"]))
		self.local_buffer.clear()

	def evaluate_policy(self, total_steps, use_best_x=False):
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
			avg_test_steps += 1
			for _ in range(self._episode_max_steps):
				if self._normalize_obs:
					obs = self._obs_normalizer(obs, update=False)
				act, _ = self._policy.get_action(obs, test=True)
				act = (act if is_discrete(self._env.action_space) else
					   np.clip(act, self._env.action_space.low, self._env.action_space.high))
				next_obs, reward, done, _ = self._test_env.step(act, param_opt=param_opt_to_env)
				avg_test_steps += 1
				if self._save_test_path:
					replay_buffer.add(
						obs=obs, act=act, next_obs=next_obs,
						rew=reward, done=done)

				if self._save_test_movie:
					frames.append(self._test_env.render(mode='rgb_array'))
				elif self._show_test_progress:
					self._test_env.render()
				episode_return += reward
				obs = next_obs
				if done:
					break
			prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(
				total_steps, i, episode_return)
			if self._save_test_path:
				save_path(replay_buffer.sample(self._episode_max_steps),
						  os.path.join(self._output_dir, prefix + ".pkl"))
				replay_buffer.clear()
			if self._save_test_movie:
				frames_to_gif(frames, prefix, self._output_dir)
			avg_test_return += episode_return
		if self._show_test_images:
			images = tf.cast(
				tf.expand_dims(np.array(obs).transpose(2, 0, 1), axis=3),
				tf.uint8)
			tf.summary.image('train/input_img', images, )
		return avg_test_return / self._test_episodes, avg_test_steps / self._test_episodes
