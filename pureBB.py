import os
import sys
import nevergrad as ng
import wandb
import time
import numpy as np
import argparse



class BBTrainer(object):
	def __init__(self, env, args, test_env, param_opt, param_domain, optimisation_mask, optimiser, max_steps=3e6, eval_using_online_param=False):
		self._env = env
		self.budget = 1e5
		self.num_workers = 1
		self.domains = param_domain
		self.optimisation_mask = optimisation_mask
		self.optimiser_name = optimiser
		assert len(self.optimisation_mask) == 9
		self.param_opt = param_opt
		self.max_steps = max_steps
		self.eval_using_online_param = eval_using_online_param

		self._test_interval = args.test_interval
		self._test_episodes = args.test_episodes

		self.param_opt_masked = [self.param_opt[i] for i in range(len(self.optimisation_mask)) if self.optimisation_mask[i] == '1']
		self.domains_masked = [self.domains[i] for i in range(len(self.optimisation_mask)) if self.optimisation_mask[i] == '1']


		self._test_env = self._env if test_env is None else test_env

		print("optimisation_mask: ", self.optimisation_mask)
		print("param_opt_masked: ", self.param_opt_masked)
		print("domains_masked: ", np.array(self.domains_masked))

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

		self.x_list = []
		self.y_list = []
		self.best_x_list = []
		self.best_y_list = []
		self.best_x, self.best_y = None, float('-inf')

		log_dir = "results"
		if not os.path.isdir(log_dir) :
			os.mkdir(log_dir)
		run_i = 0
		while (os.path.exists(os.path.join(log_dir, "PBB_{}{:03d}".format(int(time.strftime("%Y%m%d", time.localtime())), run_i))) or 
			   os.path.exists(os.path.join(log_dir, "PBB_{}{:03d}F".format(int(time.strftime("%Y%m%d", time.localtime())), run_i))) or
			   os.path.exists(os.path.join(log_dir, "PBB_{}{:03d}T".format(int(time.strftime("%Y%m%d", time.localtime())), run_i)))):
			run_i += 1
		self.log_dir = os.path.join(log_dir, "PBB_{}{:03d}".format(int(time.strftime("%Y%m%d", time.localtime())), run_i))
		if not os.path.isdir(self.log_dir) :
			os.mkdir(self.log_dir)
		self._output_dir = self.log_dir
		# config = {"optimiser": OPTIMISER, "initial_guess": self.best_guess, "domains": self.domains, "budget": BUDGET, "log_directory": self.log_dir, "PD_ver": PD_VER}
		# for conf in DYN_CONFIG:
		# 	config["dynamics/"+conf] = DYN_CONFIG[conf]
		# wandb.init(config=config, project="Standing Cheetah Lite", name=f"Pure{OPTIMISER}[Ver{PD_VER}]", dir=os.path.abspath(os.path.join(self.log_dir, "wandb")), mode="online" if WANDB else "disabled")
		self._policy = self.DummyPolicy()



	def __call__(self):

		total_steps = 0
		n_episode = 0
		start_time = time.time()

		while total_steps < self.max_steps:
			self.ng_param_opt = self.optimiser.ask()

			self.param_opt_masked = list(self.ng_param_opt.value[0])
			j = 0
			for i in range(len(self.optimisation_mask)):
				if self.optimisation_mask[i] == '1':
					self.param_opt[i] = self.param_opt_masked[j]
					j+=1

			self.x_list.append(self.param_opt)

			done = False
			p_error = 0
			d_error = 0
			self._env.reset()
			episode_return = 0


			while not done:

				s, reward, done, info = self._env.step([0]*6, param_opt=self.param_opt)
				episode_return += reward
				total_steps += 1

				if total_steps % self._test_interval == 0:
					avg_test_return, avg_test_steps = self.evaluate_policy()
					print("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
						total_steps, avg_test_return, self._test_episodes))

					wandb.log({f"Evaluation Return": avg_test_return, "Evaluation Episode Length": avg_test_steps}, step=total_steps)



			if episode_return > self.best_y:
				self.best_x, self.best_y = [i for i in self.param_opt], episode_return

			hours, rem = divmod(time.time()-start_time, 3600)
			minutes, seconds = divmod(rem, 60)

			print("EPISODE: {}  (STEP: {})  X: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]  Y: {:.4f} ".format(n_episode, total_steps, self.param_opt[0], self.param_opt[1], self.param_opt[2], self.param_opt[3], self.param_opt[4], self.param_opt[5], self.param_opt[6],self.param_opt[7],self.param_opt[8], episode_return) )
			print("Optimal Point: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]  Optimal Value: {:.4f}  ELAPSE: {:0>2}:{:0>2}:{:05.2f}".format(self.best_x[0], self.best_x[1], self.best_x[2], self.best_x[3], self.best_x[4],self.best_x[5],self.best_x[6],self.best_x[7],self.best_x[8],self.best_y, int(hours),int(minutes),seconds) )
			self.y_list.append(episode_return)
			self.best_x_list.append(self.best_x)
			self.best_y_list.append(self.best_y)
			loss = -episode_return
			self.optimiser.tell(self.ng_param_opt, loss)
			n_episode += 1

			for i, x in enumerate(self.param_opt):
				wandb.log({f"x_{i}": x}, step=total_steps)
			for i, x in enumerate(self.best_x):
				wandb.log({f"Optimal x_{i}": x}, step=total_steps)
			wandb.log({f"Fitting": episode_return, "Optimal Fitting":self.best_y}, step=total_steps)


			

			if n_episode % 200 == 0:
				self._save()


	def evaluate_policy(self):

		total_return = 0
		total_step = 0

		for _ in range(self._test_episodes):
			self._test_env.reset()
			done = False

			while not done:
				s, reward, done, info = self._test_env.step([0]*6, param_opt = self.param_opt if self.eval_using_online_param else self.best_x)
				total_return += reward
				total_step += 1

		return total_return / self._test_episodes, total_step / self._test_episodes


	def _save(self):
		np.save(os.path.join(self.log_dir, "x_list"), self.x_list)
		np.save(os.path.join(self.log_dir, "y_list"), self.y_list)
		np.save(os.path.join(self.log_dir, "best_x_list"), self.best_x_list)
		np.save(os.path.join(self.log_dir, "best_y_list"), self.best_y_list)
		np.save(os.path.join(self.log_dir, "optimal_parameters"), self.best_x if self.best_x is not None else self.param_opt)


	class DummyPolicy():
		def __init__(self):
			self.policy_name = "None"

