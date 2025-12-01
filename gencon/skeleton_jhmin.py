import copy
import numpy as np

class Belief :
	def __init__(self, params) :
	def update(self, stimuli) :
		return copy.deepcopy(self)
	def generate_response(self) :

class State :
	def __init__(self, params) :
		self.is_finished = False
	def update(self, response) :
		return copy.deepcopy(self)
	def generate_stimulus(self) :

def run_episode(init_state, init_belief) :
	state, belief = init_state, init_belief
	states, stimuli, beliefs, responses = [], [], [], []
	
	while not state.is_finished :
		state, stimulus, belief, response = run_trial(state, belief)
		states.append(state)
		stimuli.append(stimulus)
		beliefs.append(belief)
		responses.append(response)
	
	return states, stimuli, beliefs, responses

def run_trial(state, belief, stimulus=None, response=None) :
	if (not stimulus and not response) : #guided
		#stimulus = state.generate_stimulus()
		updated_belief = belief.update(stimulus)
		next_response = belief.generate_response()
		#next_state = state.update(next_response)
		next_state = state.update(response)
		
		return next_state, updated_belief, next_response
	else :
		stimulus = state.generate_stimulus()
		updated_belief = belief.update(stimulus)
		next_response = belief.generate_response()
		next_state = state.update(next_response)
		
		return next_state, stimulus, updated_belief, next_response

def guided_run_episode(init_state, init_belief, stimuli, responses) :
	state, belief = init_state, init_belief
	states, beliefs, recovered_responses = [], [], []
	t = 0

	while not state.is_finished :
		state, belief, recovered_response = run_trial(state, belief, stimuli[t], response[t])
		states.append(state)
		beliefs.append(belief)
		recovered_responses.append(recovered_response)
		t += 1
	
	return states, beliefs, responses



### 25-10-29 ###
def Loss(original_responses, beliefs) :
	pass

def optimize(simulate, loss_function, truth_reward_distribution, param_init, param_condition) :
	pass
	return params, loss

def optimize_multiple_inputs(simulate, loss, truth_reward_distribution, params_init_list, params_condition) :
	pass
	return params, loss

# bound -> list of possible params
# list of ground truth params
def parameter_recovery(simulation_function, loss, truth_param_list, param_init, param_condition, return_distribution = False) :
	"""
	Input : 
		simulate 			: function gets param, and outputs P(r) list
		loss 				: function calculating loss from distributions of rewards (e.g. cross entropy loss)
		truth_param_list 	: parameter list that we want to recover
		param_init 			: initial parameter value that we starts to fit (it would be list)
		param_condition 	: 1) set of possible values of parameters 2) parameter bound
		option				: whether or not you want to get the behavior of recovered data
	Output : 
		if option ->
			recovered_param_list 			: recovered parameter values
			loss_list 						: loss of recovered parameter 
			recovered_reward_distributions 	: behavior of recovered parameter (here, we mean P(r) list)
		else ->
			recovered_param_list
			loss_list 
	"""
	truth_reward_distributions = [simulation_function(truth_param) for truth_param in truth_param_list]
	recovered_param_list, recovered_loss_list = [], []

	for truth_reward_distribution in truth_reward_distributions :
		"""each truth_reward_distribution means [P(r_1), P(r_2), ...]"""

		recovered_param, recovered_loss = optimize(simulation_function, loss, truth_reward_distribution, param_init, param_condition)
		#recovered_param, recovered_loss = optimize_multiple_inputs(objective, loss, truth_result, param_init, param_condition)
		
		recovered_param_list.append(recovered_param)
		recovered_loss_list.append(recovered_loss)

	if return_distribution :
		recovered_reward_distributions = [simulation_function(recovered_param) for recovered_param in recovered_param_list]
		return recovered_param_list, recovered_loss_list, recovered_reward_distributions
	else :
		return recovered_param_list, recovered_loss_list


def model_recovery(simulation_ftns, loss, truth_param_lists, param_init_list, param_condition_list, return_distribution = False) :
	recovery_result, recovered_distribution_lists_list = []

	for simulation_ftn, truth_param_list in zip(simulation_ftns, truth_param_lists) :
		recovered_param_lists, loss_lists = [], []

		for param_init, param_condition in zip(param_init_list, param_condition_list) :
			recovered_param_list, loss_list = parameter_recovery(simulation_ftn, loss, truth_param_list, param_init, param_condition)

			recovered_param_lists.append(recovered_param_list)
			loss_lists.append(loss_list)
		
		transposed_matrix = np.array(loss_lists).T 
		min_loss_indices = np.argmin(transposed_matrix, axis=1).tolist()
		recovery_result.append(np.bincount(min_loss_indices))
	
	if return_distribution :
		return recovery_result, recovered_distribution_lists_list
	else : 
		return recovery_result
		