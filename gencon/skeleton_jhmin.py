import copy

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

def optimize(objective, loss_function, truth_param, param_init, param_condition) :
	pass
	return params, loss

def optimize_multiple_inputs(objective, loss, truth_param, params_init_list, params_condition) :
	pass
	return params, loss

# bound -> list of possible params
# list of ground truth params
def parameter_recovery(objective, loss, truth_param_list, param_init, param_condition, option = False) :
"""
	Input : 
		objective 			: function gets param and outputs some behavior
		loss 				: function calculating loss
		truth_param_list 	: parameter list that we want to recover
		param_init 			: initial parameter value that we starts to fit (it would be list)
		param_condition 	: 1) set of possible values of parameters 2) parameter bound
		option				: whether or not you want to get the behavior of recovered data
	Output : 
		if option ->
			recovered_param_list 	: recovered parameter values
			loss_list 				: loss of recovered parameter 
			recovered_result_list 	: behavior of recovered parameter
		else ->
			recovered_param_list
			loss_list 
"""
	truth_result_list = [objective(truth_param) for truth_param in truth_param_list]
	recovered_param_list, recovered_loss_list = [], []

	for truth_result in truth_result_list :
		recovered_param, recovered_loss = optimize(objective, loss, truth_result, param_init, param_condition)
		#recovered_param, recovered_loss = optimize_multiple_inputs(objective, loss, truth_result, param_init, param_condition)
		
		recovered_param_list.append(recovered_param)
		loss_list.append(recovered_loss)

	if option :
		recovered_result_list = [objective(recovered_param) for recovered_param in recovered_param_list]
		return recovered_param_list, loss_list, recovered_result_list
	else :
		return recovered_param_list, loss_list

def model_recovery(objective_list, loss, truth_param_list_list, param_init_list, param_condition_list, option = False) :
	# loss가 여러개? 한 개?
	for src, (objective, truth_param_list) in enumerate(zip(objective_list, truth_param_list_list)) :
		recovered_param_list_list, loss_list_list = [], []
		for tgt, (param_init, param_condition) in enumerate(zip(param_init_list, param_condition_list)) :
			recovered_param_list, loss_list = parameter_recovery(objective, loss, truth_param_list, param_init, param_condition, option = False)
			recovered_param_list_list.append(recovered_param_list)
			loss_list_list.append(loss_list)
		transposed_matrix = np.array(loss_list_list).T 
		min_loss_indices = np.argmin(transposed_matrix, axis=1).tolist()
		res.append(np.bincount(min_loss_indices))
	
	return res
		