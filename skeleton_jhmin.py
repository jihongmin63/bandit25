import copy

class Belief :
	def __init__(self, params) :
	def update(self, stimulus) :
		return copy.deepcopy(self)
	def generate_response(self) :

class state :
	def __init__(self, params) :
		self.is_finished = False
	def update(self, response) :
		return copy.deepcopy(self)
	def generate_stimulus(self) :

def run_trial(state, belief) :
	stimulus = state.generate_stimulus()
	updated_belief = belief.update(stimulus)
	next_response = belief.generate_response()
	next_state = state.update(next_response)
	
	return next_state, stimulus, updated_belief, response

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

def recovery(given_responses, init_state, init_belief) :
	_, _, beliefs, responses = run_episode(init_state, init_belief)
	return Loss(given_responses, beliefs, responses)