import copy

class Belief :
	def __init__(self, params) :
	def update(self, stimuli) :
		return copy.deepcopy(self)
	def generate_response(self) :

class state :
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

#loss를 belief에만 의존하게 할지
def Loss(original_responses, beliefs, responses) :

def recovery(given_responses, init_state, init_belief) :
	#원래 data를 simulate하는 걸 각 시행마다 하게 할지
	_, stimuli, beliefs, responses = run_episode(init_state, init_belif)

	#두 종류의 recovery 방식
	_, _, beliefs, responses = run_episode(init_state, init_belief)
	#괜찮음.. 근데 나는 더 general하게 만들고 싶었던 거라
	_, beliefs, responses = guided_run_episode(init_state, init_belief, stimuli, responses)

	return Loss(given_responses, beliefs, responses)