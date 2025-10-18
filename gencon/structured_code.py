import torch
from function import prospect
from models import Variance
import copy

# Belief generating class
# State generating class
# maze, metaRL task

class Belief :
	def __init__(self, params, info) :
		self.loss, self.risk_plus, self.risk_minus, self.init_mean, self.init_noise, self.theta, self.alpha, self.rho, self.phi, self.xi = params
		self.env_lamb, self.env_noise, self.env_noise_d, self.env_mean, self.size = info
		self.mu = self.init_mean.expand(self.size).clone()
		self.sigma = self.init_noise.expand(self.size).clone()
		self.preservation = torch.zeros(self.size, dtype=self.init_mean.dtype, device=self.init_mean.device)
		self.prev_reward = 0
	def update(self, stimulus) : 
		#계산 효율적
		#실제로 사용하기도 하고
		#ideal implementation에선 동일
		if stimulus.sum() != 0 :
			nonzero_idx = torch.nonzero(stimulus, as_tuple=False).squeeze()
			action = torch.nn.functional.one_hot(nonzero_idx, num_classes=stimulus.size(0)).sum(dim=0).float()
		else : 
			action = torch.zeros_like(stimulus)
		self.preservation = self.preservation + self.alpha * (action - self.preservation)
		K_t = torch.square(self.sigma) / (torch.square(self.sigma) + self.env_noise ** 2)

		mu_post = self.mu + action * (stimulus - self.mu) * K_t
		sigma_post = self.sigma + action * torch.sqrt((1 - K_t) * torch.square(self.sigma))
		self.mu = self.env_lamb * mu_post + (1 - self.env_lamb) * self.env_mean
		self.sigma = torch.sqrt((self.env_lamb ** 2) * torch.square(sigma_post) + self.env_noise_d ** 2)
		self.prev_reward = stimulus
		return copy.deepcopy(self)
	def generate_response(self) :
		#deterministically determined
		mu = prospect(self.mu, self.loss, self.risk_plus, self.risk_minus, xi =  self.xi, prev_result = self.prev_reward)
		var = Variance(self.risk_plus, self.risk_minus, self.loss, self.mu, self.sigma, self.xi, self.prev_reward)
		sigma = torch.sqrt(var)
		temp = self.theta * (mu + self.rho * self.preservation + self.phi * sigma)
		probs = torch.softmax(temp, dim=-1)
		#random sampled
		idx = torch.multinomial(probs, num_samples=1, replacement=True)[0]
		return torch.nn.functional.one_hot(idx, num_classes=len(probs)).float()

class State :
	def __init__(self, params) :
		self.bandits, self.lamb, self.noise, self.noise_d, self.theta, self.limit = params
		self.selected_idx = torch.zeros_like(self.bandits)
		self.is_finished = False
		self.time = 0
	def update(self, response) :
		self.time += 1
		self.is_finished = (self.time >= self.limit)
		self.selected_idx = response
		bandit_post = torch.normal(self.bandits, self.noise)
		self.bandits = torch.normal(self.lamb * bandit_post + (1- self.lamb) * self.theta, self.noise_d)
		return copy.deepcopy(self)
	def generate_stimulus(self) :
		stimulus = self.bandits * self.selected_idx
		stimulus += ((torch.randn(1) * self.noise)[0]) * self.selected_idx
		return stimulus

def run_trial(state, belief) :
	stimulus = state.generate_stimulus()
	updated_belief = belief.update(stimulus)
	next_response = belief.generate_response()
	next_state = state.update(next_response)
	
	return next_state, stimulus, updated_belief, next_response

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

if __name__ == "__main__":
	#param = self.loss, self.risk_plus, self.risk_minus, self.init_mean, self.init_noise, self.theta, self.alpha, self.rho, self.phi, self.xi, self.size
	#info = self.env_lamb, self.env_noise, self.env_noise_d
	init_belief = Belief(torch.tensor([2, 1, 2, 0, 5, 0, 0.5, 10, 10, 0], dtype=torch.float64), [0.9836, 4, 2.8, 0, torch.Size([4])])
	#self.bandits, self.lamb, self.noise, self.noise_d, self.theta, self.limit
	init_state = State([torch.tensor(torch.normal(0, 10.0, size=(4,), dtype=torch.float64), dtype=torch.float64) , 0.9836, 4, 2.8, 0, 300])

	states, stimuli, beliefs, responses = run_episode(init_state, init_belief)
	print("[Stmuli]")
	print(stimuli)
	print("[Responses]")
	print(responses)