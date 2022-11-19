from data_processing import *
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_size, 2*input_size),
            nn.ReLU(),
            nn.Linear(2*input_size, 4*input_size),
            nn.ReLU(),
            nn.Linear(4*input_size, 8*input_size),
            nn.ReLU(),
            nn.Linear(8*input_size, 4*input_size),
            nn.ReLU(),
            nn.Linear(4*input_size, 2*input_size),
            nn.ReLU(),
            nn.Linear(2*input_size, output_size),
        ).to(device)
        
        self.actor = self.actor.double()
        
        self.critic = nn.Sequential(
            nn.Linear(input_size, 2*input_size),
            nn.ReLU(),
            nn.Linear(2*input_size, 4*input_size),
            nn.ReLU(),
            nn.Linear(4*input_size, 8*input_size),
            nn.ReLU(),
            nn.Linear(8*input_size, 4*input_size),
            nn.ReLU(),
            nn.Linear(4*input_size, 2*input_size),
            nn.ReLU(),
            nn.Linear(2*input_size, 1)
        ).to(device)

        self.critic = self.critic.double()
    
    def forward(self, x):
        value = self.critic(x)
        #generate logits or probs (if the last actor layer is a softmax) from which we will generate a distribution we will sample from
        logits = self.actor(x)
        
        dist = torch.distributions.Categorical(logits=logits)
        
        return dist, value
    
    def policy_loss(self, old_prob, prob, advantage, eps):
        ratio = prob / old_prob
        
        clip = torch.clamp(ratio, 1-eps, 1+eps)*advantage
        
        loss = torch.min(ratio*advantage, clip)
        
        return -loss

    #TODO
    def critic_loss(self):
        pass

    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        ids = np.random.permutation(batch_size)
        ids = np.split(ids[:batch_size // mini_batch_size * mini_batch_size], batch_size // mini_batch_size)
        for i in range(len(ids)):
            yield states[ids[i]], actions[ids[i]], log_probs[ids[i]], returns[ids[i]], advantage[ids[i]]

    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                dist, value = self(state)
                #entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss #- 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, env, epochs=10, steps_per_epoch = 100, lr=0.001, mini_batch_size=32, ppo_epochs=30):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        state = env.reset()
        
        for epoch in range(epochs):
            print("Epoch : ", epoch)
            actions = []
            values = []
            rewards = []
            states = []
            #log probabilities of each action taken on this trajectory
            log_probs = []
            masks = []
            
            for step in range(steps_per_epoch):
                print("Step : ", step, end='\r')
                state = torch.DoubleTensor(state).to(device)
                dist, value = self.forward(state.detach())
                
                action = dist.sample()
                
                next_state, reward, done, _ = env.step(action.cpu().numpy())
                
                
                log_prob = dist.log_prob(action)
                
                states.append(state)
                rewards.append(torch.DoubleTensor([reward]).unsqueeze(1).to(device))
                actions.append(action.reshape(-1))
                values.append(value)
                log_probs.append(log_prob.reshape(-1))
                masks.append(torch.DoubleTensor(1-done).unsqueeze(1).to(device))
                
                state = next_state
            print("", end='')
            
            if epoch % 5 == 0:
                N = np.arange(steps_per_epoch)
                '''
                plt.plot(N, [i.cpu() for i in rewards], label="Rewards")
                plt.title("Evolution of the rewards at epoch {}".format(epoch))
                
                plt.show()
                '''
                #env.render()
                
            next_state = torch.DoubleTensor(next_state).clone().detach().to(device)
            _, next_value = self.forward(next_state)
            
            #TODO Dig more into the implementation
            returns = self.compute_gae(next_value, rewards, masks, values)
            
            #Detach the tensors that will not be needed for gradient descent to avoid bugs
            states = torch.cat(states)
            rewards = torch.cat(rewards).detach()
            print(actions)
            actions = torch.cat(actions)
            values = torch.cat(values).detach()
            log_probs = torch.cat(log_probs).detach()
            returns = torch.cat(returns).detach()
            advantages = returns - values
            
            self.ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages)