import QNet
import torch
import random
import numpy as np


class Harness:

    def __init__(self, config, all_config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('PyTorch is using ', self.device)
        self.qnet = QNet.QNet(config, all_config)
        self.qnet.to(self.device)
        self.replay_buffer_states = []
        self.replay_buffer_actions = []
        self.replay_buffer_utility = []
        self.replay_buffer_limit = 200
        self.learning_iterations = 50
        self.discount_factor = config["discount_factor"]
        self.epsilon = config["epsilon"]
        self.learning_rate = config["learning_rate"]
        self.relearning_rate = config["relearning_rate"]
        print(self.qnet._modules)
        self.optimizer = torch.optim.SGD(self.qnet.parameters(), lr=self.learning_rate, momentum=0.95)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=25, verbose=True)
        self.loss_fn = torch.nn.MSELoss()
    
    def start_new_rollout(self, epsilon_factor=1.0):
        self.epsilon *= epsilon_factor
        print('epsilon: ', self.epsilon)
        # Make lists to hold the data from this rollout
        self.current_state_list = []
        self.current_action_list = []
        self.current_reward_list = []
        # Clear the memory of the qnet
        self.qnet.reset_hidden()

    def action_for_state(self, state, eval_mode=False):
        # Don't learn here
        with torch.no_grad():
            all_utilities = self.qnet(state.to(self.device))
            print(all_utilities)

        # This is what our network thinks is the best action
        action = torch.argmax(all_utilities)

        # If we are learning, then we need to explore
        if not eval_mode:
            # Epsilon-greedy
            if random.random() < self.epsilon:
                action = np.random.randint(0, all_utilities.shape[1])
            # Record state and action
            self.current_state_list.append(state)
            self.current_action_list.append(action)
        return action

    # Note the reward experiencd after the last action
    def set_reward_for_last_action(self, reward):
        self.current_reward_list.append(reward)

    def replace_reward_for_last_action(self,reward):
        self.current_reward_list[-1] = reward
        
    def end_rollout(self):
        # Move the data for the current rollout into the replay buffer
        self.replay_buffer_states.append(self.current_state_list)
        self.replay_buffer_actions.append(self.current_action_list)

        # Utility is the discounted future rewards
        utility_list = []
        current_utility = 0.0
        # Walk the list of rewards backwards
        for rr in reversed(self.current_reward_list):
            # Discount future rewards
            current_utility = rr + self.discount_factor * current_utility
            utility_list.insert(0, current_utility)
        self.replay_buffer_utility.append(utility_list)

        # If we have too many rollouts, then remove the oldest one
        while len(self.replay_buffer_states) > self.replay_buffer_limit:
            self.replay_buffer_states.pop(0)
            self.replay_buffer_actions.pop(0)
            self.replay_buffer_utility.pop(0)
    
    def learn_from_replay_buffer(self):
        rollout_count = len(self.replay_buffer_actions)
        print('learning from ', rollout_count, ' rollouts')
        loss_sums = []
        for i in range(self.learning_iterations * 2*(rollout_count+1)):
            # Pick a rollout
            if i < self.learning_iterations:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.learning_rate
                current_rollout = -1
            elif i < self.learning_iterations * (rollout_count + 1):
                for g in self.optimizer.param_groups:
                    g['lr'] = self.relearning_rate
                simple_prob_fxn = (np.arange(rollout_count) + 1) / np.sum((np.arange(rollout_count) + 1))
                current_rollout = np.random.choice(np.arange(rollout_count), size=1, p=simple_prob_fxn)
                current_rollout = int(current_rollout)
                current_rollout = int(i / (self.learning_iterations * 1.0)) - 1
                if current_rollout >= rollout_count:
                    current_rollout = -1
            elif i % 20 == 0:
                current_rollout = random.randint(0, rollout_count-1)
            current_state_list = self.replay_buffer_states[current_rollout]
            current_action_list = self.replay_buffer_actions[current_rollout]
            current_utility_list = self.replay_buffer_utility[current_rollout]

            # Clear the memory of the qnet
            self.qnet.reset_hidden()
            loss_sum = 0

            # Walk forward through the states
            for j in range(len(current_state_list)):
                state = current_state_list[j]
                action = current_action_list[j]
                utility = current_utility_list[j]
                with torch.no_grad():
                    est_q_values = self.qnet(state.to(self.device))
                    est_q_values[0, action] = utility
                self.optimizer.zero_grad()
                outputs = self.qnet(state.to(self.device))
                loss = self.loss_fn(outputs, est_q_values)
                loss.backward()
                loss_sum += loss.item()
                self.optimizer.step()
            print('epoch: ', i, 'rollout: ', current_rollout, 'loss: ', loss_sum)
            loss_sums.append(loss_sum)
            # self.scheduler.step(loss_sum)
        return loss_sums
