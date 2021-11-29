from collections import Counter
import torch
import numpy as np

AGENT_1_ID = 0
AGENT_2_ID = 1

NULL_ACTION_ID = 23

class SharedInfo():
    def __init__(self):
        self.num_agents = 2
        self.num_completed_batches = [0] * self.num_agents

        self.reset()

    def batch_counts_equal(self):
        return self.num_completed_batches[0] == self.num_completed_batches[1]
    
    def get_num_wins(self):
        num_wins_agent_1 = len([t for t in self.mat_reward[AGENT_1_ID][0] if t > 0])
        num_wins_agent_2 = len([t for t in self.mat_reward[AGENT_2_ID][0] if t > 0])

        return (num_wins_agent_1, num_wins_agent_2)
    
    def get_action_counts(self):
        action1 = [float(x) for x in self.mat_action[AGENT_1_ID]]
        action2 = [float(x) for x in self.mat_action[AGENT_2_ID]]

        return list(Counter(action1).items()), list(Counter(action2).items())
    
    def reset(self):
        # one per episode
        self.mat_done = []
        self.episode_log = []

        # one per agent
        self.mat_action = [[] for _ in range(self.num_agents)]
        self.mat_state = [[] for _ in range(self.num_agents)]
        self.mat_reward = [[] for _ in range(self.num_agents)]
    
    def get_turn_balanced_actions(self):
        balanced_action_1, balanced_action_2 = self.balance_arrays(self.mat_action[AGENT_1_ID], self.mat_action[AGENT_2_ID])

        # replace nones with NULL_ACTION_ID
        balanced_action_1 = [(NULL_ACTION_ID, x[1]) if x[0] == None else x for x in balanced_action_1]
        balanced_action_2 = [(NULL_ACTION_ID, x[1]) if x[0] == None else x for x in balanced_action_2]

        assert len(balanced_action_1) == len(balanced_action_2)

        return self.remove_turns(balanced_action_1), self.remove_turns(balanced_action_2)
    
    def get_turn_balanced_states(self):
        balanced_state_1, balanced_state_2 = self.balance_arrays(self.mat_state[AGENT_1_ID], self.mat_state[AGENT_2_ID])

        # states are just turns for now; replace None states with the turn value
        balanced_state_1 = [(torch.FloatTensor(x[1]), x[1]) for x in balanced_state_1]
        balanced_state_2 = [(torch.FloatTensor(x[1]), x[1]) for x in balanced_state_2]

        assert len(balanced_state_1) == len(balanced_state_2)

        return self.remove_turns(balanced_state_1), self.remove_turns(balanced_state_2)
    
    def get_turn_balanced_rewards(self):
        balanced_reward_1, balanced_reward_2 = self.balance_arrays(self.mat_reward[AGENT_1_ID], self.mat_reward[AGENT_2_ID])

        # replace null rewards with 0
        balanced_reward_1 = [(torch.FloatTensor(np.array([0])), x[1]) if x[0] == None else x for x in balanced_reward_1]
        balanced_reward_2 = [(torch.FloatTensor(np.array([0])), x[1]) if x[0] == None else x for x in balanced_reward_2]

        assert len(balanced_reward_1) == len(balanced_reward_2)

        return self.remove_turns(balanced_reward_1), self.remove_turns(balanced_reward_2)
    
    def get_turn_balanced_done(self):
        balanced_reward_1, _ = self.balance_arrays(self.mat_reward[AGENT_1_ID], self.mat_reward[AGENT_2_ID])

        # prepend new 1 - int(False) = 1 entries for each new term in here
        dummy_terms = [torch.FloatTensor(np.array([1]))] * (len(balanced_reward_1) - len(self.mat_done))
        return dummy_terms + self.remove_turns(self.mat_done)
    
    def get_turn(self, action):
        return action[1]
    
    def remove_turns(self, arr):
        return [x[0] for x in arr]

    def balance_arrays(self, arr_0, arr_1):
        new_arr_0 = []
        new_arr_1 = []

        i = 0
        j = 0

        while i < len(arr_0) and j < len(arr_1):      
            arr_0_turn = self.get_turn(arr_0[i])
            arr_1_turn = self.get_turn(arr_1[j])

            if arr_0_turn == arr_1_turn:
                new_arr_0.append(arr_0[i])
                new_arr_1.append(arr_1[j])

                i += 1
                j += 1
            elif arr_0_turn < arr_1_turn:
                null_entry = (None, arr_0_turn)

                new_arr_0.append(arr_0[i])
                new_arr_1.append(null_entry)

                i += 1
            else:
                null_entry = (None, arr_1_turn)

                new_arr_0.append(null_entry)
                new_arr_1.append(arr_1[j])

                j += 1

        while i < len(arr_0):
            arr_0_turn = self.get_turn(arr_0[i])
            null_entry = (None, arr_0_turn)

            new_arr_0.append(arr_0[i])
            new_arr_1.append(null_entry)
        
            i += 1

        while j < len(arr_1):
            arr_1_turn = self.get_turn(arr_1[j])
            null_entry = (None, arr_1_turn)

            new_arr_0.append(null_entry)
            new_arr_1.append(arr_1[j])
        
            j += 1
        
        return new_arr_0, new_arr_1