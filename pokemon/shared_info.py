from collections import Counter
import numpy as np
import torch

from pokemon_constants import AGENT_1_ID, AGENT_2_ID, NULL_ACTION_ID, STATE_DIM

class SharedInfo():
    def __init__(self):
        self.num_agents = 2
        self.num_completed_battles = [0] * self.num_agents

        self.reset()

    def num_battles_equal(self):
        return self.num_completed_battles[0] == self.num_completed_battles[1]
    
    def get_num_wins(self):
        num_wins_agent_1 = len([t for t in self.mat_reward[AGENT_1_ID] if t > 0])
        num_wins_agent_2 = len([t for t in self.mat_reward[AGENT_2_ID] if t > 0])

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

    # returns -1 if item1 < item2, 0 if item1 = item2, 1 if item1 > item2
    def compare_batch_turn(self, batch1, turn1, batch2, turn2):
        if batch1 < batch2:
            return -1
        elif batch1 > batch2:
            return 1
        elif turn1 < turn2:
            return -1
        elif turn1 > turn2:
            return 1
        else:
            return 0

    def get_batch_turn(self, x):
        return x[1], x[2]

    def balance_arrays(self, arr_0, arr_1, null_value):
        new_arr_0 = []
        new_arr_1 = []
        
        i = 0
        j = 0
        
        while i < len(arr_0) and j < len(arr_1):      
            arr_0_batch, arr_0_turn = self.get_batch_turn(arr_0[i])
            arr_1_batch, arr_1_turn = self.get_batch_turn(arr_1[j])

            comparison_result = self.compare_batch_turn(arr_0_batch, arr_0_turn, arr_1_batch, arr_1_turn)
        
            if comparison_result == 0: # equal
                new_arr_0.append(arr_0[i])
                new_arr_1.append(arr_1[j])
        
                i += 1
                j += 1
            elif comparison_result == -1: # less than
                null_entry = (null_value, arr_0_turn)
        
                new_arr_0.append(arr_0[i])
                new_arr_1.append(null_entry)
        
                i += 1
            else:
                null_entry = (null_value, arr_1_turn)
        
                new_arr_0.append(null_entry)
                new_arr_1.append(arr_1[j])
        
                j += 1
        
        while i < len(arr_0):
            arr_0_batch, arr_0_turn = self.get_batch_turn(arr_0[i])
            null_entry = (null_value, arr_0_batch, arr_0_turn)
        
            new_arr_0.append(arr_0[i])
            new_arr_1.append(null_entry)
        
            i += 1
        
        while j < len(arr_1):
            arr_1_batch, arr_1_turn = self.get_batch_turn(arr_1[j])
            null_entry = (null_value, arr_1_batch, arr_1_turn)
        
            new_arr_0.append(null_entry)
            new_arr_1.append(arr_1[j])
        
            j += 1
        
        return new_arr_0, new_arr_1
    
    def get_element(self, arr):
        return [x[0] for x in arr]

    def get_turn_balanced_states(self):
        null_state = torch.FloatTensor(np.array([0] * STATE_DIM))

        b1, b2 = self.balance_arrays(self.mat_state[AGENT_1_ID],
                                    self.mat_state[AGENT_2_ID],
                                    null_state
                                    )
        return self.get_element(b1), self.get_element(b2)
    
    def get_turn_balanced_actions(self):
        null_action = torch.tensor(NULL_ACTION_ID)
        b1, b2 = self.balance_arrays(self.mat_action[AGENT_1_ID],
                                    self.mat_action[AGENT_2_ID],
                                    null_action
                                    )
        return self.get_element(b1), self.get_element(b2)

    def get_turn_balanced_rewards(self):
        null_reward = torch.FloatTensor(np.array([0]))
        b1, b2 = self.balance_arrays(self.mat_reward[AGENT_1_ID],
                                    self.mat_reward[AGENT_2_ID],
                                    null_reward
                                    )
        return self.get_element(b1), self.get_element(b2)

    def get_turn_balanced_done(self):
        # balance rewards to get length of balanced arrays, then add dones
        null_reward = torch.FloatTensor(np.array([0]))
        b1, _ = self.balance_arrays(self.mat_reward[AGENT_1_ID],
                                    self.mat_reward[AGENT_2_ID],
                                    null_reward
                                    )
        # prepend new 1 - int(False) = 1 entries for each new term in here
        dummy_terms = [torch.tensor(1)] * (len(b1) - len(self.mat_done))
        return dummy_terms + self.get_element(self.mat_done)