AGENT_1_ID = 0
AGENT_2_ID = 1
NULL_ACTION_ID = 24

from collections import Counter

class SharedInfo():
    def __init__(self):
        self.num_agents = 2
        self.num_completed_batches = [0] * self.num_agents

        self.reset()

    def batch_counts_equal(self):
        return self.num_completed_batches[0] == self.num_completed_batches[1]
    
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
    
    def get_turn(self, action):
        return action[1]

    def balance_action_array_lengths(self, action_0, action_1):
        new_action_0 = []
        new_action_1 = []

        i = 0
        j = 0

        while i < len(action_0) and j < len(action_1):      
            action_0_turn = self.get_turn(action_0[i])
            action_1_turn = self.get_turn(action_1[j])

            if action_0_turn == action_1_turn:
                new_action_0.append(action_0[i])
                new_action_1.append(action_1[j])

                i += 1
                j += 1
            elif action_0_turn < action_1_turn:
                null_action = (NULL_ACTION_ID, action_0_turn)

                new_action_0.append(action_0[i])
                new_action_1.append(null_action)

                i += 1
            else:
                null_action = (NULL_ACTION_ID, action_1_turn)

                new_action_0.append(null_action)
                new_action_1.append(action_1[j])

                j += 1
        

        while i < len(action_0):
            action_0_turn = self.get_turn(action_0[i])
            null_action = (NULL_ACTION_ID, action_0_turn)

            new_action_0.append(action_0[i])
            new_action_1.append(null_action)
        
            i += 1

        while j < len(action_1):
            action_1_turn = self.get_turn(action_1[j])
            null_action = (NULL_ACTION_ID, action_1_turn)

            new_action_0.append(null_action)
            new_action_1.append(action_1[j])
        
            j += 1
        
        return new_action_0, new_action_1


# class SharedInfo():
#     def __init__(self):
#         self.num_agents = 2
#         self.num_completed_batches = [0] * self.num_agents

#         self.reset()

#     def batch_counts_equal(self):
#         return self.num_completed_batches[0] == self.num_completed_batches[1]
    
#     def get_num_wins(self):
#         num_wins_agent_1 = len([t for t in self.mat_reward[AGENT_1_ID] if t > 0])
#         num_wins_agent_2 = len([t for t in self.mat_reward[AGENT_2_ID] if t > 0])

#         return (num_wins_agent_1, num_wins_agent_2)
    
#     def get_action_counts(self):
#         action1 = [float(x) for x in self.parsed_mat_action[AGENT_1_ID]]
#         action2 = [float(x) for x in self.parsed_mat_action[AGENT_2_ID]]

#         return list(Counter(action1).items()), list(Counter(action2).items())
    
#     def reset(self):
#         # one per episode
#         self.mat_done = []
#         self.episode_log = []

#         # one per agent
#         self.mat_action = [[] for _ in range(self.num_agents)]
#         self.parsed_mat_action = [[] for _ in range(self.num_agents)]
#         self.mat_state = [[] for _ in range(self.num_agents)]
#         self.mat_reward = [[] for _ in range(self.num_agents)]
    
    # def get_turn(self, action):
    #     return action[1]

    # def balance_action_array_lengths(self, action_0, action_1):
    #     new_action_0 = []
    #     new_action_1 = []

    #     i = 0
    #     j = 0

    #     while i < len(action_0) and j < len(action_1):      
    #         action_0_turn = self.get_turn(action_0[i])
    #         action_1_turn = self.get_turn(action_1[j])

    #         if action_0_turn == action_1_turn:
    #             new_action_0.append(action_0[i])
    #             new_action_1.append(action_1[j])

    #             i += 1
    #             j += 1
    #         elif action_0_turn < action_1_turn:
    #             null_action = (NULL_ACTION_ID, action_0_turn)

    #             new_action_0.append(action_0[i])
    #             new_action_1.append(null_action)

    #             i += 1
    #         else:
    #             null_action = (NULL_ACTION_ID, action_1_turn)

    #             new_action_0.append(null_action)
    #             new_action_1.append(action_1[j])

    #             j += 1
        

    #     while i < len(action_0):
    #         action_0_turn = self.get_turn(action_0[i])
    #         null_action = (NULL_ACTION_ID, action_0_turn)

    #         new_action_0.append(action_0[i])
    #         new_action_1.append(null_action)
        
    #         i += 1

    #     while j < len(action_1):
    #         action_1_turn = self.get_turn(action_1[j])
    #         null_action = (NULL_ACTION_ID, action_1_turn)

    #         new_action_0.append(null_action)
    #         new_action_1.append(action_1[j])
        
    #         j += 1
        
    #     return new_action_0, new_action_1
