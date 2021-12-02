from collections import Counter

AGENT_1_ID = 0
AGENT_2_ID = 1

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