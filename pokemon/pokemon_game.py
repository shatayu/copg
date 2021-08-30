import numpy as np
import random
import poke_env

# currently RPS; need to repurpose into Pokemon battle
# look at https://poke-env.readthedocs.io/en/stable/player.html#module-poke_env.player.env_player
# should be able to repurpose this into manner usable by classical Gym-style

class rps_game():
    def __init__(self):
        self.number_of_players = 2
        #self.state = np.zeros(2)
        self.reward = np.zeros(2)
        self.done = False

    def step(self, action):
        #self.state = action#np.array((action[0]*action[1],action[0]*action[1]))
        if action[0] == 0:
            if action[1] ==0:
                reward = 0
            elif action[1] ==1:
                reward = 1
            else:
                reward = -1
        elif action[0] == 1:
            if action[1] == 0:
                reward = -1
            elif action[1] ==1:
                reward = 0
            else:
                reward = 1
        else:
            if action[1] == 0:
                reward = 1
            elif action[1] ==1:
                reward = -1
            else:
                reward = 0

        self.reward = np.array((-reward,reward))
        info = {}
        return 0, self.reward[0], self.reward[1], True, info

    def reset(self):
        #self.state = np.zeros(2)
        self.reward = np.zeros(2)
        self.done = False
        info = {}
        return 0, self.reward[0], self.reward[1], self.done, info