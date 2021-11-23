import asyncio
import numpy as np
from poke_env.player.random_player import RandomPlayer
from tabulate import tabulate
from threading import Thread

from poke_env.utils import to_id_str
from poke_env.player.env_player import (
    Gen8EnvSinglePlayer,
)
from poke_env.player.utils import cross_evaluate
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from poke_env.player_configuration import PlayerConfiguration


NUM_BATTLES = 1
AGENT_0_ID = 0
AGENT_1_ID = 1
NULL_ACTION_ID = -1

team = """
Garchomp (M) @ Sitrus Berry
Ability: Rough Skin
EVs: 248 HP / 252 SpA / 8 Spe
Adamant Nature
- Dragon Claw
- Fire Fang
- Shadow Claw

Lucario (M) @ Sitrus Berry
Ability: Inner Focus
EVs: 248 HP / 252 SpA / 8 Spe
Adamant Nature
- Close Combat
- Earthquake
- Crunch

Tyranitar (M) @ Sitrus Berry
Ability: Sand Stream
EVs: 248 HP / 252 SpA / 8 Spe
Adamant Nature
- Rock Slide
- Thunder Fang
- Stone Edge
"""

# return turn number in state

class RandomGen8EnvPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        # my_pokemon = battle.available_switches
        return np.array([battle.turn])

class SharedInfo():
    def __init__(self):
        self.num_agents = 2
        self.num_completed_batches = [0] * self.num_agents
        self.actions = [[], []]
        self.states = [[], []]
        self.num_completed_battles = [0, 0]
        self.num_completed_turns = [0, 0]
    
    def num_completed_battles_equal(self):
        return self.num_completed_battles[AGENT_0_ID] == self.num_completed_battles[AGENT_1_ID]

    def num_completed_turns_equal(self):
        return self.num_completed_turns[AGENT_0_ID] == self.num_completed_turns[AGENT_1_ID]

    def reset(self):
        self.actions = [[], []]
        self.num_completed_turns = [0, 0]

def get_turn(action):
    return action[1]

def action_length_balancer(action_0, action_1):
    new_action_0 = []
    new_action_1 = []

    i = 0
    j = 0

    while i < len(action_0) and j < len(action_1):      
        action_0_turn = get_turn(action_0[i])
        action_1_turn = get_turn(action_1[j])

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
        action_0_turn = get_turn(action_0[i])
        null_action = (NULL_ACTION_ID, action_0_turn)

        new_action_0.append(action_0[i])
        new_action_1.append(null_action)
    
        i += 1

    while j < len(action_1):
        action_1_turn = get_turn(action_1[j])
        null_action = (NULL_ACTION_ID, action_1_turn)

        new_action_0.append(null_action)
        new_action_1.append(action_1[j])
    
        j += 1
    
    return new_action_0, new_action_1

def env_algorithm(env, n_battles, id, shared_info):
    for b in range(n_battles):
        done = False
        observation = env.reset()

        while not done:
            action = 0
            print(observation)

            other_id = AGENT_1_ID if id == AGENT_0_ID else AGENT_0_ID

            observation, reward, done, _ = env.step(action)
            shared_info.states[id].append(observation)

            if len(shared_info.actions[id]) > len(shared_info.actions[other_id]) + 1:
                shared_info.actions[other_id].append(NULL_ACTION_ID)

            shared_info.actions[id].append(action)

            shared_info.num_completed_turns[id] += 1                

    if shared_info.num_completed_battles_equal():
        print(f'Battle #{b}')
        print(f'Number of actions Agent 0 took: {len(shared_info.actions[AGENT_0_ID])}')
        print(f'Number of actions Agent 1 took: {len(shared_info.actions[AGENT_1_ID])}')
        print(f'Number of unique actions: {len(set(shared_info.actions[AGENT_0_ID]))}')

        for i in range(len(shared_info.actions)):
            print(f'{i}: {shared_info.actions[i]}')
        
        shared_info.reset()


def env_algorithm_wrapper(player, num_battles, id, shared_info):
    env_algorithm(player, num_battles, id, shared_info)

    player._start_new_battle = False
    while True:
        try:
            player.complete_current_battle()
            player.reset()
        except OSError:
            break

async def launch_battles(player, opponent):
    battles_coroutine = asyncio.gather(
        player.send_challenges(
            opponent=to_id_str(opponent.username),
            n_challenges=1,
            to_wait=opponent.logged_in,
        ),
        opponent.accept_challenges(opponent=to_id_str(player.username), n_challenges=1),
    )
    await battles_coroutine

teambuilder = ConstantTeambuilder(team)

p1 = RandomGen8EnvPlayer(battle_format="gen8ou", log_level=40, team=teambuilder)
p2 = RandomGen8EnvPlayer(battle_format="gen8ou", log_level=40, team=teambuilder)

p1._start_new_battle = True
p2._start_new_battle = True

loop = asyncio.get_event_loop()

env_algorithm_kwargs = {"n_battles": 1}
shared_info = SharedInfo()

t1 = Thread(target=lambda: env_algorithm_wrapper(p1, NUM_BATTLES, AGENT_0_ID, shared_info))
t1.start()

t2 = Thread(target=lambda: env_algorithm_wrapper(p2, NUM_BATTLES, AGENT_1_ID, shared_info))
t2.start()

while p1._start_new_battle:
    loop.run_until_complete(launch_battles(p1, p2))
t1.join()
t2.join()