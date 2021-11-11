# -*- coding: utf-8 -*-
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

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
from copg_optim import CoPG
from torch.distributions import Categorical
import numpy as np
from rps.rps_game import rps_game
from torch.utils.tensorboard import SummaryWriter

from rps.network import policy1, policy2
import time

team = """
Garchomp (M) @ Sitrus Berry
Ability: Rough Skin
EVs: 248 HP / 252 SpA / 8 Spe
Adamant Nature
- Dragon Claw
- Fire Fang
- Shadow Claw
"""

# initialize policies
p1 = policy1() 
p2 = policy2()
for p in p1.parameters():
    print(p)
for p in p2.parameters():
    print(p)

# initialize CoPG
optim = CoPG(p1.parameters(),p2.parameters(), lr =0.5)

folder_location = 'tensorboard/pokemon/'
experiment_name = 'copg_pokemon_100k_iterations/'
directory = '../' + folder_location + experiment_name + 'model'

if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter('../' + folder_location + experiment_name + 'data')

class CoPGGen8PlayerK(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        # state
        return np.array([0])    

def env_algorithm_player_1(env, shared_info, n_episodes):
    for episode in range(n_episodes):
        print(f'starting episode {episode} in p1')
        done = False
        observation = env.reset()

        pi1 = p1()
        print(f'pi1: {pi1}')
        dist1 = Categorical(pi1)
        action1 = dist1.sample()
        while not done:
            # action_1 = np.random.choice(env.action_space)
            observation, reward, done, _ = env.step(action1)
            shared_info['mat_state1'].append(torch.FloatTensor(observation))
            shared_info['mat_action1'].append(action1)
            shared_info['mat_reward1'].append(torch.FloatTensor(np.array([reward])))
            shared_info['mat_done1'].append(torch.FloatTensor(1 - done))
        shared_info[f'battle_complete1'][episode] = True

        while shared_info['battle_complete1'][episode] != True and shared_info['battle_complete2'][episode] != True:
            pass
        
        ## something here was the issue
        action = np.array([shared_info['mat_action1'][-1], shared_info['mat_action2'][-1]])
        mat_action = [torch.FloatTensor(action)]
        action_both = torch.stack(mat_action)

        writer.add_scalar('Entropy/Agent1', dist1.entropy().data, episode)

        writer.add_scalar('Action/Agent1', torch.mean(action_both[:,0]), episode)
        writer.add_scalar('Action/agent2', torch.mean(action_both[:,1]), episode)

        val1_p = torch.stack(shared_info['mat_reward1']).transpose(0,1)
        print(val1_p)
        if val1_p.size(0)!=1:
            raise 'error'

        pi_a1_s = p1()
        dist_pi1 = Categorical(pi_a1_s)
        action_both = torch.stack(mat_action)
        log_probs1 = dist_pi1.log_prob(action_both[:,0])

        pi_a2_s = p2()
        dist_pi2 = Categorical(pi_a2_s)
        log_probs2 = dist_pi2.log_prob(action_both[:,1])

        objective = log_probs1*log_probs2*(val1_p)
        if objective.size(0) != 1:
            raise 'error'
        
        writer.add_scalar('Agent1/sm1', pi1.data[0], episode)
        writer.add_scalar('Agent1/sm2', pi1.data[1], episode)
        writer.add_scalar('Agent1/sm3', pi1.data[2], episode)

        ob = objective.mean()

        s_log_probs1 = log_probs1.clone() # otherwise it doesn't change values
        s_log_probs2 = log_probs2.clone()

        for i in range(1,log_probs1.size(0)):
            s_log_probs1[i] = torch.add(s_log_probs1[i - 1],log_probs1[i])
            s_log_probs2[i] = torch.add(s_log_probs2[i - 1], log_probs2[i])

        objective2 = s_log_probs1*log_probs2*(val1_p)

        objective3 = log_probs1*s_log_probs2*(val1_p)

        lp1 = log_probs1*val1_p
        lp1=lp1.mean()
        lp2 = log_probs2*val1_p
        lp2=lp2.mean()
        optim.zero_grad()

        optim.step(ob, lp1,lp2)

        if episode % 100 == 0:
            print(episode)
            print(p1.state_dict())
            torch.save(p1.state_dict(),
                    '../' + folder_location + experiment_name + 'model/agent1_' + str(
                        episode) + ".pth")
            torch.save(p2.state_dict(),
                    '../' + folder_location + experiment_name + 'model/agent2_' + str(
                        episode) + ".pth")

        shared_info['update_complete1'][episode] = True

        while shared_info['update_complete1'][episode] != True and shared_info['update_complete2'][episode] != True:
            pass
    

def env_algorithm_player_2(env, shared_info, n_episodes):
    for episode in range(n_episodes):
        print(f'starting episode {episode} in p2')
        done = False
        observation = env.reset()

        pi2 = p2()
        dist2 = Categorical(pi2)
        action2 = dist2.sample()
        while not done:
            observation, reward, done, _ = env.step(action2)
            shared_info['mat_state2'].append(torch.FloatTensor(observation))
            shared_info['mat_action2'].append(action2)
            shared_info['mat_reward2'].append(torch.FloatTensor(np.array([reward])))
            shared_info['mat_done2'].append(torch.FloatTensor(1 - done))
        shared_info[f'battle_complete2'][episode] = True

        while shared_info['battle_complete1'][episode] != True and shared_info['battle_complete2'][episode] != True:
            pass
        writer.add_scalar('Entropy/agent2', dist2.entropy().data, episode)

        writer.add_scalar('Agent2/sm1', pi2.data[0], episode)
        writer.add_scalar('Agent2/sm2', pi2.data[1], episode)
        writer.add_scalar('Agent2/sm3', pi2.data[2], episode)

        
        shared_info['update_complete2'][episode] = True
        while shared_info['update_complete1'][episode] != True and shared_info['update_complete2'][episode] != True:
            pass

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

def env_algorithm_wrapper(player, player_index, shared_info, kwargs):
    if player_index == 1:
        env_algorithm_player_1(player, shared_info, kwargs['n_battles'])
    else:
        env_algorithm_player_2(player, shared_info, kwargs['n_battles'])

    player._start_new_battle = False
    while True:
        try:
            player.complete_current_battle()
            player.reset()
        except OSError:
            break

teambuilder = ConstantTeambuilder(team)

player1 = CoPGGen8PlayerK(battle_format="gen8ou", log_level=25, team=teambuilder)
player2 = CoPGGen8PlayerK(battle_format="gen8ou", log_level=25, team=teambuilder)

player1._start_new_battle = True
player2._start_new_battle = True

loop = asyncio.get_event_loop()

env_algorithm_kwargs = {"n_battles": 100000}

shared_info = {
    "mat_state1": [],
    "mat_state2": [],
    "mat_action1": [],
    "mat_action2": [],
    "mat_reward1": [],
    "mat_reward2": [],
    "mat_done1": [],
    "mat_done2": [],
    "battle_complete1": [False] * env_algorithm_kwargs['n_battles'],
    "battle_complete2": [False] * env_algorithm_kwargs['n_battles'],
    "update_complete1": [False] * env_algorithm_kwargs['n_battles'],
    "update_complete2": [False] * env_algorithm_kwargs['n_battles'],
}

t1 = Thread(target=lambda: env_algorithm_wrapper(player1, 1, shared_info, env_algorithm_kwargs))
t1.start()

t2 = Thread(target=lambda: env_algorithm_wrapper(player2, 2, shared_info, env_algorithm_kwargs))
t2.start()

while player1._start_new_battle:
    loop.run_until_complete(launch_battles(player1, player2))
t1.join()
t2.join()