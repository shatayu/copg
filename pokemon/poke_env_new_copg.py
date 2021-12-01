import asyncio
import numpy as np
from tabulate import tabulate
from threading import Thread
from shared_info import SharedInfo

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

from markov_soccer.networks import policy
from markov_soccer.networks import critic

from copg_optim.critic_functions import critic_update, get_advantage
from markov_soccer.soccer_state import get_relative_state, get_two_state
from collections import Counter
import time

# https://pokepast.es/78d5c6bf90f769de
# selected because this team peaked at #3 on the showdown ladder
# https://youtu.be/xwj9d3wjwhg

team = """
Tapu Fini @ Expert Belt  
Ability: Misty Surge  
EVs: 4 Def / 252 SpA / 252 Spe  
Modest Nature  
- Hydro Pump  
- Ice Beam  
- Moonblast  
- Knock Off  

Mamoswine @ Leftovers  
Ability: Thick Fat  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Earthquake  
- Icicle Crash  
- Ice Shard  
- Substitute  

Magnezone @ Air Balloon  
Ability: Magnet Pull  
Shiny: Yes  
EVs: 252 Def / 116 SpA / 140 Spe  
Bold Nature  
IVs: 0 Atk  
- Iron Defense  
- Body Press  
- Thunderbolt  
- Flash Cannon  

Dragonite @ Heavy-Duty Boots  
Ability: Multiscale  
EVs: 248 HP / 52 Atk / 56 Def / 152 Spe  
Adamant Nature  
- Dragon Dance  
- Ice Punch  
- Earthquake  
- Roost  

Heatran @ Leftovers  
Ability: Flash Fire  
EVs: 252 HP / 128 SpD / 128 Spe  
Calm Nature  
IVs: 0 Atk  
- Stealth Rock  
- Magma Storm  
- Earth Power  
- Taunt  

Kartana @ Choice Scarf  
Ability: Beast Boost  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Leaf Blade  
- Sacred Sword  
- Knock Off  
- Smart Strike  
"""

AGENT_1_ID = 0
AGENT_2_ID = 1

NULL_ACTION_ID = 23

# initialize policies
p1 = policy(1,24)
q = critic(1)

# initialize CoPG
optim_q = torch.optim.Adam(q.parameters(), lr=0.001)

optim = CoPG(p1.parameters(),p1.parameters(), lr=1e-2)

batch_size = 1
num_episode = 10

folder_location = 'tensorboard/pokemon/'
experiment_name = 'copg_6v6_v1/'
directory = '../' + folder_location + experiment_name + 'model'

if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter('../' + folder_location + experiment_name + 'data')

class COPGGen8EnvPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        # opponent_healths = [mon._current_hp for mon in battle.opponent_team.values()]
        self.battle = battle

        available_moves = list(range(len(battle.available_moves)))
        available_switches = [16 + x for x in list(range(len(battle.available_switches)))]
        return (np.array([battle.turn]), available_moves, available_switches)
# things to log
# which move sequences were used
# what percentage of the time did the agent use the optimal move sequence (as measured by winning in as few moves as possible)
# who went first each turn with those move sequences
# who won the battle

def env_algorithm(env, id, shared_info, n_battles):
    for episode in range(n_battles):
        print(f'Episode {episode}')

        # gather states from batch_size batches
        for _ in range(batch_size):
            done = False
            all_data = env.reset()
            observation = all_data[0]

            turn = observation[0]

            while not done:
                shared_info.episode_log.append(f'State (E{episode}A{id}): {observation}')
                action_prob = p1(torch.FloatTensor(observation))
                dist = Categorical(action_prob)
                action = dist.sample()

                legal_actions = all_data[1] + all_data[2]

                while action.item() not in legal_actions:
                    action = dist.sample()

                shared_info.episode_log.append(f'Action by {id} (E{episode}A{id}): {action}')

                all_data, reward, done, _ = env.step(action)
                observation = all_data[0]

                # new way with turns for balanced action counts
                shared_info.mat_state[id].append((torch.FloatTensor([observation]), turn))
                shared_info.mat_action[id].append((action, turn))
                shared_info.mat_reward[id].append((torch.FloatTensor(np.array([reward])), turn))

                if id == AGENT_1_ID:
                    is_done = torch.FloatTensor([1 - int(done)])
                    shared_info.mat_done.append((is_done, turn))

            shared_info.num_completed_batches[id] += 1

        # battles are multithreaded so one agent may finish a battle slightly earlier than the second;
        # the code below will run only when the second agent is fully caught up with the first
        if shared_info.batch_counts_equal():
            reward1, reward2 = shared_info.get_num_wins()

            writer.add_scalar('Steps/agent_1_win_rate', 100 * float(reward1) / (reward1 + reward2), episode)
            writer.add_scalar('Steps/agent_2_win_rate', 100 * float(reward1) / (reward1 + reward2), episode)
            writer.add_scalar('Steps/average_battle_length_in_batch', len(shared_info.mat_done) / batch_size, episode)
            
            mat_state1, mat_state2 = shared_info.get_turn_balanced_states()

            mat_reward1, mat_reward2 = shared_info.get_turn_balanced_rewards()

            mat_done = shared_info.get_turn_balanced_done()
            
            # generate mat_action array
            mat_action1, mat_action2 = shared_info.get_turn_balanced_actions()

            assert len(mat_action1) == len(mat_action2)
            mat_action = list(zip(mat_action1, mat_action2))
            mat_action = list(map(lambda x: torch.FloatTensor(np.array(list(x))), mat_action))

            val1 = q(torch.stack(mat_state1))
            val1 = val1.detach()
            next_value = 0  # because currently we end ony when its done which is equivalent to no next state

            returns_np1 = get_advantage(next_value, torch.stack(mat_reward1), val1, torch.stack(mat_done), gamma=0.99, tau=0.95)

            returns1 = torch.cat(returns_np1)
            advantage_mat1 = returns1 - val1.transpose(0,1)

            val2 = q(torch.stack(mat_state2))
            val2 = val2.detach()
            next_value = 0  # because currently we end ony when its done which is equivalent to no next state
            returns_np2 = get_advantage(next_value, torch.stack(mat_reward2), val2, torch.stack(mat_done), gamma=0.99, tau=0.95)

            returns2 = torch.cat(returns_np2)
            advantage_mat2 = returns2 - val2.transpose(0,1)

            for loss_critic, gradient_norm in critic_update(torch.cat([torch.stack(mat_state1),torch.stack(mat_state2)]), torch.cat([returns1,returns2]).view(-1,1), q, optim_q):
                writer.add_scalar('Loss/critic', loss_critic, episode)
            ed_q_time = time.time()

            val1_p = advantage_mat1

            pi_a1_s = p1(torch.stack(mat_state1))
            dist_batch1 = Categorical(pi_a1_s)
            action_both = torch.stack(mat_action)
            log_probs1 = dist_batch1.log_prob(action_both[:,0])

            pi_a2_s = p1(torch.stack(mat_state2))
            dist_batch2 = Categorical(pi_a2_s)
            log_probs2 = dist_batch2.log_prob(action_both[:,1])

            objective = log_probs1*log_probs2*(val1_p)
            if objective.size(0)!=1:
                raise 'error'

            ob = objective.mean()

            s_log_probs1 = log_probs1[0:log_probs1.size(0)].clone() # otherwise it doesn't change values
            s_log_probs2 = log_probs2[0:log_probs2.size(0)].clone()

            mask = torch.stack(mat_done)

            s_log_probs1[0] = 0
            s_log_probs2[0] = 0

            for i in range(1,log_probs1.size(0)):
                s_log_probs1[i] = torch.add(s_log_probs1[i - 1], log_probs1[i-1])*mask[i-1]
                s_log_probs2[i] = torch.add(s_log_probs2[i - 1], log_probs2[i-1])*mask[i-1]

            objective2 = s_log_probs1[1:s_log_probs1.size(0)]*log_probs2[1:log_probs2.size(0)]*(val1_p[0,1:val1_p.size(1)])
            ob2 = objective2.sum()/(objective2.size(0)-batch_size+1)


            objective3 = log_probs1[1:log_probs1.size(0)]*s_log_probs2[1:s_log_probs2.size(0)]*(val1_p[0,1:val1_p.size(1)])
            ob3 = objective3.sum()/(objective3.size(0)-batch_size+1)

            lp1 = log_probs1*val1_p
            lp1=lp1.mean()
            lp2 = log_probs2*val1_p
            lp2=lp2.mean()
            optim.zero_grad()
            optim.step(ob + ob2 + ob3, lp1,lp2)
            ed_time = time.time()

            writer.add_scalar('Entropy/agent1', dist_batch1.entropy().mean().detach(), episode)
            writer.add_scalar('Entropy/agent2', dist_batch2.entropy().mean().detach(), episode)
            
            shared_info.reset()

            if episode%100==0:
                torch.save(p1.state_dict(),
                        '../' + folder_location + experiment_name + 'model/agent1_' + str(
                            episode) + ".pth")
                torch.save(q.state_dict(),
                        '../' + folder_location + experiment_name + 'model/val_' + str(
                            episode) + ".pth")

# boilerplate code to run battles
def env_algorithm_wrapper(player, id, shared_info, kwargs):
    env_algorithm(player, id, shared_info, **kwargs)

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
player1 = COPGGen8EnvPlayer(battle_format="gen8ou", log_level=40, team=teambuilder)
player2 = COPGGen8EnvPlayer(battle_format="gen8ou", log_level=40, team=teambuilder)

player1._start_new_battle = True
player2._start_new_battle = True

loop = asyncio.get_event_loop()

env_algorithm_kwargs = {"n_battles": num_episode}

shared_info = SharedInfo()

t1 = Thread(target=lambda: env_algorithm_wrapper(player1, AGENT_1_ID, shared_info, env_algorithm_kwargs))
t1.start()

t2 = Thread(target=lambda: env_algorithm_wrapper(player2, AGENT_2_ID, shared_info, env_algorithm_kwargs))
t2.start()

while player1._start_new_battle:
    loop.run_until_complete(launch_battles(player1, player2))
t1.join()
t2.join()