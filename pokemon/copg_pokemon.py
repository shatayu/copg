import asyncio
import numpy as np
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
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
from copg_optim import CoPG
from torch.distributions import Categorical
import numpy as np
from rps.rps_game import rps_game
from torch.utils.tensorboard import SummaryWriter

from rps.network import policy1, policy2
import time

from poke_env.environment.status import Status

from markov_soccer.networks import policy
from markov_soccer.networks import critic

from copg_optim.critic_functions import critic_update, get_advantage
from markov_soccer.soccer_state import get_relative_state, get_two_state
from collections import Counter
import time

from shared_info import SharedInfo
from pokemon_constants import AGENT_1_ID, AGENT_2_ID, NUM_ACTIONS, NULL_ACTION_ID, TEAM, SWITCH_OFFSET, NUM_MOVES, STATE_DIM

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.player.battle_order import ForfeitBattleOrder


from poke_env.data import POKEDEX, MOVES, NATURES
from functools import lru_cache

import requests
import sys


# accept command line arguments

if len(sys.argv) != 5 and len(sys.argv) != 6:
    print('Please provide batch_size, num_episode, num_superbatches, critic lr, nd a name (optional) in that order')
    sys.exit()

batch_size = int(sys.argv[1])
num_episode = int(sys.argv[2])
NUM_SUPERBATCHES = int(sys.argv[3])
critic_lr = float(sys.argv[4])

user_provided_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%s') # default to timestamp
if len(sys.argv) == 6:
    user_provided_name = sys.argv[5]

print(user_provided_name)

@lru_cache(None)
def pokemon_to_int(mon):
    return POKEDEX[mon.species]["num"]

# initialize policies
p1 = policy(STATE_DIM, NUM_ACTIONS + 1) # support null action being the last action id
p2 = policy(STATE_DIM, NUM_ACTIONS + 1)
q = critic(STATE_DIM)

# initialize CoPG
optim_q = torch.optim.Adam(q.parameters(), lr=critic_lr)

optim = CoPG(p1.parameters(), p2.parameters(), lr=1e-4)

folder_location = f'tensorboard/pokemon_{user_provided_name}/'
experiment_name = f'observations'
directory = '../' + folder_location + '/' + experiment_name + 'model'

if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter('../' + folder_location + experiment_name + 'data')

def one_hot_encode_status(status, fill_null_values):
    array = [0] * len(Status)
    statuses = [s for s in Status]
    
    if status in statuses and not fill_null_values:
        index = statuses.index(status)
        array[index] = 1
        
    return array
    
def encode_pokemon(p, fill_null_values=False):
    if fill_null_values:
        return [0.0, 1.0] + one_hot_encode_status(None, fill_null_values)
    else:
        return [1 if p.active else 0, p.current_hp_fraction] + one_hot_encode_status(p.status, fill_null_values)

def get_team_encoding(team):
    pokemon_object_list = list(team.values())
    pokemon_object_list.sort(key=lambda x: x.species)

    return sum([encode_pokemon(p) for p in pokemon_object_list], [])

# fills in unrevealed Pokemon from opponent with Pokemon from the agent's team
def get_opponent_team_encoding(opponent_team, agent_team):
    agent_pokemon_object_list = list(agent_team.values())
    agent_pokemon_object_list.sort(key=lambda x: x.species)

    opponent_pokemon_object_list = list(opponent_team.values())

    opponent_pokemon_revealed_species = set([p.species for p in opponent_pokemon_object_list])
    opponent_pokemon_full_team = []

    for pa in agent_pokemon_object_list:
        if pa.species in opponent_pokemon_revealed_species:
            opponent_pokemon_full_team.append(next(po for po in opponent_pokemon_object_list if pa.species == po.species))
        else:
            opponent_pokemon_full_team.append(pa)
    
    return sum([encode_pokemon(p, True) for p in opponent_pokemon_full_team], []) 

def get_current_state(battle):
    # remember to change STATE_DIM in pokemon_constants.py
    result = np.array(
        [battle.turn, len(battle.available_moves), len(battle.available_switches)] + \
            get_team_encoding(battle.team) + \
            get_opponent_team_encoding(battle.opponent_team, battle.team)
        )

    # result = np.array([battle.turn, len(battle.available_moves), len(battle.available_switches)])

    return result

class COPGGen8EnvPlayer(Gen8EnvSinglePlayer):
    def _action_to_move(  # pyre-ignore
        self, action, battle
    ):
        if action == -1:
            return ForfeitBattleOrder()
        elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0
            <= action - 4
            < len(battle.active_pokemon.available_z_moves)  # pyre-ignore
        ):
            return self.create_order(
                battle.active_pokemon.available_z_moves[action - 4], z_move=True
            )
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action - 8], mega=True)
        elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action - 12], dynamax=True)
        elif 0 <= action - 16 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 16])
        else:
            return ForfeitBattleOrder()

    def embed_battle(self, battle):
        return get_current_state(battle)
    
    def compute_reward(self, battle):
        return self.reward_computing_helper(
            battle,
            fainted_value=10,
            hp_value=1,
            victory_value=1000,
        )


class COPGTestPlayer(Player):
    def choose_move(self, battle):
        observation = get_current_state(battle)
        
        # If the player can attack, it will
        action_prob = p1(torch.FloatTensor(observation))
        dist = Categorical(action_prob)
        action = make_action_legal(dist.sample().item(), observation[1], observation[2])
        action_for_env = adjust_action_for_env(action.item())

        if battle.available_moves and action_for_env < len(battle.available_moves):
            # Finds the best move among available ones
            best_move = battle.available_moves[action_for_env]

            return self.create_order(best_move)

        elif battle.available_switches and action_for_env >= 16 and action_for_env - 16 < len(battle.available_switches):
            best_switch = battle.available_switches[action_for_env - 16]
            return self.create_order(best_switch)
        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

def make_action_legal(action_number, num_available_moves, num_available_switches):
    if action_number == NULL_ACTION_ID:
        # sampled null_action; give it any random action
        legal_action = np.random.randint(0, NUM_ACTIONS)
    elif num_available_moves == 0 and action_number < NUM_MOVES:
        # wants to use a move but no moves, adjust to random switch
        legal_action = np.random.randint(NUM_MOVES, NUM_MOVES + num_available_switches) if num_available_switches > 0 else 0
    elif num_available_switches == 0 and action_number >= NUM_MOVES:
        legal_action = np.random.randint(0, num_available_moves) if num_available_moves > 0 else 0
    elif action_number < NUM_MOVES and action_number >= num_available_moves:
        # wants to use a move, adjust to a random move
        legal_action = np.random.randint(0, num_available_moves)
    elif action_number >= NUM_MOVES and action_number - NUM_MOVES > num_available_switches:
        # wants to switch, give it a random switch
        legal_action = np.random.randint(NUM_MOVES, NUM_MOVES + num_available_switches)
    else:
        legal_action = action_number

    return torch.tensor(legal_action)

def adjust_action_for_env(action_number):
    if action_number >= NUM_MOVES:
        return action_number + SWITCH_OFFSET
    else:
        return action_number


state_dict = {}
def env_algorithm(env, id, shared_info, superbatch, n_battles):
    for episode in range(n_battles):
        timestamp = superbatch * n_battles + episode
        print(f'Superbatch {superbatch} Episode {episode}')

        # gather states from batch_size batches
        for b in range(batch_size):
            done = False
            observation = env.reset()

            turn = observation[0]

            while not done:
                # debug code to print hashes for states
                observation_str = f'{id}_{observation}'
                if observation_str in state_dict:
                    observation_hash = state_dict[observation_str]
                else:
                    observation_hash = f'{id}_{len(state_dict)}'
                    state_dict[observation_str] = observation_hash

                print(f'State (E{episode}, Agent{id}): {observation_hash}')

                if id == AGENT_1_ID:
                    action_prob = p1(torch.FloatTensor(observation))
                else:
                    action_prob = p2(torch.FloatTensor(observation))

                print(f'Action probs (Agent {id}): {action_prob}')
                dist = Categorical(action_prob)
                action = dist.sample()
                action_for_env = adjust_action_for_env(action.item())

                print(f'Action by {id} (E{episode}, Agent{id}): {action}')

                observation, reward, done, _ = env.step(action_for_env)

                shared_info.mat_state[id].append((torch.FloatTensor(observation), b, turn))
                shared_info.mat_action[id].append((action, b, turn))
                shared_info.mat_action_log_probs[id].append((dist.log_prob(action), b, turn))
                shared_info.mat_reward[id].append((torch.FloatTensor(np.array([reward])), b, turn))
                shared_info.mat_reward_actual_number[id].append(reward)

                # debug code to print hashes for states
                observation_str = f'{id}_{observation}'
                if observation_str in state_dict:
                    observation_hash = state_dict[observation_str]
                else:
                    observation_hash = f'{id}_{len(state_dict)}'
                    state_dict[observation_str] = observation_hash

                print(f'Resulting state (E{episode}, Agent{id}): {observation_hash}')

                if id == AGENT_1_ID:
                    shared_info.mat_done.append(torch.FloatTensor([1 - int(done)]))

                turn = observation[0]

            shared_info.num_completed_battles[id] += 1

            if id == AGENT_1_ID:
                shared_info.mat_num_turns.append(int(turn))

        # battles are multithreaded so one agent may finish a battle slightly earlier than the second;
        # the code below will run only when the second agent is fully caught up with the first
        # after all battles in the batch are complete
        if shared_info.num_battles_equal():         
            # log trajectory

            # print(f'num_turns_array = {shared_info.mat_num_turns}')
            # print(f'trajectory length = {np.mean(shared_info.mat_num_turns)}')
            writer.add_scalar('trajectory_length', np.mean(shared_info.mat_num_turns), timestamp)

            # log rewards
            # print(f'agent 1 reward = {np.mean(shared_info.mat_reward_actual_number[AGENT_1_ID])}')
            # print(f'agent 2 reward = {np.mean(shared_info.mat_reward_actual_number[AGENT_2_ID])}')
            # print(np.mean(shared_info.mat_reward_actual_number[AGENT_1_ID]))
            writer.add_scalar('agent_1_reward', np.mean(shared_info.mat_reward_actual_number[AGENT_1_ID]), timestamp)
            writer.add_scalar('agent_2_reward', np.mean(shared_info.mat_reward_actual_number[AGENT_2_ID]), timestamp)

            mat_state1, mat_state2 = shared_info.get_turn_balanced_states()

            mat_reward1, mat_reward2 = shared_info.get_turn_balanced_rewards()

            if len(mat_state1) == 0 or len(mat_state2) == 0:
                empty_array = 1 if len(mat_state1) == 0 else 2
                print(f'Dumping episode log because mat_state{empty_array} is empty')
                print(shared_info.episode_log)

            mat_done = shared_info.get_turn_balanced_done()

            mat_action1, mat_action2 = shared_info.get_turn_balanced_actions()
            mat_action1_log_probs, mat_action2_log_probs = shared_info.get_turn_balanced_action_log_probs()

            assert len(mat_action1) == len(mat_action2)
            mat_action = list(zip(mat_action1, mat_action2))
            mat_action = list(map(lambda x: torch.FloatTensor(np.array(list(x))), mat_action))

            val1 = q(torch.stack(mat_state1))
            val1 = val1.detach()
            next_value = 0  # because currently we end ony when its done which is equivalent to no next state

            returns_np1 = get_advantage(next_value, torch.stack(mat_reward1), val1, torch.stack(mat_done), gamma=0.99, tau=0.95)
            
            # compute returns
            returns1 = torch.cat(returns_np1)
            returns_1_number = torch.sum(returns1).item()
            writer.add_scalar('Returns/agent_1', returns_1_number, timestamp)
            advantage_mat1 = returns1 - val1.transpose(0,1)
            
            # compute loss for actor1
            actor1_loss = (-torch.Tensor(mat_action1_log_probs) * advantage_mat1)[0]
            total_actor1_loss = torch.sum(actor1_loss).item()
            writer.add_scalar('Loss/actor1', total_actor1_loss, timestamp)

            val2 = q(torch.stack(mat_state2))
            val2 = val2.detach()
            next_value = 0  # because currently we end ony when its done which is equivalent to no next state
            returns_np2 = get_advantage(next_value, torch.stack(mat_reward2), val2, torch.stack(mat_done), gamma=0.99, tau=0.95)
            
            # compute and log returns for agent 2
            returns2 = torch.cat(returns_np2)
            returns_2_number = torch.mean(returns2).item()
            writer.add_scalar('Returns/agent_2', returns_2_number, timestamp)
            advantage_mat2 = returns2 - val2.transpose(0,1)

            # compute loss for actor2
            actor2_loss = (-torch.Tensor(mat_action2_log_probs) * advantage_mat2)[0]
            total_actor2_loss = torch.sum(actor2_loss).item()
            writer.add_scalar('Loss/actor2', total_actor2_loss, timestamp)

            for loss_critic, gradient_norm in critic_update(torch.cat([torch.stack(mat_state1),torch.stack(mat_state2)]), torch.cat([returns1,returns2]).view(-1,1), q, optim_q):
                writer.add_scalar('Loss/critic', loss_critic, timestamp)
            ed_q_time = time.time()

            val1_p = advantage_mat1

            pi_a1_s = p1(torch.stack(mat_state1))
            dist_batch1 = Categorical(pi_a1_s)
            action_both = torch.stack(mat_action)
            log_probs1 = dist_batch1.log_prob(action_both[:,0])

            pi_a2_s = p2(torch.stack(mat_state2))
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

            writer.add_scalar('Entropy/agent1', dist_batch1.entropy().mean().detach(), timestamp)
            writer.add_scalar('Entropy/agent2', dist_batch2.entropy().mean().detach(), timestamp)
            
            shared_info.reset()

            if episode%100==0:
                torch.save(p1.state_dict(),
                        '../' + folder_location + experiment_name + 'model/agent1_' + str(
                            timestamp) + ".pth")
                torch.save(p2.state_dict(),
                        '../' + folder_location + experiment_name + 'model/agent1_' + str(
                            timestamp) + ".pth")
                torch.save(q.state_dict(),
                        '../' + folder_location + experiment_name + 'model/val_' + str(
                            timestamp) + ".pth")

# boilerplate code to run battles
def env_algorithm_wrapper(player, id, shared_info, superbatch, kwargs):
    env_algorithm(player, id, shared_info, superbatch, **kwargs)

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

def new_timestamp_str():
    STR_LENGTH = 12

    timestamp_str = str(time.time_ns())
    n = len(timestamp_str)

    return timestamp_str[n - STR_LENGTH:]

teambuilder = ConstantTeambuilder(TEAM)
player1_config = PlayerConfiguration(f'p1_{new_timestamp_str()}', None)  # Or password instead of None if playing online
player2_config = PlayerConfiguration(f'p2_{new_timestamp_str()}', None)  # Or password instead of None if playing online

player1 = COPGGen8EnvPlayer(player_configuration=player1_config, battle_format="gen8ou", log_level=40, team=teambuilder)
player2 = COPGGen8EnvPlayer(player_configuration=player2_config, battle_format="gen8ou", log_level=40, team=teambuilder)

random_player = RandomPlayer(
    player_configuration=PlayerConfiguration(f'r_{new_timestamp_str()}', None),
    team=teambuilder,
    battle_format="gen8ou",
    log_level=40
)

copg_test_vs_random = COPGTestPlayer(
    player_configuration=PlayerConfiguration(f'cr_{new_timestamp_str()}', None),
    team=teambuilder,
    battle_format="gen8ou",
    log_level=40
)

max_damage_player = MaxDamagePlayer(
    player_configuration=PlayerConfiguration(f'md_{new_timestamp_str()}', None),
    team=teambuilder,
    battle_format="gen8ou",
    log_level=40
)

copg_test_vs_max_damage = COPGTestPlayer(
    player_configuration=PlayerConfiguration(f'cmd_{new_timestamp_str()}', None),
    team=teambuilder,
    battle_format="gen8ou",
    log_level=40
)

async def test(superbatch):
    start = time.time()

    await copg_test_vs_random.battle_against(random_player, n_battles=100)

    wins_vs_random = copg_test_vs_random.n_won_battles
    time_vs_random = time.time() - start

    print(
        "COPG won %d / 100 battles vs. random [this took %f seconds]"
        % (
            wins_vs_random, time_vs_random
        )
    )

    start = time.time()

    await copg_test_vs_max_damage.battle_against(max_damage_player, n_battles=100)
    wins_vs_max_damage = copg_test_vs_max_damage.n_won_battles
    time_vs_max_damage = time.time() - start

    print(
        "COPG won %d / 100 battles vs. max_damage [this took %f seconds]"
        % (
            wins_vs_max_damage, time_vs_max_damage
        )
    )

    writer.add_scalar('Vs/random_win_rate', wins_vs_random, superbatch)
    writer.add_scalar('Vs/max_damage_win_rate', wins_vs_max_damage, superbatch)

    # with open(f'{experiment_name}_random.txt', "a") as results_file_random:
    #     results_file_random.write(f'{wins_vs_random}\n')

    # with open(f'{experiment_name}_max_damage.txt', "a") as results_file_max_damage:
    #     results_file_max_damage.write(f'{wins_vs_max_damage}\n')


for superbatch in range(NUM_SUPERBATCHES):
    player1._start_new_battle = True
    player2._start_new_battle = True

    loop = asyncio.get_event_loop()

    env_algorithm_kwargs = {"n_battles": num_episode}

    shared_info = SharedInfo()

    t1 = Thread(target=lambda: env_algorithm_wrapper(player1, AGENT_1_ID, shared_info, superbatch, env_algorithm_kwargs))
    t1.start()

    t2 = Thread(target=lambda: env_algorithm_wrapper(player2, AGENT_2_ID, shared_info, superbatch, env_algorithm_kwargs))
    t2.start()

    while player1._start_new_battle:
        loop.run_until_complete(launch_battles(player1, player2))
    t1.join()
    t2.join()

    # test
    loop.run_until_complete(test(superbatch))
