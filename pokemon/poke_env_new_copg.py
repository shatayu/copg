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
optim = CoPG(p1.parameters(),p2.parameters(), lr=0.5)

folder_location = 'tensorboard/pokemon/'
experiment_name = 'copg_v2_test/'
directory = '../' + folder_location + experiment_name + 'model'

class COPGGen8EnvPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        return np.array([0])

def env_algorithm(env, n_battles):
    for _ in range(n_battles):
        done = False
        observation = env.reset()
        while not done:
            observation, reward, done, _ = env.step(np.random.choice(env.action_space))

def env_algorithm_wrapper(player, kwargs):
    env_algorithm(player, **kwargs)

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
p1 = COPGGen8EnvPlayer(battle_format="gen8ou", log_level=40, team=teambuilder)
p2 = COPGGen8EnvPlayer(battle_format="gen8ou", log_level=40, team=teambuilder)

p1._start_new_battle = True
p2._start_new_battle = True

loop = asyncio.get_event_loop()

env_algorithm_kwargs = {"n_battles": 1}

t1 = Thread(target=lambda: env_algorithm_wrapper(p1, env_algorithm_kwargs))
t1.start()

t2 = Thread(target=lambda: env_algorithm_wrapper(p2, env_algorithm_kwargs))
t2.start()

while p1._start_new_battle:
    loop.run_until_complete(launch_battles(p1, p2))
t1.join()
t2.join()