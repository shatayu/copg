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

team = """
Goodra (M) @ Assault Vest
Ability: Sap Sipper
EVs: 248 HP / 252 SpA / 8 Spe
Modest Nature
IVs: 0 Atk
- Dragon Pulse
- Flamethrower
- Sludge Wave
- Thunderbolt
"""

class RandomGen8EnvPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        return np.array([0])


def env_algorithm_player_1(env, n_battles):
    for _ in range(n_battles):
        done = False
        observation = env.reset()
        while not done:
            action_1 = np.random.choice(env.action_space)
            observation, reward, done, _ = env.step(action_1)

def env_algorithm_player_2(env, n_battles):
    for _ in range(n_battles):
        done = False
        observation = env.reset()
        while not done:
            action_2 = np.random.choice(env.action_space)
            observation, reward, done, _ = env.step(action_2)


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

def env_algorithm(player, n_battles):
    for _ in range(n_battles):
        done = False
        player.reset()
        while not done:
            _, _, done, _ = player.step(np.random.choice(player.action_space))

def env_algorithm_wrapper(player, kwargs):
    env_algorithm(player, **kwargs)

    player._start_new_battle = False
    while True:
        try:
            player.complete_current_battle()
            player.reset()
        except OSError:
            break

teambuilder = ConstantTeambuilder(team)

p1 = RandomGen8EnvPlayer(battle_format="gen8ou", log_level=25, team=teambuilder)
p2 = RandomGen8EnvPlayer(battle_format="gen8ou", log_level=25, team=teambuilder)

p1._start_new_battle = True
p2._start_new_battle = True

loop = asyncio.get_event_loop()

env_algorithm_kwargs = {"n_battles": 5}

t1 = Thread(target=lambda: env_algorithm_wrapper(p1, env_algorithm_kwargs))
t1.start()

t2 = Thread(target=lambda: env_algorithm_wrapper(p2, env_algorithm_kwargs))
t2.start()

while p1._start_new_battle:
    loop.run_until_complete(launch_battles(p1, p2))
t1.join()
t2.join()