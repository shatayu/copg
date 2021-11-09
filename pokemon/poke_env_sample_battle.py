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

class RandomGen8EnvPlayer(Gen8EnvSinglePlayer):
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

p1 = RandomGen8EnvPlayer(log_level=25)
p2 = RandomGen8EnvPlayer(log_level=25)

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