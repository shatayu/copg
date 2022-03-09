import asyncio
import os
import sys
import time
import torch
from torch.distributions import Categorical


from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')

from markov_soccer.networks import policy
from player_classes import COPGTestPlayer, MaxDamagePlayer
from pokemon_constants import NUM_ACTIONS, TEAM, NUM_MOVES, STATE_DIM

NUM_BATTLES = 10

p1 = policy(STATE_DIM, NUM_ACTIONS + 1) # support null action being the last action id
p1.load_state_dict(
    torch.load("../tensorboard/pokemon_critic_lr_1e2_mar2/observationsmodel/agent1_4950.pth"))

teambuilder = ConstantTeambuilder(TEAM)

random_player = RandomPlayer(
    team=teambuilder,
    battle_format="gen8ou",
    log_level=40,
    max_concurrent_battles=NUM_BATTLES
)

copg_test_vs_random = COPGTestPlayer(
    prob_dist=p1,
    team=teambuilder,
    battle_format="gen8ou",
    log_level=40,
    max_concurrent_battles=NUM_BATTLES
)

max_damage_player = MaxDamagePlayer(
    team=teambuilder,
    battle_format="gen8ou",
    log_level=40,
    max_concurrent_battles=NUM_BATTLES
)


copg_test_vs_max_damage = COPGTestPlayer(
    prob_dist=p1,
    team=teambuilder,
    battle_format="gen8ou",
    log_level=40,
    max_concurrent_battles=NUM_BATTLES
)

async def test():
    start = time.time()

    await copg_test_vs_random.battle_against(random_player, n_battles=NUM_BATTLES)

    wins_vs_random = copg_test_vs_random.n_won_battles
    time_vs_random = time.time() - start

    print(
        f'COPG won {wins_vs_random} / {NUM_BATTLES} battles vs. random [this took {time_vs_random} seconds]'
    )

    start = time.time()

    await copg_test_vs_max_damage.battle_against(max_damage_player, n_battles=NUM_BATTLES)
    wins_vs_max_damage = copg_test_vs_max_damage.n_won_battles
    time_vs_max_damage = time.time() - start

    print(
        f'COPG won {wins_vs_max_damage} / {NUM_BATTLES} battles vs. max_damage [this took {time_vs_max_damage} seconds]'
    )

loop = asyncio.get_event_loop()
loop.run_until_complete(test())


