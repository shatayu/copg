# -*- coding: utf-8 -*-
import asyncio
import time

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer

team = """
Garchomp
Ability: Rough Skin
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Dragon Claw
- Earthquake
- Swords Dance
- Brick Break
"""

"""
FOCUS ON GEN 8
env_player class notes:
    Documentation: https://poke-env.readthedocs.io/en/stable/player.html#module-poke_env.player.env_player

    (for gen 8)
    https://github.com/hsahovic/poke-env/blob/3aa5bdb4926e2eb5a82df77df519f4fc94aca890/src/poke_env/player/env_player.py#L567

    0 <= action < 4:
        The actionth available move in battle.available_moves is executed.
    4 <= action < 8:
        The action - 4th available move in battle.available_moves is executed, with
        z-move.
    8 <= action < 12:
        The action - 8th available move in battle.available_moves is executed, with
        mega-evolution.
    8 <= action < 12:
        The action - 8th available move in battle.available_moves is executed, with
        mega-evolution.
    12 <= action < 16:
        The action - 12th available move in battle.available_moves is executed,
        while dynamaxing.
    16 <= action < 22
        The action - 16th available switch in battle.available_switches is executed.

    If the proposed action is illegal, a random legal move is performed.

battle class notes:
    
"""

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


async def main():
    start = time.time()

    # We create two players.
    random_player = RandomPlayer(
        battle_format="gen8ou",
        team=team
    )
    max_damage_player = MaxDamagePlayer(
        battle_format="gen8ou",
        team=team
    )

    # Now, let's evaluate our player
    await max_damage_player.battle_against(random_player, n_battles=1000)

    print(
        "Max damage player won %d / 100 battles [this took %f seconds]"
        % (
            max_damage_player.n_won_battles, time.time() - start
        )
    )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())