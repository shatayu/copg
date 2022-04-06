import torch
from torch.distributions import Categorical

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.player.player import Player

from utils import get_current_state, adjust_action_for_env

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
            fainted_value=1e-3,
            hp_value=1e-4,
            victory_value=1,
        )

class COPGTestPlayer(Player):
    def __init__(self, **kwds):
        self.prob_dist = None
        super().__init__(**kwds)

    def set_prob_dist(self, prob_dist):
        self.prob_dist = prob_dist

    def choose_move(self, battle):
        observation = get_current_state(battle)
        
        # If the player can attack, it will
        action_prob = self.prob_dist(torch.FloatTensor(observation))
        dist = Categorical(action_prob)
        action = dist.sample()
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