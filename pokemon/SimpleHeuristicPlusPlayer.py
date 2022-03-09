# https://raw.githubusercontent.com/attraylor/poke-env/master/examples/BigBoy_refactor/players/SimpleHeuristicPlusPlayer.py

# -*- coding: utf-8 -*-
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.side_condition import SideCondition
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer  # noqa: F401
from poke_env.environment.pokemon_type import PokemonType



from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import LocalhostServerConfiguration
from poke_env.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from poke_env.utils import to_id_str

from players.RandomEnvPlayer import RandomEnvPlayer
import numpy as np



class SimpleHeuristicPlusPlayer(RandomEnvPlayer):
    def __init__(self, 
                name, 
                shortname,
                team, 
                battle_format="gen8ou", 
                log_level = 0, 
                server_configuration=None, 
                save_replays=False):
        pc = PlayerConfiguration(name, None)
        super().__init__(
                    team=team,
                    name=name, 
                    shortname=shortname,
                    battle_format=battle_format, 
                    log_level = log_level,  
                    server_configuration=server_configuration, 
                    save_replays=save_replays)
        
        self.name = name

    ENTRY_HAZARDS = {
        "spikes": SideCondition.SPIKES,
        "stealhrock": SideCondition.STEALTH_ROCK,
        "stickyweb": SideCondition.STICKY_WEB,
        "toxicspikes": SideCondition.TOXIC_SPIKES,
    }

    ANTI_HAZARDS_MOVES = {"rapidspin", "defog"}

    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4
    SWITCH_OUT_MATCHUP_THRESHOLD = -2

    def _estimate_matchup(self, mon, opponent):
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score

    def _should_dynamax(self, battle, n_remaining_mons):
        if battle.can_dynamax:
            # Last full HP mon
            if (
                len([m for m in battle.team.values() if m.current_hp_fraction == 1])
                == 1
                and battle.active_pokemon.current_hp_fraction == 1
            ):
                return True
            # Matchup advantage and full hp on full hp
            if (
                self._estimate_matchup(
                    battle.active_pokemon, battle.opponent_active_pokemon
                )
                > 0
                and battle.active_pokemon.current_hp_fraction == 1
                and battle.opponent_active_pokemon.current_hp_fraction == 1
            ):
                return True
            if n_remaining_mons == 1:
                return True
        return False

    def _should_switch_out(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if battle.turn == 1:
            self.switches_in_a_row = 0
            self.forced_attack = False
        elif self.switches_in_a_row > 6:
            self.switches_in_a_row = 0
            self.forced_attack = True
            return False
        else:
            self.forced_attack = False
            self.switches_in_a_row += 1
        # If there is a decent switch in...
        if [
            m
            for m in battle.available_switches
            if self._estimate_matchup(m, opponent) > 0
        ]:
            # ...and a 'good' reason to switch out
            if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3:
                return True
            if (
                active.boosts["atk"] <= -3
                and active.stats["atk"] >= active.stats["spa"]
            ):
                return True
            if (
                active.boosts["spa"] <= -3
                and active.stats["atk"] <= active.stats["spa"]
            ):
                return True
            if (
                self._estimate_matchup(active, opponent)
                < self.SWITCH_OUT_MATCHUP_THRESHOLD
            ):
                return True
        self.switches_in_a_row = 0
        self.forced_attack = False
        return False

    def _stat_estimation(self, mon, stat):
        # Stats boosts value
        if mon.boosts[stat] > 1:
            boost = (2 + mon.boosts[stat]) / 2
        else:
            boost = 2 / (2 - mon.boosts[stat])
        return ((2 * mon.base_stats[stat] + 31) + 5) * boost

    def _move_details(self, mon, opponent, move):
        move_multiplier = 1
        if mon.ability == "pixilate" and move.type.name == "NORMAL":
            #treat it as a fairy move
            move_multiplier = 1.3 * 1.5 #STAB + pixilate boost-- assume we have stab (sylveon)
            move_multiplier *= opponent.damage_multiplier(PokemonType.from_name("FAIRY"))
            #TODO: Make this work for Sylveon targeting Dragapult with HV
        elif "levitate" in opponent.possible_abilities and move.type.name == "GROUND":
            move_multiplier = 0
        elif "flashfire" in opponent.possible_abilities and move.type.name == "FIRE":
            move_multiplier = 0
        elif "sapsipper" in opponent.possible_abilities and move.type.name == "GRASS":
            move_multiplier = 0
        elif "stormdrain" in opponent.possible_abilities and move.type.name == "WATER":
            move_multiplier = 0

        return move_multiplier


    def embed_battle(self, battle):
        # Main mons shortcuts
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # Rough estimation of damage ratio
        physical_ratio = self._stat_estimation(active, "atk") / self._stat_estimation(
            opponent, "def"
        )
        special_ratio = self._stat_estimation(active, "spa") / self._stat_estimation(
            opponent, "spd"
        )

        if battle.available_moves and (
            not self._should_switch_out(battle) or not battle.available_switches
        ):
            n_remaining_mons = len(
                [m for m in battle.team.values() if m.fainted is False]
            )
            n_opp_remaining_mons = 6 - len(
                [m for m in battle.team.values() if m.fainted is True]
            )

            # Entry hazard...
            for idx, move in enumerate(battle.available_moves):
                # ...setup
                if (
                    n_opp_remaining_mons >= 3
                    and move.id in self.ENTRY_HAZARDS
                    and self.ENTRY_HAZARDS[move.id]
                    not in battle.opponent_side_conditions
                ):
                    return [idx]

                # ...removal
                elif (
                    battle.side_conditions
                    and move.id in self.ANTI_HAZARDS_MOVES
                    and n_remaining_mons >= 2
                ):
                    return [idx]

            # Setup moves
            if (
                active.current_hp_fraction == 1
                and self._estimate_matchup(active, opponent) > 0
            ):
                for idx, move in enumerate(battle.available_moves):
                    if (
                        move.boosts
                        and sum(move.boosts.values()) >= 2
                        and move.target == "self"
                        and min(
                            [active.boosts[s] for s, v in move.boosts.items() if v > 0]
                        )
                        < 6
                    ):
                        return [idx]

            move = max(
                battle.available_moves,
                key=lambda m: m.base_power
                * (1.5 if m.type in active.types else 1)
                * (
                    physical_ratio
                    if m.category == MoveCategory.PHYSICAL
                    else special_ratio
                )
                * m.accuracy
                * m.expected_hits
                * opponent.damage_multiplier(m)
                * self._move_details(active, opponent, m),
            )
            move_idx = battle.available_moves.index(move)
            if self._should_dynamax(battle, n_remaining_mons):
                move_idx += 12
            return [move_idx]

        if battle.available_switches:
            switch_idx = battle.available_switches.index(max(
                    battle.available_switches,
                    key=lambda s: self._estimate_matchup(s, opponent),
                ))
            return [16 + switch_idx]

        return [-1]


    def select_action(self, state=None, action_mask=None, test=None, current_step=None):
        # If the player can attack, it will
        if state is not None and state[0] == -1: #Pick a random move
            #Subroutine copy pasted from RandomEnvPlayer
            if action_mask is not None:
                action_indices = [i for i in range(len(action_mask)) if action_mask[i] == 1]
                return np.random.choice(action_indices)
            else: #shouldnt happen
                return 0
        else: #return the max damage move
            return state[0]