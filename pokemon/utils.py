from poke_env.environment.status import Status
import numpy as np

from pokemon_constants import NUM_MOVES, SWITCH_OFFSET


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

    return result

def adjust_action_for_env(action_number):
    if action_number >= NUM_MOVES:
        return action_number + SWITCH_OFFSET
    else:
        return action_number