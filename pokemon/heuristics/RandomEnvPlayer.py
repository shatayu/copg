from poke_env.player.env_player import (
	Gen8EnvSinglePlayer,
)
import numpy as np

from poke_env.player_configuration import _create_player_configuration_from_player

class RandomEnvPlayer(Gen8EnvSinglePlayer):
	def __init__(self, 
				name,
				shortname, 
				team, 
				battle_format="gen8ou", 
				log_level = 0, 
				server_configuration=None, 
				save_replays=False):
		self.shortname = shortname
		self.name = name
		pc = _create_player_configuration_from_player(self)
		super().__init__(player_configuration = pc,
						team=team,
						battle_format=battle_format,
						log_level = log_level,
						server_configuration=server_configuration,
						save_replays=save_replays)

	def embed_battle(self, battle):
		return np.array([0])

	def select_action(self, state=None, action_mask=None, test=None, current_step=None):
		if action_mask is not None:
			action_indices = [i for i in range(len(action_mask)) if action_mask[i] == 1]
			return np.random.choice(action_indices)
		else: #shouldnt happen
			return 0
