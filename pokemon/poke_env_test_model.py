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

p1 = policy(STATE_DIM, NUM_ACTIONS + 1) # support null action being the last action id
