AGENT_1_ID = 0
AGENT_2_ID = 1

# initialize policies
NUM_ACTIONS = 6 # 4 moves + 2 legal switches
NULL_ACTION_ID = NUM_ACTIONS
STATE_DIM = 57


NUM_MOVES = 4
SWITCH_OFFSET = 12 # env takes 16-21 as numbers corresponding to switches; CoPG distributions returns 4-5

TEAM = """
Tapu Fini @ Expert Belt  
Ability: Misty Surge  
EVs: 4 Def / 252 SpA / 252 Spe  
Modest Nature  
- Hydro Pump  
- Ice Beam  
- Moonblast  
- Knock Off  

Mamoswine @ Leftovers  
Ability: Thick Fat  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Earthquake  
- Icicle Crash  
- Ice Shard  
- Substitute  

Dragonite @ Heavy-Duty Boots  
Ability: Multiscale  
EVs: 248 HP / 52 Atk / 56 Def / 152 Spe  
Adamant Nature  
- Dragon Dance  
- Ice Punch  
- Earthquake  
- Roost  
"""