AGENT_1_ID = 0
AGENT_2_ID = 1

# initialize policies
NUM_ACTIONS = 9 # 4 moves + 5 legal switches
NULL_ACTION_ID = NUM_ACTIONS
STATE_DIM = 111 # 3 + (2 players * 1 team/player * 6 Pokemon/team * 9 bits of info/Pokemon)


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

Magnezone @ Air Balloon
Ability: Magnet Pull
EVs: 252 Def / 116 SpA / 140 Spe
Bold Nature
IVs: 0 Atk
- Iron Defense
- Body Press
- Thunderbolt
- Flash Cannon

Heatran @ Leftovers
Ability: Flash Fire
EVs: 248 HP / 128 SpD / 132 Spe
Calm Nature
IVs: 0 Atk
- Stealth Rock
- Magma Storm
- Earth Power
- Taunt

Kartana @ Choice Scarf
Ability: Beast Boost
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Leaf Blade
- Sacred Sword
- Knock Off
- Smart Strike
"""