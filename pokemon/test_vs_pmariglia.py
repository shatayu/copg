from player_classes import MaxDamagePlayer
import asyncio
from poke_env.server_configuration import LocalhostServerConfiguration
from poke_env.player_configuration import PlayerConfiguration


async def main():
    total = 0

    for i in range(50):
        username = f"poke_env_vs_pm_{i}"
        pc = PlayerConfiguration(username, None)
        player = MaxDamagePlayer(server_configuration=LocalhostServerConfiguration, log_level=20, player_configuration=pc)
        await player.send_challenges(
            "turtle_test_2", 1
        )
        print(f'{username} Won {player.n_won_battles} games.')
        total += player.n_won_battles
    
    print(f'In total, MaxDamagePlayer won {total} / 20 games.')


asyncio.get_event_loop().run_until_complete(main())
