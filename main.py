import os
import sys

import matplotlib.pyplot as plt

from environment.connectFourEnv import ConnectFourEnv
from agent.policy import VanillaPolicyModel
from agent.agent import ConnectFourAgent

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    response = None
    while response not in ['n', 'y', 'yes', 'no']:
        response = input('Wanna play? (y/n)')
        if response in ['n', 'no']:
            print('by!')
            sys.exit()

    print('building model...')
    policy = VanillaPolicyModel(with_cnn=True)
    agent = ConnectFourAgent(policy=policy, training=False, id_=1)
    agent.load_model(os.path.join('saved_models', 'best.pt'))
    environment = ConnectFourEnv()
    print('model ready')

    response = 'y'
    while response in ['y', 'yes']:
        is_over = False
        while not is_over:
            # display current state
            os.system('clear')
            plt.imshow(environment.state)
            plt.show()

            # perform player move
            player_move = None
            while player_move not in [str(x) for x in range(18)]:
                player_move = input('Give column index (1 to 7)')
            player_move = int(player_move)
            is_over, _, win = environment.add_piece(player_move, -1)
            if is_over:
                if win:
                    winner = 'player'
            else:
                # if game not over, perform agent move
                dist = agent.actions_distribution(environment.state)
                agent_action = dist.sample()
                is_over, _, win = agent.perform_action(agent_action, environment)
                if win:
                    winner = 'agent'

        os.system('clear')
        if winner == 'player':
            print('Congrats you won!')
        else:
            print('You lost :(')
        plt.matshow(environment.state)
        plt.show()
        environment.reset_state()

        response = None
        while response not in ['n', 'y', 'yes', 'no']:
            response = input('Wanna play again? (y/n)')
