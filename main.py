
from matplotlib import pyplot as plt

import argparse
from GameWrapper import GameWrapper
import os, sys
import utils
import numpy as np


def im():
    import matplotlib.pyplot as plt

    # Data for total times per player
    player1_times = [
        [0.0] * 13, [0.0] * 14, [0.0] * 6, [0.0] * 12, [0.0] * 10, [0.0] * 14, [0.0] * 14,
        [0.0] * 11, [0.0] * 14, [0.0] * 14, [0.0] * 7, [0.0] * 12, [0.0] * 10, [0.0] * 14, [0.0] * 6, [0.0] * 8,
        [0.0] * 6,
        [0.0] * 14, [0.0] * 16, [0.0] * 12
    ]

    player2_times = [
        [0.0468, 0.0312, 0.0469, 0.0780, 0.0780], [0.0626, 0.0781, 0.0469, 0.0625, 0.0625, 0.0312, 0.0625, 0.0781],
        [0.1255, 0.0944, 0.0624, 0.0937, 0.0936, 0.0780, 0.0624, 0.0781],
        [0.1248, 0.0942, 0.0783, 0.0938, 0.0937, 0.0780, 0.0625, 0.0782],
        [0.1248, 0.0944, 0.1249, 0.0780, 0.0937, 0.0780],
        [0.2499, 0.1875, 0.1093, 0.1874, 0.1817, 0.1405, 0.0781, 0.0936, 0.0627, 0.0781, 0.0470, 0.0469, 0.0791, 0.0783,
         0.0781],
        [0.2504, 0.1718, 0.1094, 0.2034, 0.1720, 0.1412, 0.0468, 0.0627, 0.0781, 0.0467, 0.0781, 0.0314, 0.0470, 0.0314,
         0.0313, 0.0625],
        [0.1873, 0.1248, 0.0467, 0.0625, 0.0624, 0.0937, 0.0941, 0.0624, 0.0625, 0.0311, 0.0625, 0.0781, 0.0469, 0.0469,
         0.0781, 0.0313, 0.0781, 0.0469],
        [0.1718, 0.1404, 0.1248, 0.0780, 0.0982, 0.0625, 0.0470, 0.0627, 0.0781, 0.0477, 0.0781, 0.0625, 0.0781, 0.0467,
         0.0625, 0.0470],
        [0.0468, 0.0312, 0.0469, 0.0780, 0.0780], [1.3290, 1.0013, 0.5786, 0.0788, 0.1249],
        [0.0468, 0.0312, 0.0469, 0.0780, 0.0780],
        [0.1096, 0.1407, 0.0631, 0.1249, 0.1250, 0.1563, 0.0781, 0.1250, 0.1255, 0.1405, 0.1250, 0.1250, 0.0936, 0.0781,
         0.1569, 0.0810, 0.1435],
        [0.1419, 0.1100, 0.1218, 0.0946, 0.0956, 0.0625, 0.1100, 0.1271, 0.1571, 0.0937, 0.1100, 0.1257],
        [0.1944, 0.1719, 0.1719, 0.1562, 0.1250, 0.0781, 0.1405, 0.1405],
        [0.1405, 0.1093, 0.1093, 0.0624, 0.1249, 0.1251, 0.1250, 0.1250],
        [0.1405, 0.0944, 0.1409, 0.1409, 0.1249, 0.1563, 0.1406, 0.0938, 0.0781, 0.0628, 0.1250, 0.1095, 0.1413],
        [0.1250, 0.0782, 0.1563, 0.1406, 0.1409, 0.0781, 0.0781, 0.1406, 0.0625, 0.1257, 0.1572, 0.1569],
        [0.1418, 0.1256, 0.1420, 0.0794, 0.0786, 0.1100, 0.0785, 0.0632, 0.0945, 0.1578, 0.1407, 0.0628, 0.1404, 0.0640,
         0.0947]
    ]

    # Calculate total time per round for each player
    player1_total_times = [sum(times) for times in player1_times]
    player2_total_times = [43, 24,16,30,60,52,49, 80, 50,24,37,76,34,12,63,44,87,102,80, 67]

    # Rounds
    rounds = list(range(1, 21))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, player1_total_times, label="Player 1 (Random)", marker='o')
    plt.plot(rounds, player2_total_times, label="Player 2 (QLearning)", marker='s')

    plt.xlabel("Round")
    plt.ylabel("Total Running Time (s)")
    plt.title("Total Running Time for Each Player Across 20 Rounds")
    plt.legend()
    plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    players_options = [x + 'Player' for x in ['Live','Minimax', 'Alphabeta','QLearning','Random']]

    parser = argparse.ArgumentParser()

    parser.add_argument('-player1', default='LivePlayer', type=str,
                        help='The type of the first player.',
                        choices=players_options)
    parser.add_argument('-player2', default='LivePlayer', type=str,
                        help='The type of the second player.',
                        choices=players_options)

    parser.add_argument('-board', default='mub1.csv', type=str,
                        help='Name of board file (.csv).')
    # parser.add_argument('-board', default='default_board.csv', type=str,
    #                     help='Name of board file (.csv).')
    parser.add_argument('-move_time', default=3, type=float,
                        help='Time (sec) for each turn.')
    parser.add_argument('-game_time', default=2000, type=float,
                        help='Global game time (sec) for each player.')
    parser.add_argument('-penalty_score', default=50, type=float,
                        help='Penalty points for a player when it cant move or its time ends.')
    parser.add_argument('-max_fruit_score', default=300, type=float,
                        help='Max points for a fruit on board.')
    # parser.add_argument('-max_fruit_time', default=15, type=float,
    #                     help='Max time for fruit on the board (turns).')

    parser.add_argument('-terminal_viz', action='store_true',
                        help='Show game in terminal only.')
    parser.add_argument('-dont_print_game', action='store_true',
                        help='Together with "terminal_viz", show in terminal only the winner.')
    args = parser.parse_args()

    # check validity of game and turn times
    if args.game_time < args.move_time:
        raise Exception('Wrong time arguments.')

    # check validity of board file type and existance
    board_file_type = args.board.split('.')[-1]
    if board_file_type != 'csv':
        print("saar")
        raise Exception(f'Wrong board file type argument, {board_file_type}.')
    if not args.board in os.listdir('boards'):
        raise Exception(f'Board file {args.board} does not exist in "boards" directory.')

    # Players inherit from AbstractPlayer - this allows maximum flexibility and modularity
    player_1_type = 'players.' + args.player1
    player_2_type = 'players.' + args.player2
    game_time = args.game_time
    penalty_score = args.penalty_score
    __import__(player_1_type)
    __import__(player_2_type)
    player_1 = sys.modules[player_1_type].Player(game_time, penalty_score)
    player_2 = sys.modules[player_2_type].Player(game_time, penalty_score)

    board = utils.get_board_from_csv(args.board)

    # print game info to terminal
    print('Starting Game!')
    print(args.player1, 'VS', args.player2)
    print('Board', args.board)
    print('Players have', args.move_time, 'seconds to make a signle move.')
    print('Each player has', game_time, 'seconds to play in a game (global game time, sum of all moves).')

    # create game with the given args
    game = GameWrapper(board[0], board[1], board[2], player_1=player_1, player_2=player_2,
                       terminal_viz=False,  # Ensure this is False for graphical display
                       print_game_in_terminal=not args.dont_print_game,
                       time_to_make_a_move=args.move_time,
                       game_time=game_time,
                       penalty_score=args.penalty_score,
                       max_dust_score=args.max_fruit_score)

    # start playing!
    game.start_game()
    # im()

