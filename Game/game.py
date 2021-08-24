from Environment.environment import UnoEnvironment
from Environment.card import Card
from Environment.player import Player
from Players.HumanPlayer import HumanPlayer


class UnoGame(UnoEnvironment):
    def __init__(self, agent: Player):
        super().__init__(True)
        self.reset(HumanPlayer(), agent)
        self.wait_for_action()

    def wait_for_action(self):
        player = self.players[0]

        legal_actions = self.get_legal_actions()
        illegal_input = True
        while illegal_input:
            action_input = int(input("Input action:"))
            if 0 <= action_input <= player.get_hand_size():
                if action_input == 0:
                    action = 54
                    self.played_card = None
                    illegal_input = False
                else:
                    self.played_card = player.cards[action_input - 1]
                    action = player.cards[action_input - 1].action_number
                    if action in legal_actions:
                        illegal_input = False
                    else:
                        print("You can't do that")
            else:
                print("You can't do that")
        self.step_with_opp_step(action)

    def next_turn(self, skip):
        if skip:
            self.previous_player_index = self.current_player_index
        else:
            self.current_player = self.next_player()  # cycles 0 and 1
            self.previous_player_index = self.current_player_index
            self.current_player_index = self.next_player_index()

        self.pretty_print_state()
        if self.current_player_index == 0:
            self.wait_for_action()

    def skip(self):
        self.pretty_print_state()
        self.wait_for_action()

    def pretty_print_state(self):
        print("-------------------------------------------------------------------------------------------------------")
        print("Turn " + str(self.turn))
        Card.pretty_print_cards(self.played_cards[len(self.played_cards) - 1:], False)

        for i, player in enumerate(self.players):

            if i == self.current_player_index:
                print("ACTIVE-", end='')
            print(player.name + "'s hand:")
            if i == 0:
                player.print_hand()
            else:
                print(len(player.cards), "cards")
        print("-------------------------------------------------------------------------------------------------------")