import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pkg_resources
import random
from typing import Optional, Tuple
import colorama
from colorama import Fore
from colorama import Style

colorama.init(autoreset=True)

# global game variables
GAME_LENGTH = 6
WORD_LENGTH = 5


def encodeToStr(encoding):
    string = ""
    for enc in encoding:
        string += chr(ord('a') + enc)
    return string

def strToEncode(lines):
    encoding = []
    for line in lines:
        assert len(line.strip()) == 5
        encoding.append([ord(char) - 97 for char in line.strip()])
    return encoding

with open("5_words.txt", "r") as f:
    WORDS = strToEncode(f.readlines())

class InvalidWordException(Exception):
  pass

class WordleEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(WordleEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([26] * WORD_LENGTH)
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=-1, high=2, shape=(GAME_LENGTH, WORD_LENGTH), dtype=int),
            'alphabet': spaces.Box(low=-1, high=2, shape=(26,), dtype=int)
        })
        self.valid_words = WORDS
        self.correct_positions = [None] * WORD_LENGTH
        self.wrong_positions = [set() for _ in range(WORD_LENGTH)]
        self.incorrect_letters = set()
        self.guesses = []

    def step(self, action) -> Tuple[dict, float, bool, bool, dict]:
        if not tuple(action) in self.valid_words:
            action = self.generate_valid_action()

        board_row_idx = GAME_LENGTH - self.guesses_left
        for idx, char in enumerate(action):
            if self.hidden_word[idx] == char:
                encoding = 2
            elif char in self.hidden_word:
                encoding = 1
            else:
                encoding = 0

            self.board[board_row_idx, idx] = encoding
            self.alphabet[char] = encoding

        self.guesses_left -= 1
        self.guesses.append(action)

        if all(self.board[board_row_idx, :] == 2):
            reward = 1.0
            done = True
        else:
            if self.guesses_left > 0:
                reward = 0.0
                done = False
            else:
                reward = -1.0
                done = True

        truncated = False
        self.update_constraints(action)
        return self._get_obs(), reward, done, truncated, {}
    

    def generate_valid_action(self):
        for word in self.valid_words:
            if self.is_valid_guess(word):
                return word
        return random.choice(self.valid_words)

    def is_valid_guess(self, word):
        for i, char in enumerate(word):
            if self.correct_positions[i] is not None and self.correct_positions[i] != char:
                return False
            if char in self.incorrect_letters:
                return False
            if char in self.wrong_positions[i]:
                return False
        return True

    def update_constraints(self, action):
        board_row_idx = GAME_LENGTH - self.guesses_left - 1
        for i, char in enumerate(action):
            if self.board[board_row_idx, i] == 2:
                self.correct_positions[i] = char
            elif self.board[board_row_idx, i] == 1:
                self.wrong_positions[i].add(char)
            elif self.board[board_row_idx, i] == 0:
                self.incorrect_letters.add(char)

    def _get_obs(self) -> dict:
        return {'board': self.board, 'alphabet': self.alphabet}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        super().reset(seed=seed)
        self.hidden_word = random.choice(self.valid_words)
        self.guesses_left = GAME_LENGTH
        self.board = np.negative(np.ones(shape=(GAME_LENGTH, WORD_LENGTH), dtype=int))
        self.alphabet = np.negative(np.ones(shape=(26,), dtype=int))
        self.correct_positions = [None] * WORD_LENGTH
        self.wrong_positions = [set() for _ in range(WORD_LENGTH)]
        self.incorrect_letters = set()
        self.guesses = []
        return self._get_obs(), {}

    def render(self, mode="human"):
        assert mode in ["human"], "Invalid mode, must be \"human\""
        print('###################################################')
        for i in range(len(self.guesses)):
            for j in range(WORD_LENGTH):
                letter = chr(ord('a') + self.guesses[i][j])
                if self.board[i][j] == 0:
                    print(Fore.BLACK + Style.BRIGHT + letter + " ", end='')
                elif self.board[i][j] == 1:
                    print(Fore.YELLOW + Style.BRIGHT + letter + " ", end='')
                elif self.board[i][j] == 2:
                    print(Fore.GREEN + Style.BRIGHT + letter + " ", end='')
            print()
        print()

        for i in range(len(self.alphabet)):
            letter = chr(ord('a') + i)
            if self.alphabet[i] == 0:
                print(Fore.BLACK + Style.BRIGHT + letter + " ", end='')
            elif self.alphabet[i] == 1:
                print(Fore.YELLOW + Style.BRIGHT + letter + " ", end='')
            elif self.alphabet[i] == 2:
                print(Fore.GREEN + Style.BRIGHT + letter + " ", end='')
            elif self.alphabet[i] == -1:
                print(letter + " ", end='')
        print()
        print('###################################################')
        print()
