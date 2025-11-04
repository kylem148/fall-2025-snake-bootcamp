from typing import Deque, Tuple, List, Optional

# Import necessary libraries
import torch
import torch.nn as nn
from collections import deque
import random
import numpy as np
from game import Game

from model import LinearQNet, QTrainer


# Define constants for the DQN agent
MAX_MEMORY = 100_000  # Maximum number of experiences to store
BATCH_SIZE = 1000     # Number of experiences to sample for training
LR = 0.001           # Learning rate for the neural network
GAMMA = 0.9          # Discount factor for future rewards
EPSILON = 80         # Initial exploration rate (will decay over time)
INPUT_SIZE = 13      # Number of input features for the neural network
HIDDEN_SIZE = 256    # Number of neurons in the hidden layer
OUTPUT_SIZE = 3      # Number of possible actions (straight, right, left)


class DQN:
    """
    Deep Q-Network agent for playing Snake using reinforcement learning.

    This agent uses a neural network to learn the optimal policy for playing Snake.
    It learns through trial and error, getting rewards for good actions (eating food)
    and penalties for bad actions (hitting walls or itself).
    """

    def __init__(self: "DQN") -> None:
        """Initialize the DQN agent with all necessary components."""
        # Initialize training statistics
        self.n_games = 0
        self.epsilon = EPSILON  # Exploration rate (starts high, decreases over time)
        self.gamma = GAMMA      # Discount factor for future rewards
        
        # Initialize memory for experience replay
        self.memory = deque(maxlen=MAX_MEMORY)  # Store experiences for training
        
        # Initialize the neural network and trainer
        self.model = LinearQNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Track previous game state for distance-based rewards
        self.previous_distance_to_food = None

    def get_state(self, game: "Game") -> List[float]:
        """
        Extract the current state of the game as input features for the neural network.

        The state includes:
        - Danger detection in three directions (straight, right, left)
        - Food direction relative to snake head (up, down, left, right)
        - Normalized distances to food
        - Current snake direction
        """
        # Get the snake's head position
        head = game.snake.head
        
        # Helper function to normalize distances
        def normalize_distance(distance):
            max_distance = max(game.grid_width, game.grid_height)
            return distance / max_distance
        
        # Get current direction as one-hot encoding
        direction_up = game.snake.direction == (0, -1)
        direction_down = game.snake.direction == (0, 1)
        direction_left = game.snake.direction == (-1, 0)
        direction_right = game.snake.direction == (1, 0)
        
        # Helper function to check if a point is dangerous
        def is_collision(point):
            x, y = point
            # Check wall collision
            if x < 0 or x >= game.grid_width or y < 0 or y >= game.grid_height:
                return True
            # Check self collision
            if point in game.snake.body:
                return True
            return False
        
        # Detect dangers in three directions relative to current direction
        # Get points for straight, right turn, and left turn
        current_dir = game.snake.direction
        
        # Calculate relative directions based on current direction
        if current_dir == (0, -1):  # Moving up
            straight = (head[0], head[1] - 1)
            right = (head[0] + 1, head[1])
            left = (head[0] - 1, head[1])
        elif current_dir == (0, 1):  # Moving down
            straight = (head[0], head[1] + 1)
            right = (head[0] - 1, head[1])
            left = (head[0] + 1, head[1])
        elif current_dir == (-1, 0):  # Moving left
            straight = (head[0] - 1, head[1])
            right = (head[0], head[1] - 1)
            left = (head[0], head[1] + 1)
        else:  # Moving right (1, 0)
            straight = (head[0] + 1, head[1])
            right = (head[0], head[1] + 1)
            left = (head[0], head[1] - 1)
        
        danger_straight = is_collision(straight)
        danger_right = is_collision(right)
        danger_left = is_collision(left)
        
        # Get food direction relative to snake head
        food_pos = game.food.position
        food_left = food_pos[0] < head[0]
        food_right = food_pos[0] > head[0]
        food_up = food_pos[1] < head[1]
        food_down = food_pos[1] > head[1]
        
        # Calculate normalized distances to food
        distance_x = abs(head[0] - food_pos[0])
        distance_y = abs(head[1] - food_pos[1])
        normalized_distance_x = normalize_distance(distance_x)
        normalized_distance_y = normalize_distance(distance_y)
        
        # Combine all features into a single state vector
        state = [
            # Danger detection (3 features)
            danger_straight,
            danger_right,
            danger_left,
            
            # Current direction (4 features)
            direction_left,
            direction_right,
            direction_up,
            direction_down,
            
            # Food direction (4 features)
            food_left,
            food_right,
            food_up,
            food_down,
            
            # Normalized distances (2 features)
            normalized_distance_x,
            normalized_distance_y
        ]
        
        # Convert boolean values to floats for the neural network
        return [float(x) for x in state]

    def calculate_reward(self, game: "Game", done: bool) -> int:
        """
        Calculate the reward for the current game state.

        Rewards encourage good behavior:
        - Positive reward for eating food
        - Small positive reward for moving closer to food
        - Small negative reward for moving away from food
        - Large negative reward for dying
        """
        reward = 0
        
        # Get current positions
        head = game.snake.head
        food_pos = game.food.position
        
        # Calculate current distance to food
        current_distance = abs(head[0] - food_pos[0]) + abs(head[1] - food_pos[1])
        
        # Distance-based rewards (encourage moving toward food)
        if self.previous_distance_to_food is not None:
            if current_distance < self.previous_distance_to_food:
                reward += 1  # Reward for getting closer to food
            elif current_distance > self.previous_distance_to_food:
                reward -= 1  # Small penalty for moving away from food
        
        # Update previous distance for next calculation
        self.previous_distance_to_food = current_distance
        
        # Big reward for eating food
        if len(game.snake.body) > getattr(self, 'previous_snake_length', 1):
            reward += 10
            self.previous_distance_to_food = None  # Reset distance tracking
        
        # Store current snake length for next comparison
        self.previous_snake_length = len(game.snake.body)
        
        # Big penalty for dying
        if done:
            reward -= 10
            self.previous_distance_to_food = None  # Reset for next game
        
        return reward

    def remember(
        self,
        state: List[float],
        action: List[int],
        reward: int,
        next_state: List[float],
        done: bool,
    ) -> None:
        """Store an experience in memory for later training (experience replay)."""
        # Add the experience to memory as a tuple
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        """Train the neural network on a batch of experiences from memory."""
        if len(self.memory) > BATCH_SIZE:
            # Sample a random batch of experiences
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            # Use all available experiences if we don't have enough for a full batch
            mini_sample = self.memory
        
        # Unpack the batch into separate lists
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # Train the model on the batch
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(
        self,
        state: List[float],
        action: List[int],
        reward: int,
        next_state: List[float],
        done: bool,
    ) -> None:
        """Train the neural network on a single experience (immediate learning)."""
        # Train on the single experience
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: List[float]) -> List[int]:
        """
        Choose an action based on the current state.

        Uses epsilon-greedy strategy:
        - With probability epsilon: choose random action (exploration)
        - With probability 1-epsilon: choose best action from neural network (exploitation)

        Actions: [1,0,0] = straight, [0,1,0] = turn right, [0,0,1] = turn left
        """
        # Decay epsilon over time (explore less as agent learns)
        self.epsilon = EPSILON - self.n_games
        
        # Initialize action array
        final_move = [0, 0, 0]
        
        # Epsilon-greedy action selection
        if random.randint(0, 200) < self.epsilon:
            # Random action (exploration)
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Action from neural network (exploitation)
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
