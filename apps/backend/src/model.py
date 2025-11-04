import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime
from typing import Any


class LinearQNet(nn.Module):
    """
    A simple neural network for Q-learning in the Snake game.

    This is a basic feedforward neural network with:
    - Input layer: game state features (13 inputs)
    - Hidden layer: fully connected layer with ReLU activation
    - Output layer: Q-values for each action (3 outputs: straight, right, left)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initialize the neural network layers.

        Args:
            input_size: Number of input features (13 for snake game state)
            hidden_size: Number of neurons in hidden layer (e.g., 256)
            output_size: Number of output actions (3: straight, right, left)
        """
        # Initialize the neural network as a PyTorch nn.Module
        super(LinearQNet, self).__init__()

        # Create the network layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Any) -> Any:
        """
        Forward pass through the neural network.

        Args:
            x: Input tensor containing the game state

        Returns:
            Output tensor with Q-values for each action
        """
        # Apply ReLU activation to first layer
        x = F.relu(self.linear1(x))
        # Apply second layer (no activation for Q-values)
        x = self.linear2(x)
        return x

    def save(self, file_name: str = None) -> str:
        """Save the trained model to disk with timestamp."""
        # Create model directory if it doesn't exist
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # Generate filename with timestamp if not provided
        if file_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"snake_model_{timestamp}.pth"
        
        if not file_name.endswith('.pth'):
            file_name += '.pth'
            
        file_path = os.path.join(model_folder_path, file_name)
        
        # Save the model state dictionary
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")
        return file_path

    def load(self, file_name: str) -> None:
        """Load a previously saved model from disk."""
        # Construct full file path
        if not file_name.endswith('.pth'):
            file_name += '.pth'
            
        if not os.path.dirname(file_name):
            file_name = os.path.join('./models', file_name)
            
        # Load the model state dictionary
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
            self.eval()
            print(f"Model loaded from {file_name}")
        else:
            raise FileNotFoundError(f"Model file {file_name} not found")


class QTrainer:
    """
    Trainer class for the Q-learning neural network.

    Handles the training process using the Bellman equation:
    Q(s,a) = r + γ * max(Q(s',a'))

    Where:
    - Q(s,a) = Q-value for state s and action a
    - r = immediate reward
    - γ = discount factor (gamma)
    - s' = next state
    - a' = possible actions in next state
    """

    def __init__(self, model: LinearQNet, lr: float, gamma: float) -> None:
        """
        Initialize the trainer with model and hyperparameters.

        Args:
            model: The neural network to train
            lr: Learning rate for the optimizer
            gamma: Discount factor for future rewards
        """
        # Store hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.model = model
        
        # Initialize Adam optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Initialize Mean Squared Error loss function
        self.criterion = nn.MSELoss()

    def train_step(
        self, state: Any, action: Any, reward: Any, next_state: Any, done: Any
    ) -> None:
        """
        Perform one training step on the neural network.

        This implements the Q-learning algorithm update rule.

        Args:
            state: Current game state(s)
            action: Action(s) taken
            reward: Reward(s) received
            next_state: Next game state(s)
            done: Whether the game ended
        """
        # Handle both single experiences and batches
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # Ensure all tensors have batch dimension
        if len(state.shape) == 1:
            # Single experience - add batch dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        # Get current Q-values from the model
        pred = self.model(state)
        
        # Clone predictions to create target values
        target = pred.clone()
        
        # Update target values using Bellman equation
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        # Perform gradient descent
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
