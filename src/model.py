import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _auto_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class AlphaZeroNet(nn.Module):
    def __init__(self, game, num_resBlocks=5, num_hidden=128, device=None):
        super().__init__()
        self.num_resBlocks = num_resBlocks
        self.num_hidden = num_hidden
        self.action_size = game.action_size
        self.device = _auto_device() if device is None else torch.device(device)
        
        # 2-channel input: (current player's pieces, opponent's pieces)
        self.num_input_channels = 2
        self.startBlock = nn.Sequential(
            nn.Conv2d(self.num_input_channels, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.rows * game.cols, self.action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * game.rows * game.cols, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        self.to(self.device)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
            
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

    @staticmethod
    def encode_board(board):
        """
        Encodes a canonical board (2D numpy array where 1=current player,
        -1=opponent) into a 2-channel representation:
          Channel 0: current player's pieces (1 where player has a piece)
          Channel 1: opponent's pieces (1 where opponent has a piece)
        Returns a (2, rows, cols) numpy array.
        """
        current = (board == 1).astype(np.float32)
        opponent = (board == -1).astype(np.float32)
        return np.stack([current, opponent], axis=0)

    @staticmethod
    def encode_boards(boards):
        boards = np.asarray(boards, dtype=np.int8)
        current = (boards == 1).astype(np.float32)
        opponent = (boards == -1).astype(np.float32)
        return np.stack([current, opponent], axis=1)

    def predict(self, board):
        """
        Takes a single board (2D numpy array), runs it through the network,
        and returns policy (probabilities) and value.
        """
        policies, values = self.predict_batch([board])
        return policies[0], values[0]

    def predict_batch(self, boards):
        """
        Runs a batch of canonical boards through the network and returns
        masked policy probabilities plus scalar values.
        """
        encoded = self.encode_boards(boards)
        valid_moves = np.asarray([board[0] == 0 for board in boards], dtype=np.bool_)
        board_tensor = torch.as_tensor(encoded, dtype=torch.float32, device=self.device)

        self.eval()
        with torch.inference_mode():
            policy_logits, values = self(board_tensor)
            policy_logits = policy_logits.masked_fill(
                ~torch.as_tensor(valid_moves, device=self.device),
                float("-inf"),
            )
            policy = torch.softmax(policy_logits, dim=1)

            # Terminal boards should not be sent here, but guard against it so
            # callers never receive NaNs if the mask removes every action.
            invalid_rows = torch.isnan(policy).any(dim=1)
            if invalid_rows.any():
                policy[invalid_rows] = 0.0

        return policy.cpu().numpy(), values.squeeze(1).cpu().numpy()
