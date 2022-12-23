import numpy as np
import torch


class VanillaPolicyModel(torch.nn.Module):
    def __init__(
            self,
            with_cnn: bool = False,
            num_dense: int = 2,
            dense_dim: int = 100
    ):
        super().__init__()
        self.with_cnn = with_cnn
        self.num_dense = num_dense
        self.dense_dim = dense_dim
        if with_cnn:
            self.cnn = torch.nn.Conv2d(1, 2, 5, padding='same')
            self.dense1 = torch.nn.Linear(84, dense_dim)
        else:
            self.dense1 = torch.nn.Linear(42, dense_dim)

        self.denses = {}
        for i in range(num_dense):
            self.denses[i] = torch.nn.Linear(dense_dim, dense_dim)

        self.final_dense = torch.nn.Linear(dense_dim, 7)

    def forward(self, x):
        if self.with_cnn:
            x = self.cnn(x)
            x = x.reshape(1, 84)
        else:
            x = x.reshape(1, 42)
        x = self.dense1(x)
        x = torch.relu(x)
        for d in self.denses.values():
            x = d(x)
            x = torch.relu(x)
        x = self.final_dense(x)
        return x

    def get_distribution(self, state):
        # Here we prevent the model to select a value that has its column already full

        full_columns = torch.Tensor(
            np.where(np.sum(np.abs(state), axis=0) == state.shape[0], -np.inf, 0)
        ).reshape(1, state.shape[1])
        state = torch.Tensor(state).reshape(1, 6, 7)
        logits = self(state)
        logits = torch.add(logits, full_columns)
        # All categorical does is a softmax and build a distribution
        # function that allows for sampling over it (using .sample)
        try:
            return torch.distributions.Categorical(logits=logits)
        except ValueError:
            print(state)
            print(full_columns)
            print(logits)
            raise