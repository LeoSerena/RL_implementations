import torch


class ConnectFourAgent:
    def __init__(
            self,
            policy: torch.nn.Module,
            training: bool = True,
            id_: int = 1,
            lr: float = 1e-4
    ):
        self.policy = policy
        self.training = training
        self.id_ = id_
        self.reward = 0
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def perform_action(self, action, environment):
        return environment.add_piece(action, self.id_)

    def reset_reward(self):
        self.reward = 0

    def actions_distribution(self, state):
        if self.id_ == -1:
            state = -state
        if not self.training:
            with torch.no_grad():
                return self.policy.get_distribution(state)
        else:
            return self.policy.get_distribution(state)

    def no_grad(self):
        self.optimizer.zero_grad()

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path)
        self.policy.load_state_dict(state_dict)