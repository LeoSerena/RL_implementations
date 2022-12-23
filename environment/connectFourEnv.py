import numpy as np

NUM_ROWS = 6
NUM_COLS = 7


class ConnectFourEnv:
    def __init__(
            self,
            num_rows: int = NUM_ROWS,
            num_cols: int = NUM_COLS,
            win_reward: float = 1,
            blank_reward: float = -1e-4
    ):
        self.state = None
        self.cnt = 0
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.reset_state()
        self.win_reward = win_reward
        self.blank_reward = blank_reward

    def reset_state(self):
        self.state = np.zeros((self.num_rows, self.num_cols))
        self.cnt = 0

    def add_piece(self, i, value):
        """
        Updates the state with a new value in a given column
        """
        if i < 0 or i > self.num_cols - 1:
            raise ValueError(f"The given index <{i}> is not in [0, {self.num_cols}]")
        if self.state[0, i] != 0:
            raise ValueError(f"Trying to insert into a full column <{i}>")
        column = self.state[:, i]
        for j, x in enumerate(column):
            if x != 0:
                j = j - 1
                self.state[j, i] = value
                break
            if j == self.num_rows - 1:
                self.state[j, i] = value
                break

        return self.check_if_over(i, j)

    def check_if_over(self, col_index, row_index):
        def check_traj(X, Y):
            current, cnt = 0, 0
            for x, y in zip(X, Y):
                if x < 0 or y < 0 or x > self.num_cols - 1 or y > self.num_rows - 1:
                    continue
                val = self.state[y, x]
                if val == 0:
                    current = 0
                    cnt = 0
                else:
                    if val == current:
                        cnt += 1
                        if cnt == 4:
                            return True
                    else:
                        cnt = 1
                    current = val
            return False

        r_x = range(col_index - 3, col_index + 4)
        r_y = range(row_index - 3, row_index + 4)

        is_over = check_traj(r_x, [row_index] * 7) \
                  or check_traj([col_index] * 7, r_y) \
                  or check_traj(r_x, r_y) \
                  or check_traj(r_x, reversed(r_y))

        self.cnt += 1
        if not is_over and self.cnt == self.num_rows * self.num_cols:
            return True, 0, False

        return is_over, self.win_reward if is_over else self.blank_reward, True
