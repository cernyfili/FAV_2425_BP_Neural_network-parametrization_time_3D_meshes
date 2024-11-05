# region Neural network
# region Data preparation

import torch.nn as nn


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")

# create surface points for all meshes


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        self.decoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        time_value = x[:, 3].unsqueeze(1)  # Extract time value and keep it as a column vector
        encoded_features = self.encoder(x)
        encoded_with_time = torch.cat((encoded_features, time_value), dim=1)  # Concatenate encoded features with time
        decoded_output = self.decoder(encoded_with_time)
        return decoded_output

# endregion
# region Neural network training





# endregion
# endregion