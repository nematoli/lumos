import torch
import torch.nn as nn


class GRUCellStack(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        layer_size = self.hidden_size // self.n_layers
        layers = [nn.GRUCell(self.input_size, layer_size)]
        layers.extend([nn.GRUCell(layer_size, layer_size) for _ in range(self.n_layers - 1)])
        self.layers = nn.ModuleList(layers)

    def forward(self, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        input_states = state.chunk(self.n_layers, -1)
        output_states = []
        x = input
        for i in range(self.n_layers):
            x = self.layers[i](x, input_states[i])
            output_states.append(x)
        return torch.cat(output_states, -1)
