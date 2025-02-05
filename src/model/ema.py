import copy
import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay=0.9999, device=None):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        if device is not None:
            self.model.to(device)

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            ema_state = self.model.state_dict()
            current_state = model.state_dict()
            for k in ema_state.keys():
                if k in current_state:
                    ema_state[k].mul_(self.decay).add_(
                        current_state[k], alpha=1.0 - self.decay
                    )

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
