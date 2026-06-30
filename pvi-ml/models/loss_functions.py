import torch
import torch.nn as nn

class MorphologyLoss(nn.Module):
    def __init__(self,
                 base_loss: nn.Module = nn.MSELoss(),
                 base_weight: float = 0.2) -> None:
        super().__init__()
        
        self._alias = type(self).__name__

        self.base_loss_fn = base_loss
        self.base_weight = base_weight
        
    def _scale_minmax(self, x: torch.Tensor) -> torch.Tensor:
        xmin = self._extract_minmax(x)[:,0].unsqueeze(-1)
        xmax = self._extract_minmax(x)[:,1].unsqueeze(-1)
        
        return (x - xmin)/(xmax - xmin)

    
    def _extract_minmax(self, x: torch.Tensor) -> torch.Tensor:
        xmin = torch.min(x,dim=-1)[0].unsqueeze(-1)
        xmax = torch.max(x,dim=-1)[0].unsqueeze(-1)
        
        return torch.cat((xmin,xmax),dim=-1)
        
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        # this is more like a projection bases and not the
        labels = torch.ones(predictions.shape[0]).to(predictions.device)
        # cosine loss varies between 0 and 2 (lower is better)
        base_loss = self.base_loss_fn(predictions, targets)
        cosine_loss = nn.CosineEmbeddingLoss()(predictions, targets, labels)
        
        loss = self.base_weight * base_loss + (1 - self.base_weight) * cosine_loss
        
        return loss

    def get_params_shallow(self) -> dict:

        params_available = sum([p.numel() for p in self.parameters()])
        params_trainable = sum([p.numel() for p in self.parameters() if p.requires_grad])

        dict_out = {'name': self._alias,
                    'total_params': params_available,
                    'trainable_params': params_trainable,
                    'base_loss_fn': str(self.base_loss_fn),
                    'base_weight': self.base_weight}

        all_modules = {}
        for name, module in self.named_modules():
            if name:
                if len(list(module.children())) == 0:  # leaf module
                    all_modules[name] = str(module)
                else:  # container
                    all_modules[name] = ''

        dict_out['modules'] = all_modules

        return dict_out
    
# Example usage
if __name__ == "__main__":
    
    print('dasdasd')

    loss_func = MorphologyLoss()