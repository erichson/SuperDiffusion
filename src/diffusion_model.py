"""
Created on Fri Jul  5, 2024

@author: ben
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class GaussianDiffusionModelSR(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(GaussianDiffusionModelSR, self).__init__()
        
        
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self,  snapshots: torch.Tensor, conditioning_snapshots: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: tries to guess the epsilon value from snapshot_t using the eps_model.
        See Alg 1 in https://arxiv.org/pdf/2006.11239
        """

        conditioning_snapshots_interpolated = nn.functional.interpolate(conditioning_snapshots, 
                                                                        size=[snapshots.shape[2], snapshots.shape[3]], 
                                                                        mode='bilinear')

        residual_snapshots = snapshots - conditioning_snapshots_interpolated
        

        _ts = torch.randint(1, self.n_T + 1, (snapshots.shape[0],)).to(snapshots.device) # t ~ Uniform(0, n_T)
        eps = torch.randn_like(snapshots)  # eps ~ N(0, 1)

        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        snapshots_t = (self.sqrtab[_ts, None, None, None] * residual_snapshots + self.sqrtmab[_ts, None, None, None] * eps)  

        
        # We should predict the "error term" from this x_t. Loss is what we return.
        eps_predicrted = self.eps_model(snapshots_t, _ts / self.n_T, lowres_snapshot=conditioning_snapshots_interpolated)
        return self.criterion(eps, eps_predicrted)



    def sample(self, n_sample: int, size, conditioning_snapshots: torch.Tensor, device='cuda') -> torch.Tensor:
        """
        Let's sample
        See Alg 2 in https://arxiv.org/pdf/2006.11239
        """
        snapshots_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        

        conditioning_snapshots_interpolated = nn.functional.interpolate(conditioning_snapshots[0:snapshots_i.shape[0]], 
                                                                       size=[snapshots_i.shape[2], snapshots_i.shape[3]], 
                                                                       mode='bilinear')

        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(snapshots_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample), lowres_snapshot=conditioning_snapshots_interpolated)
            snapshots_i = (self.oneover_sqrta[i] * (snapshots_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z)

        return snapshots_i + conditioning_snapshots_interpolated
    
    
    
class GaussianDiffusionModel(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(GaussianDiffusionModel, self).__init__()
        
        
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self,  snapshots: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: tries to guess the epsilon value from snapshot_t using the eps_model.
        See Alg 1 in https://arxiv.org/pdf/2006.11239
        """

        _ts = torch.randint(1, self.n_T + 1, (snapshots.shape[0],)).to(snapshots.device) # t ~ Uniform(0, n_T)
        eps = torch.randn_like(snapshots)  # eps ~ N(0, 1)

        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        snapshots_t = (self.sqrtab[_ts, None, None, None] * snapshots + self.sqrtmab[_ts, None, None, None] * eps)          

        # We should predict the "error term" from this x_t. Loss is what we return.
        eps_predicrted = self.eps_model(snapshots_t, _ts / self.n_T)
        return self.criterion(eps, eps_predicrted)



    def sample(self, n_sample: int, size, device='cuda') -> torch.Tensor:
        """
        Let's sample
        See Alg 2 in https://arxiv.org/pdf/2006.11239
        """
        snapshots_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(snapshots_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample))
            snapshots_i = (self.oneover_sqrta[i] * (snapshots_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z)

        return snapshots_i    
    

def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }
