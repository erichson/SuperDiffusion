"""
Created on Fri Jul  5, 2024

@author: ben
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from torch_ema import ExponentialMovingAverage

class GaussianDiffusionModelSR(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        prediction_type: str,
        sampler:str,
        ema_val = 0.999,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(GaussianDiffusionModelSR, self).__init__()
        
        self.prediction_type = prediction_type
        assert self.prediction_type in ['v','eps','x'], "ERROR: Prediction not supported. Options are v, eps, x"
        self.sampler = sampler
        assert self.sampler in ['ddpm','ddim'], "ERROR: Sampler not supported. Options are ddpm and ddim"

        
        self.eps_model = eps_model
        self.ema = ExponentialMovingAverage(self.eps_model.parameters(), decay=ema_val)
        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self,  snapshots: torch.Tensor, conditioning_snapshots: torch.Tensor) -> torch.Tensor:


        conditioning_snapshots_interpolated = nn.functional.interpolate(conditioning_snapshots, 
                                                                        size=[snapshots.shape[2], snapshots.shape[3]], 
                                                                        mode='bilinear')

        residual_snapshots = snapshots - conditioning_snapshots_interpolated


        if self.sampler == 'ddpm':
            """
            Forward diffusion: tries to guess the epsilon value from snapshot_t using the eps_model.
            See Alg 1 in https://arxiv.org/pdf/2006.11239
            """
            _ts = torch.randint(1, self.n_T + 1, (residual_snapshots.shape[0],)).to(residual_snapshots.device) # t ~ Uniform(0, n_T)
            eps = torch.randn_like(residual_snapshots)  # eps ~ N(0, 1)
            alpha = self.sqrtab[_ts, None, None, None]
            sigma = self.sqrtmab[_ts, None, None, None]
            _ts = _ts/self.n_T
            
            
        elif self.sampler == 'ddim':        
            _ts = torch.rand(size=(residual_snapshots.shape[0],)).to(residual_snapshots.device)
            eps = torch.randn_like(residual_snapshots)  # eps ~ N(0, 1)
            _, alpha, sigma = get_logsnr_alpha_sigma(_ts)
            
        residual_snapshots_t = alpha * residual_snapshots + eps * sigma

        # We should predict the "error term" from this snapshots_t. Loss is what we return.
        eps_predicted = self.eps_model(residual_snapshots_t, _ts, lowres_snapshot=conditioning_snapshots_interpolated)
        
        
        # Different predictions schemes
        if self.prediction_type == 'x':
            target = residual_snapshots
        elif self.prediction_type == 'eps':
            target = eps
        elif self.prediction_type == 'v':
            target = alpha * eps - sigma * residual_snapshots
        
        return self.criterion(eps_predicted, target)




    def sample(self, n_sample: int, size, conditioning_snapshots: torch.Tensor, device='cuda') -> torch.Tensor:
        """
        Let's sample
        See Alg 2 in https://arxiv.org/pdf/2006.11239
        """
        snapshots_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        

        conditioning_snapshots_interpolated = nn.functional.interpolate(conditioning_snapshots[0:snapshots_i.shape[0]], 
                                                                       size=[snapshots_i.shape[2], snapshots_i.shape[3]], 
                                                                       mode='bilinear')


        if self.sampler == 'ddpm':  
            for i in range(self.n_T, 0, -1):
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                alpha = self.sqrtab[i]
                sigma = self.sqrtmab[i]
                
                pred = self.eps_model(snapshots_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample),
                                      lowres_snapshot=conditioning_snapshots_interpolated)

                if self.prediction_type == 'eps':
                    eps = pred
                elif self.prediction_type == 'x':
                    eps = (alpha * pred - snapshots_i) / sigma
                elif self.prediction_type == 'v':
                    eps = alpha * pred + snapshots_i * sigma
                
                snapshots_i = (self.oneover_sqrta[i] * (snapshots_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z)


        elif self.sampler == 'ddim':

            for time_step in range(self.n_T, 0, -1):
                time = torch.ones((n_sample,) ) * time_step / self.n_T
                logsnr, alpha, sigma = get_logsnr_alpha_sigma(time.to(device))
                logsnr_, alpha_, sigma_ = get_logsnr_alpha_sigma(torch.ones(n_sample,).to(device) * (time_step - 1) / self.n_T)

                pred = self.eps_model(snapshots_i, time.to(device), lowres_snapshot=conditioning_snapshots_interpolated)

                if self.prediction_type == 'v':                    
                    mean = alpha * snapshots_i - sigma * pred
                    eps = pred * alpha + snapshots_i * sigma
                elif self.prediction_type == 'x':
                    mean = pred
                    eps = (alpha * pred - snapshots_i) / sigma
                elif self.prediction_type == 'eps':
                    mean = (snapshots_i - sigma * pred) / alpha
                    eps = pred
                    
                snapshots_i = alpha_ * mean + sigma_ * eps

            #Replace last prediction with the mean value
            snapshots_i = mean



        return snapshots_i + conditioning_snapshots_interpolated
    
    
    
class GaussianDiffusionModelCast(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        prediction_type: str,
        sampler:str,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(GaussianDiffusionModelCast, self).__init__()
        
        self.prediction_type = prediction_type
        assert self.prediction_type in ['v','eps','x'], "ERROR: Prediction not supported. Options are v, eps, x"
        self.sampler = sampler
        assert self.sampler in ['ddpm','ddim'], "ERROR: Sampler not supported. Options are ddpm and ddim"

        
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self,  snapshots: torch.Tensor, conditioning_snapshots: torch.Tensor, past_snapshots: torch.Tensor, s: torch.Tensor) -> torch.Tensor:


        conditioning_snapshots_interpolated = nn.functional.interpolate(conditioning_snapshots, 
                                                                        size=[snapshots.shape[2], snapshots.shape[3]], 
                                                                        mode='bilinear')

        residual_snapshots = snapshots - conditioning_snapshots_interpolated - past_snapshots


        if self.sampler == 'ddpm':
            """
            Forward diffusion: tries to guess the epsilon value from snapshot_t using the eps_model.
            See Alg 1 in https://arxiv.org/pdf/2006.11239
            """
            _ts = torch.randint(1, self.n_T + 1, (residual_snapshots.shape[0],)).to(residual_snapshots.device) # t ~ Uniform(0, n_T)
            eps = torch.randn_like(residual_snapshots)  # eps ~ N(0, 1)
            alpha = self.sqrtab[_ts, None, None, None]
            sigma = self.sqrtmab[_ts, None, None, None]
            _ts = _ts/self.n_T
            
            
        elif self.sampler == 'ddim':        
            _ts = torch.rand(size=(residual_snapshots.shape[0],)).to(residual_snapshots.device)
            eps = torch.randn_like(residual_snapshots)  # eps ~ N(0, 1)
            _, alpha, sigma = get_logsnr_alpha_sigma(_ts)
            
        residual_snapshots_t = alpha * residual_snapshots + eps * sigma

        # We should predict the "error term" from this snapshots_t. Loss is what we return.
        eps_predicted = self.eps_model(residual_snapshots_t, _ts, 
                                       lowres_snapshot=conditioning_snapshots_interpolated,
                                       past_snapshot=past_snapshots, s=s)
        
        
        # Different predictions schemes
        if self.prediction_type == 'x':
            target = residual_snapshots
        elif self.prediction_type == 'eps':
            target = eps
        elif self.prediction_type == 'v':
            target = alpha * eps - sigma * residual_snapshots
        
        return self.criterion(eps_predicted, target)




    def sample(self, n_sample: int, size, conditioning_snapshots: torch.Tensor, past_snapshots: torch.Tensor, s: torch.Tensor, device='cuda') -> torch.Tensor:
        """
        Let's sample
        See Alg 2 in https://arxiv.org/pdf/2006.11239
        """
        snapshots_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        
        

        conditioning_snapshots_interpolated = nn.functional.interpolate(conditioning_snapshots[0:snapshots_i.shape[0]], 
                                                                       size=[snapshots_i.shape[2], snapshots_i.shape[3]], 
                                                                       mode='bilinear')


        if self.sampler == 'ddpm':  
            for i in range(self.n_T, 0, -1):
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                
                alpha = self.sqrtab[i]
                sigma = self.sqrtmab[i]
                
                pred = self.eps_model(snapshots_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample),
                                      lowres_snapshot=conditioning_snapshots_interpolated,
                                      past_snapshot=past_snapshots, s=s)

                if self.prediction_type == 'eps':
                    eps = pred
                elif self.prediction_type == 'x':
                    eps = (alpha * pred - snapshots_i) / sigma
                elif self.prediction_type == 'v':
                    eps = alpha * pred + snapshots_i * sigma
                
                snapshots_i = (self.oneover_sqrta[i] * (snapshots_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z)


        elif self.sampler == 'ddim':

            for time_step in range(self.n_T, 0, -1):
                time = torch.ones((n_sample,) ) * time_step / self.n_T
                logsnr, alpha, sigma = get_logsnr_alpha_sigma(time.to(device))
                logsnr_, alpha_, sigma_ = get_logsnr_alpha_sigma(torch.ones(n_sample,).to(device) * (time_step - 1) / self.n_T)


                pred = self.eps_model(snapshots_i, time.to(device), lowres_snapshot=conditioning_snapshots_interpolated,
                                      past_snapshot=past_snapshots, s=s)

                if self.prediction_type == 'v':                    
                    mean = alpha * snapshots_i - sigma * pred
                    eps = pred * alpha + snapshots_i * sigma
                elif self.prediction_type == 'x':
                    mean = pred
                    eps = (alpha * pred - snapshots_i) / sigma
                elif self.prediction_type == 'eps':
                    mean = (snapshots_i - sigma * pred) / alpha
                    eps = pred
                    
                snapshots_i = alpha_ * mean + sigma_ * eps

            #Replace last prediction with the mean value
            snapshots_i = mean



        return snapshots_i + conditioning_snapshots_interpolated + past_snapshots

    
    
    
def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20.):
    b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
    a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - b
    return -2. * torch.log(torch.tan(a * t + b))


def get_logsnr_alpha_sigma(time):
    logsnr = logsnr_schedule_cosine(time)
    alpha = torch.sqrt(torch.sigmoid(logsnr))[:,None,None,None]
    sigma = torch.sqrt(torch.sigmoid(-logsnr))[:,None,None,None]

    return logsnr, alpha, sigma    
    

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
