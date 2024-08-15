"""
Created on Fri Jul  5, 2024

@author: ben
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

    
class GaussianDiffusionModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        lowres_model: nn.Module,
        forecast_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        prediction_type: str,
        sampler:str,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(GaussianDiffusionModel, self).__init__()
        
        self.prediction_type = prediction_type
        assert self.prediction_type in ['v','eps','x'], "ERROR: Prediction not supported. Options are v, eps, x"
        self.sampler = sampler
        assert self.sampler in ['ddpm','ddim'], "ERROR: Sampler not supported. Options are ddpm and ddim"

        
        self.base_model = base_model
        self.lowres_model = lowres_model
        self.forecast_model = forecast_model
                
        if self.sampler == 'ddpm':
        # register_buffer allows us to freely access these tensors by name. It helps device placement.
            for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
                self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self,  
                lowres_snapshots: torch.Tensor,
                snapshots: torch.Tensor,
                future_snapshots: torch.Tensor,
                s: torch.Tensor, Reynolds_number:torch.Tensor) -> torch.Tensor:


        lowres_snapshots_interpolated = nn.functional.interpolate(lowres_snapshots, 
                                                                  size=[snapshots.shape[2], snapshots.shape[3]], 
                                                                  mode='bilinear')

        
        residual_snapshots_SR = snapshots - lowres_snapshots_interpolated
        residual_snapshots_FC = future_snapshots  - snapshots

        residual_snapshots = torch.cat([residual_snapshots_SR,residual_snapshots_FC],0)

        if self.sampler == 'ddpm':
            """
            Forward diffusion: tries to guess the epsilon value from snapshot_t using the model.
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
        predicted = self.base_model(residual_snapshots_t, _ts,
                                    Re=torch.cat([Reynolds_number,Reynolds_number],0))
        
        predicted_SR, predicted_FC = torch.split(predicted,predicted.shape[0]//2,0)
        _ts_SR, _ts_FC = torch.split(_ts,predicted.shape[0]//2,0)
        
        predicted_SR = torch.cat([predicted_SR, lowres_snapshots_interpolated], dim=1)
        predicted_SR = self.lowres_model(predicted_SR, _ts_SR)
        
        predicted_FC = torch.cat([predicted_FC, snapshots], dim=1)
        predicted_FC = self.forecast_model(predicted_FC, _ts_FC, s = s)
        

        predicted = torch.cat([predicted_SR,predicted_FC],0)
        
        # Different predictions schemes
        if self.prediction_type == 'x':
            target = residual_snapshots
        elif self.prediction_type == 'eps':
            target = eps
        elif self.prediction_type == 'v':
            target = alpha * eps - sigma * residual_snapshots
        
        return self.criterion(predicted, target)




    def sample(self, n_sample: int, size,
               conditioning_snapshots: torch.Tensor, s: torch.Tensor,
               Reynolds_number: torch.Tensor, device='cuda',superres = True,) -> torch.Tensor:
        """
        Let's sample
        See Alg 2 in https://arxiv.org/pdf/2006.11239
        """
        snapshots_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        
        if superres:
            conditional = nn.functional.interpolate(conditioning_snapshots[0:snapshots_i.shape[0]], 
                                                    size=[snapshots_i.shape[2], snapshots_i.shape[3]], 
                                                    mode='bilinear').to(device)
            model_head = self.lowres_model
        else:
            conditional = conditioning_snapshots.to(device)
            model_head = self.forecast_model

        if self.sampler == 'ddpm':  
            for i in range(self.n_T, 0, -1):
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                
                alpha = self.sqrtab[i]
                sigma = self.sqrtmab[i]

                
                pred = self.base_model(snapshots_i,
                                      torch.tensor(i / self.n_T).to(device).repeat(n_sample),
                                      Re=Reynolds_number)
                pred = torch.cat([pred, conditional], dim=1)
                pred = model_head(pred,torch.tensor(i / self.n_T).to(device).repeat(n_sample),
                                  s = None if superres else s)


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


                pred = self.base_model(snapshots_i,
                                      time.to(device),
                                      Re=Reynolds_number)
                pred = torch.cat([pred, conditional], dim=1)
                pred = model_head(pred, time.to(device),s = None if superres else s)

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



        return snapshots_i + conditional
        
    
def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20.):
    b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
    a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - b
    return -2. * torch.log(torch.tan(a * t + b))


def get_logsnr_alpha_sigma(time):
    logsnr = logsnr_schedule_cosine(time)[:,None,None,None]
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))

    return 0.5*logsnr, alpha, sigma    
    

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
