"""
Created on Fri Jul  5, 2024

@author: ben
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import torch.nn.functional as F
    
class GaussianDiffusionModel(nn.Module):
    def __init__(
        self,
        encoder_model: nn.Module,
        decoder_model: nn.Module,
        lowres_model: nn.Module,
        forecast_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        prediction_type: str,
        sampler:str,
        sample_loss = False,
        criterion: nn.Module = nn.L1Loss(),
    ) -> None:
        super(GaussianDiffusionModel, self).__init__()

        self.sample_loss = sample_loss
        self.prediction_type = prediction_type
        assert self.prediction_type in ['v','eps','x'], "ERROR: Prediction not supported. Options are v, eps, x"
        self.sampler = sampler
        assert self.sampler in ['ddpm','ddim'], "ERROR: Sampler not supported. Options are ddpm and ddim"

        
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
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

        residual_snapshots_SR = snapshots - lowres_snapshots
        residual_snapshots_FC = future_snapshots  - snapshots
        
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
            if self.sample_loss:
                _ts = torch.ones((residual_snapshots_FC.shape[0],)).to(residual_snapshots_FC.device)
                
            else:
                #_ts = torch.randint(0, self.n_T+1, (residual_snapshots_FC.shape[0],)).to(residual_snapshots_FC.device)/self.n_T
                _ts = torch.rand(size=(residual_snapshots_FC.shape[0],)).to(residual_snapshots_FC.device)
                
            eps = torch.randn_like(residual_snapshots_FC)  # eps ~ N(0, 1)
            logsnr, alpha, sigma = get_logsnr_alpha_sigma(_ts)
            
            
        residual_snapshots_t_SR = alpha * residual_snapshots_SR + eps * sigma
        residual_snapshots_t_FC = alpha * residual_snapshots_FC + eps * sigma

        if self.sample_loss:
            x_pred_SR = self.sample(residual_snapshots_t_SR.shape[0],
                                    (1, residual_snapshots_t_SR.shape[2], residual_snapshots_t_SR.shape[3]),
                                    lowres_snapshots, s, Reynolds_number, residual_snapshots_t_SR.device,superres=True)
            
            x_pred_FC = self.sample(residual_snapshots_t_FC.shape[0],
                                    (1, residual_snapshots_t_FC.shape[2], residual_snapshots_t_FC.shape[3]),
                                    snapshots, s, Reynolds_number, residual_snapshots_t_FC.device)


            predicted = torch.cat([x_pred_SR,x_pred_FC])
            target = torch.cat([snapshots,future_snapshots])
            loss_clip = 0
            
        else:
            residual_snapshots_t = torch.cat([residual_snapshots_t_SR,residual_snapshots_t_FC],0)
            predicted, skip_connections = self.encoder_model(residual_snapshots_t,
                                                             torch.cat([_ts,_ts]),
                                                             Re=torch.cat([Reynolds_number,Reynolds_number],0),
                                                             s = torch.cat([torch.zeros_like(s),s],0)
                                                             )
            
            #Add clip loss to align the bottlenecks
            pred1,pred2=torch.split(predicted,predicted.shape[0]//2,0)
            #loss_clip = CLIPLoss(pred1,pred2)
            residual_snapshots_t_SR, residual_snapshots_t_FC = torch.split(residual_snapshots_t,residual_snapshots_t.shape[0]//2,0)
        

            head_SR, skips_SR = self.lowres_model(
                torch.cat([residual_snapshots_t_SR,lowres_snapshots],1),
                _ts, Re = Reynolds_number)
            
            head_FC, skips_FC = self.forecast_model(
                torch.cat([residual_snapshots_t_FC,snapshots],1),
                _ts, Re = Reynolds_number, s = s)

            skip_head = []
            for skip_SR, skip_FC in zip(skips_SR,skips_FC):
                skip_head.append(torch.cat([skip_SR,skip_FC]))

            predicted = self.decoder_model(predicted + torch.cat([head_SR,head_FC]),
                                           skip_connections, skip_head,
                                           torch.cat([_ts,_ts]),
                                           Re=torch.cat([Reynolds_number,Reynolds_number],0),
                                           s = torch.cat([torch.zeros_like(s),s],0))
                                    
        
            # Different predictions schemes
            if self.prediction_type == 'x':
                target = torch.cat([residual_snapshots_SR,residual_snapshots_FC],0)
            elif self.prediction_type == 'eps':            
                target = eps
            elif self.prediction_type == 'v':
                target = torch.cat([alpha * eps - sigma * residual_snapshots_SR,
                                    alpha * eps - sigma * residual_snapshots_FC],0)

            

        
        return self.criterion(predicted, target)
    #+ loss_clip



    #@torch.compile
    def sample(self, n_sample: int, size,               
               conditioning_snapshots: torch.Tensor, s: torch.Tensor,
               Reynolds_number: torch.Tensor, device='cuda',superres = False,snapshots_i = None,) -> torch.Tensor:
        """
        Let's sample
        See Alg 2 in https://arxiv.org/pdf/2006.11239
        """
        if snapshots_i is None:
           snapshots_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        conditional = conditioning_snapshots.to(device)
        if superres:
            model_head = self.lowres_model
        else:            
            model_head = self.forecast_model

        if self.sampler == 'ddpm':  
            for i in range(self.n_T, 0, -1):
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                
                alpha = self.sqrtab[i]
                sigma = self.sqrtmab[i]
                

                pred, skip = self.encoder_model(snapshots_i,
                                                #torch.cat([snapshots_i,conditional],1),
                                                torch.tensor(i / self.n_T).to(device).repeat(n_sample),
                                                Re=Reynolds_number,
                                                s = torch.zeros_like(s) if superres else s
                                                )
                pred_head, skip_head = model_head(torch.cat([snapshots_i,conditional],1),
                                                  torch.tensor(i / self.n_T).to(device).repeat(n_sample),
                                                  Re = Reynolds_number,
                                                  s = None if superres else s)
                
                pred = self.decoder_model(pred + pred_head, skip, skip_head,
                                          torch.tensor(i / self.n_T).to(device).repeat(n_sample),
                                          Re = Reynolds_number,
                                          s = torch.zeros_like(s) if superres else s)



                if self.prediction_type == 'eps':
                    eps = pred
                elif self.prediction_type == 'x':
                    eps = (alpha * pred - snapshots_i) / sigma
                elif self.prediction_type == 'v':
                    eps = alpha * pred + snapshots_i * sigma
                
                snapshots_i = (self.oneover_sqrta[i] * (snapshots_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z)


        elif self.sampler == 'ddim':

            for time_step in range(self.n_T, 0, -1):
                time = torch.ones((n_sample,) ).to(device) * time_step / self.n_T
                time_ = torch.ones((n_sample,) ).to(device) * (time_step-1) / self.n_T
                logsnr, alpha, sigma = get_logsnr_alpha_sigma(time)
                logsnr_, alpha_, sigma_ = get_logsnr_alpha_sigma(time_)
                
                pred, skip = self.encoder_model(snapshots_i,
                                                time.to(device),
                                                Re=Reynolds_number,
                                                s = torch.zeros_like(s) if superres else s
                                                )
                pred_head, skip_head = model_head(torch.cat([snapshots_i,conditional],1),
                                                  time.to(device),
                                                  Re = Reynolds_number,
                                                  s = None if superres else s)
                
                pred = self.decoder_model(pred + pred_head, skip, skip_head,
                                          time.to(device),
                                          Re = Reynolds_number,
                                          s = torch.zeros_like(s) if superres else s)
                
                

                if self.prediction_type == 'v':
                    
                    mean = alpha * snapshots_i - sigma * pred
                    eps = pred * alpha + snapshots_i * sigma

                elif self.prediction_type == 'x':
                    mean = pred
                    eps = (alpha * pred - snapshots_i) / sigma
                elif self.prediction_type == 'eps':
                    mean = alpha * snapshots_i - sigma * pred
                    eps = pred * alpha + snapshots_i * sigma


                # xvar = 0.1/(2. + torch.exp(logsnr))
                # zvar = xvar*(alpha_ - alpha*sigma_/sigma)**2
                # sigma_ = torch.sqrt(sigma_**2 + (256**2/torch.norm(eps,p=2,dim=(1,2,3),keepdim=True))*zvar)
                snapshots_i = alpha_ * mean + eps * sigma_


            #Replace last prediction with the mean value
            snapshots_i = mean

            
        # if conditional.shape[1] >1:
        #     conditional = conditional[:,-1,None]
        return snapshots_i + conditional



def CLIPLoss(emb1,emb2,temperature=1.0):
    #Flatten the inputs with take mean
    B, C, H, W = emb1.shape
    emb1 = emb1.view(B, C, -1).mean(dim=-1)
    emb2 = emb2.view(B, C, -1).mean(dim=-1)

        
    # Calculating the Loss
    logits = (emb1 @ emb2.T) / temperature
    emb1_similarity = emb1 @ emb1.T
    emb2_similarity = emb2 @ emb2.T
    targets = F.softmax(
        (emb1_similarity + emb2_similarity) / 2 * temperature, dim=-1
    )
    emb2_loss = cross_entropy(logits, targets, reduction='none')
    emb1_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss =  (emb1_loss + emb2_loss) / 2.0 # shape: (batch_size)
    return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

    
@torch.compile
def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20., shift = 16.):
    b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
    a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - b
    return -2. * torch.log(torch.tan(a * t + b)*shift)

def inv_logsnr_schedule_cosine(logsnr, logsnr_min=-20., logsnr_max=20.,shift=16.):
    b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
    a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - b
    return torch.atan(torch.exp(-0.5 * logsnr)/shift)/a -b/a


#@torch.compile
def get_logsnr_alpha_sigma(time):
    logsnr = logsnr_schedule_cosine(time)[:,None,None,None]
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))

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
