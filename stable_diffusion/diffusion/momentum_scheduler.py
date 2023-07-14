from typing import List, Optional

import torch
from diffusers import  (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
    PNDMScheduler,
)

def AdamBmixer(order, ets, b=1):
    cur_order = min(order, len(ets))
    if cur_order == 1:
        prime = b * ets[-1]
    elif cur_order == 2:
        prime = ((2+b) * ets[-1] - (2-b)*ets[-2]) / 2
    elif cur_order == 3:
        prime = ((18+5*b) * ets[-1] - (24-8*b) * ets[-2] + (6-1*b) * ets[-3]) / 12
    elif cur_order == 4:
        prime = ((46+9*b) * ets[-1] - (78-19*b) * ets[-2] + (42-5*b) * ets[-3] - (10-b) * ets[-4]) / 24
    elif cur_order == 5:
        prime = ((1650+251*b) * ets[-1] - (3420-646*b) * ets[-2]
                     + (2880-264*b) * ets[-3] - (1380-106*b) * ets[-4]
                     + (270-19*b)* ets[-5]) / 720
    else:
        raise NotImplementedError

    prime = prime/b
    return prime

class PLMSWithHBScheduler():
    """
    PLMS with Polyak's Heavy Ball Momentum (HB) for diffusion ODEs.
    We implement it as a wrapper for schedulers in diffusers (https://github.com/huggingface/diffusers)

    When order is an integer, this method is equivalent to PLMS without momentum.
    """
    def __init__(self, scheduler, order=1):
        self.scheduler = scheduler
        self.ets = []
        self.update_order(order)
        self.mixer = AdamBmixer

    def update_order(self, order):
        self.order = order // 1  + 1 if order%1 > 0 else order // 1
        self.beta = order % 1 if order%1 > 0 else 1
        self.vel = None

    def clear_temp(self):
        self.ets = []
        self.vel = None

    def update_ets(self, val):
        self.ets.append(val)
        if len(self.ets) > self.order:
            self.ets.pop(0)

    def _step_with_momentum(self, grads):
        self.update_ets(grads)
        prime = self.mixer(self.order, self.ets, 1.0)
        self.vel = (1 - self.beta) * self.vel + self.beta * prime
        return self.vel

    def step(
        self,
        grads: torch.FloatTensor,
        timestep: int,
        latents: torch.FloatTensor,
        output_mode: str = "scale",
    ):
        if self.vel is None: self.vel = grads
        if hasattr(self.scheduler, 'sigmas'):
            step_index = (self.scheduler.timesteps == timestep).nonzero().item()
            sigma = self.scheduler.sigmas[step_index]
            sigma_next = self.scheduler.sigmas[step_index + 1]
            del_g = sigma_next - sigma
            update_val = self._step_with_momentum(grads)
            return latents + del_g * update_val
        
        elif isinstance(self.scheduler, (DPMSolverMultistepScheduler, PNDMScheduler, DDIMScheduler)):
            if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler)):
                prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
                alpha_bar_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
            else:
                step_index = (self.scheduler.timesteps == timestep).nonzero().item()
                current_timestep = self.scheduler.timesteps[step_index]
                prev_timestep = 0 if step_index == len(self.scheduler.timesteps) - 1 else self.scheduler.timesteps[step_index + 1]

                alpha_prod_t = self.scheduler.alphas_cumprod[current_timestep]
                alpha_bar_prev = self.scheduler.alphas_cumprod[prev_timestep]

            s0 = torch.sqrt(alpha_prod_t)
            s_1 = torch.sqrt(alpha_bar_prev)
            g0 = torch.sqrt(1-alpha_prod_t)/s0
            g_1 = torch.sqrt(1-alpha_bar_prev)/s_1
            del_g = g_1 - g0

            if self.scheduler.config.prediction_type == "v_prediction":
                beta_prod_t = 1 - alpha_prod_t
                grads = (alpha_prod_t**0.5) * grads + (beta_prod_t**0.5) * latents
            elif self.scheduler.config.prediction_type == "sample":
                beta_prod_t = 1 - alpha_prod_t
                grads = (latents - alpha_prod_t ** (0.5) * grads) / beta_prod_t ** (0.5)
            elif self.scheduler.config.prediction_type != "epsilon":
                raise ValueError( f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon` or `v_prediction`")
            
            update_val = self._step_with_momentum(grads)
            if output_mode in ["scale"]:
                return (latents/s0  + del_g * update_val) * s_1
            elif output_mode in ["back"]:
                return latents + del_g * update_val * s_1
            elif output_mode in ["front"]:
                return latents + del_g * update_val * s0
            else:
                return latents + del_g * update_val
            
        else:
            raise NotImplementedError

class GHVBScheduler(PLMSWithHBScheduler):
    """
    Generalizing Polyak's Heavy Bal (GHVB) for diffusion ODEs.
    We implement it as a wrapper for schedulers in diffusers (https://github.com/huggingface/diffusers)

    When order is an integer, this method is equivalent to PLMS without momentum.
    """
    def _step_with_momentum(self, grads):
        self.vel = (1 - self.beta) * self.vel + self.beta * grads
        self.update_ets(self.vel)
        prime = self.mixer(self.order, self.ets, self.beta)
        return prime

class PLMSWithNTScheduler(PLMSWithHBScheduler):
    """
    PLMS with Nesterov Momentum (NT) for diffusion ODEs.
    We implement it as a wrapper for schedulers in diffusers (https://github.com/huggingface/diffusers)

    When order is an integer, this method is equivalent to PLMS without momentum.
    """
    def _step_with_momentum(self, grads):
        self.update_ets(grads)
        prime = self.mixer(self.order, self.ets, 1.0) # update v^{(2)}
        self.vel = (1 - self.beta) * self.vel + self.beta * prime # update v^{(1)}
        update_val = (1 - self.beta) * self.vel + self.beta * prime # update x
        return update_val

class MomentumDPMSolverMultistepScheduler(DPMSolverMultistepScheduler):
    """
    DPM-Solver++2M with HB momentum.
    Currently support only algorithm_type = "dpmsolver++" and solver_type = "midpoint"

    When beta = 1.0, this method is equivalent to DPM-Solver++2M without momentum.
    """
    def initialize_momentum(self, beta):
        self.vel = None
        self.beta = beta

    def clear_temp(self):
        self.ets = []
        self.vel = None

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        timestep_list: List[int],
        prev_timestep: int,
        sample: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:

        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.config.solver_type == "midpoint":
                diff = (D0 + 0.5 * D1)

                if self.vel is None:
                    self.vel = diff
                else:
                    self.vel = (1-self.beta)*self.vel + self.beta * diff

                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * self.vel
                )
            elif self.config.solver_type == "heun":
                raise NotImplementedError(
                    "{self.config.algorithm_type} with {self.config.solver_type} is currently not supported."
                )
        elif self.config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.config.solver_type == "midpoint":
                raise NotImplementedError(
                    "{self.config.algorithm_type} with {self.config.solver_type} is currently not supported."
                )
            elif self.config.solver_type == "heun":
                raise NotImplementedError(
                    "{self.config.algorithm_type} with {self.config.solver_type} is currently not supported."
                )
        return x_t

class MomentumUniPCMultistepScheduler(UniPCMultistepScheduler):
    """
    UniPC with HB momentum.
    Currently support only self.predict_x0 = True

    When beta = 1.0, this method is equivalent to UniPC without momentum.
    """
    def initialize_momentum(self, beta):
        self.vel_p = None
        self.vel_c = None
        self.beta = beta

    def clear_temp(self):
        self.ets = []
        self.vel_p = None
        self.vel_c = None

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.FloatTensor,
        prev_timestep: int,
        sample: torch.FloatTensor,
        order: int,
    ) -> torch.FloatTensor:

        timestep_list = self.timestep_list
        model_output_list = self.model_outputs

        s0, t = self.timestep_list[-1], prev_timestep
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            x_t = self.solver_p.step(model_output, s0, x).prev_sample
            return x_t

        lambda_t, lambda_s0 = self.lambda_t[t], self.lambda_t[s0]
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]

        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = timestep_list[-(i + 1)]
            mi = model_output_list[-(i + 1)]
            lambda_si = self.lambda_t[si]
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if self.predict_x0:
            if D1s is not None:
                pred_res = torch.einsum("k,bkchw->bchw", rhos_p, D1s)
            else:
                pred_res = 0

            val = ( h_phi_1 * m0 + B_h * pred_res ) /sigma_t /h_phi_1
            if self.vel_p is None:
                self.vel_p = val
            else:
                self.vel_p = (1-self.beta)*self.vel_p + self.beta * val
            self.vel_p = val

            x_t = sigma_t  * (x/ sigma_s0 - alpha_t * self.vel_p * h_phi_1)
        else:
            raise NotImplementedError

        x_t = x_t.to(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.FloatTensor,
        this_timestep: int,
        last_sample: torch.FloatTensor,
        this_sample: torch.FloatTensor,
        order: int,
    ) -> torch.FloatTensor:

        timestep_list = self.timestep_list
        model_output_list = self.model_outputs

        s0, t = timestep_list[-1], this_timestep
        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        lambda_t, lambda_s0 = self.lambda_t[t], self.lambda_t[s0]
        alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]

        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = timestep_list[-(i + 1)]
            mi = model_output_list[-(i + 1)]
            lambda_si = self.lambda_t[si]
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b)

        if self.predict_x0:
            if D1s is not None:
                corr_res = torch.einsum("k,bkchw->bchw", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0

            val = (h_phi_1 * m0 + B_h * (corr_res + rhos_c[-1] * D1_t))/sigma_t/h_phi_1
            if self.vel_c is None:
                self.vel_c = val
            else:
                self.vel_c = (1-self.beta)*self.vel_c + self.beta * val

            x_t = sigma_t  * (x/ sigma_s0 - alpha_t * self.vel_c * h_phi_1)
        else:
            raise NotImplementedError

        x_t = x_t.to(x.dtype)
        return x_t
