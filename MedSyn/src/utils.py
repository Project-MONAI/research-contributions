
import math
import torch


#### sample function
def get_alpha_cum(t):
    return torch.where(t >= 0, torch.cos((t + 0.008) / 1.008 * math.pi / 2)**2, 1.0).clamp(1e-4, 1-1e-4)

def get_z_t(x_0, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_cum)*x_0 + torch.sqrt(1-alpha_cum)*eps
    return x_t, eps

def get_eps_x_t(x_0, x_t, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    eps = (x_t - torch.sqrt(alpha_cum)*x_0)/torch.sqrt(1-alpha_cum)
    return eps

def get_x0_x_t(eps, x_t, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    x_0 = (x_t - eps * torch.sqrt(1-alpha_cum)) / torch.sqrt(alpha_cum)
    return x_0

def get_x0_v(v, x_t, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    sigma = 1 - alpha_cum
    return x_t - sigma*v

def get_v_x0(x0, x_t, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    sigma = 1 - alpha_cum
    v = (x_t - x0)/sigma
    return v

def get_z_t_(x_0, t):
    alpha_cum = get_alpha_cum(t)[:,None]
    return torch.sqrt(alpha_cum)*x_0, torch.sqrt(1-alpha_cum)

def get_z_t_via_z_tp1(x_0, z_tp1, t, t_p1):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    alpha_cum_p1 = get_alpha_cum(t_p1)[:, None, None, None, None]
    beta_p1 = 1 - alpha_cum_p1/alpha_cum
    mean_0 = torch.sqrt(alpha_cum)*beta_p1/(1-alpha_cum_p1)
    mean_tp1 = torch.sqrt(1-beta_p1)*(1-alpha_cum)/(1-alpha_cum_p1)

    var = (1-alpha_cum)/(1-alpha_cum_p1)*beta_p1

    return mean_0*x_0 + mean_tp1*z_tp1, var

def ddim_sample(x_0, z_tp1, t, t_p1):
    epsilon = get_eps_x_t(x_0, z_tp1, t_p1)
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    x_t = torch.sqrt(alpha_cum)*x_0 + torch.sqrt(1-alpha_cum)*epsilon
    return x_t

# helpers functions

def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def create_custom_forward(module, return_dict=None):
    def custom_forward(*inputs):
        if return_dict is not None:
            return module(*inputs, return_dict=return_dict)
        else:
            return module(*inputs)

    return custom_forward

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])