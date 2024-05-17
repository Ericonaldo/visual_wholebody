import torch

def GET_TEACHER_ACT_PROB_FUNC(option, iteration_scale):
    TEACHER_ACT_PROB_options = {
        "linear": (lambda x: max(0., 1 - 1 / iteration_scale * x)),
        "exp": (lambda x: max(0., (1 - 1 / iteration_scale) ** x)),
        "tanh": (lambda x: max(0., 0.5 * (1 - torch.tanh(1 / iteration_scale * (x - iteration_scale)))))
    }
    return TEACHER_ACT_PROB_options[option]
    