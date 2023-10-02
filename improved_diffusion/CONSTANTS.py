import torch

# Device information
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available() and DEVICE != 'cuda:0':
    raise ValueError('This code only works on a single GPU.')
DEVICE_ID = (torch.device("cuda:0") if torch.cuda.is_available()
             else torch.device("cpu"))

# NOTE only used if use_fp16 is True
INITIAL_LOG_LOSS_SCALE = 20.0

LABELS_SAR = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']

# A dictionary for the markers associated to the various sampling methods.
# https://matplotlib.org/stable/api/markers_api.html
DIC_LINE = {
    'original': 'go-',
    'variance': '^-',
    'DDIM-var': 'rs-',
    'ODE': 'p-'}

# Short cuts for plotting purposes
ODE_METHODS_SC = {
    'singlestep': 'S',
    'singlestep_fixed': 'SF',
    'multistep': 'M',
    'adaptive': 'A'}

ODE_SKIP_SC = {
    'time-uniform': 'TU',
    'logSNR': 'LSNR',
    'time-quadratic': 'TQ'}