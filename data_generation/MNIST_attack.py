import pandas as pd
import os, random
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ========= CONFIG (edit here, then Run) =========
models = [r'../datasets_ECLipsE_Gen_Local/MNIST/trained_NN/mnist_base.pt', r'./datasets/MNIST/trained_NN/mnist_jr.pt']  # path to .pt saved from training
DATA_DIR = r'./datasets'                              # MNIST root
WIDTHS = (128, 128, 128)                              # must match training
ACT = 'elu'                                           # 'relu'|'gelu'|'tanh'|'silu'|'elu'
BATCH = 256
PGD_STEPS = 40
EPS_LIST = [1/2,1/4,1/8,1/16,1/32,1/64,1/128,1/256]                      # test multiple eps values
MAX_BATCHES = None                                    # e.g., 50 for faster rough eval
SEED = 77
# ================================================

def set_seed(s=0):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

class MLP(nn.Module):
    def __init__(self, in_dim=784, widths=(128,128,128), num_classes=10, act='elu'):
        super().__init__()
        A = {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh, 'silu': nn.SiLU, 'elu': nn.ELU}.get(act, nn.ReLU)
        layers, last = [], in_dim
        for w in widths:
            layers += [nn.Linear(last, w), A()]; last = w
        layers += [nn.Linear(last, num_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x.view(x.size(0), -1))

def accuracy(model, loader, device):
    model.eval(); ok = 0; tot = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            ok += (model(x).argmax(1) == y).sum().item(); tot += y.numel()
    return ok / tot

def pgd_l2_attack(model, x, y, eps=1.0, steps=40, alpha=None):
    model.eval()
    if alpha is None: alpha = eps/10
    x0 = x.clone().detach()
    delta = torch.zeros_like(x).normal_(0, 1e-6)
    d = delta.flatten(1); d = d / (d.norm(p=2, dim=1, keepdim=True)+1e-12) * eps
    delta = d.view_as(delta).detach().requires_grad_(True)
    for _ in range(steps):
        loss = F.cross_entropy(model((x0+delta).clamp(0,1)), y)
        g = torch.autograd.grad(loss, delta)[0].flatten(1)
        g = g / (g.norm(p=2, dim=1, keepdim=True)+1e-12)
        delta = (delta.flatten(1) + alpha * g).view_as(delta)
        d = delta.flatten(1); n = d.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        d = d * (eps / n).clamp(max=1.0)
        delta = d.view_as(delta).detach().requires_grad_(True)
    return (x0 + delta).clamp(0,1)


def robust_accuracy(model, loader, device, eps=1.0, steps=40, max_batches=None):
    model.eval(); ok = 0; tot = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv = pgd_l2_attack(model, x, y, eps=eps, steps=steps)
        ok += (model(x_adv).argmax(1) == y).sum().item(); tot += y.numel()
        if (max_batches is not None) and (i+1 >= max_batches): break
    return ok / tot


if __name__ == '__main__':
    set_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    
    # data
    tfm = transforms.ToTensor()
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
    test_loader = DataLoader(test_set, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)
    
    rows = []

    for model_data in models:   
        # model
        model = MLP(784, WIDTHS, 10, ACT).to(device)
        state = torch.load(model_data, map_location=device)
        model.load_state_dict(state)
        model.eval()
        
        # robust accuracy for each eps
        print('\nRobust accuracy (PGD L2):')

        row = {'model': os.path.splitext(os.path.basename(model_data))[0]}
        for eps in EPS_LIST:
            print(f'radius: {eps}')
            rob = robust_accuracy(model, test_loader, device, eps=eps, steps=PGD_STEPS, max_batches=MAX_BATCHES)
            row[eps] = rob  # columns are the eps values
        rows.append(row)

    os.makedirs('results/robust_training', exist_ok=True)

    # Build DataFrame: rows = models, columns = eps
    df = pd.DataFrame(rows).set_index('model')
    # ensure columns appear in the EPS_LIST order
    df = df[EPS_LIST]
    

