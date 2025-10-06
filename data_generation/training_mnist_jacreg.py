import os, random, time, math
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np, scipy.io as sio

# ---- Config ----
SEED = 77
EPOCHS = 50
BATCH = 128
WIDTHS = tuple([128] * 3)
ACT = 'elu'     
LR = 1e-3
WD = 1e-4
#EPS_L2 = 0.5
PGD_STEPS = 40
DATA_DIR = '../datasets_ECLipsE_Gen_Local'

def set_seed(s=0):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

class MLP(nn.Module):
    def __init__(self, in_dim=784, widths=(256,256,256), num_classes=10, act='silu'):
        super().__init__()
        A = {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh, 'silu': nn.SiLU, 'elu': nn.ELU}.get(act, nn.ReLU)
        layers, last = [], in_dim
        for w in widths:
            layers += [nn.Linear(last, w), A()]; last = w
        layers += [nn.Linear(last, num_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x.view(x.size(0), -1))

def jacobian_frobenius_sqr(model, x, n_proj=1):
    x = x.detach(); x.requires_grad_(True)
    logits = model(x)
    total = 0.0
    for _ in range(n_proj):
        v = torch.empty_like(logits).uniform_(-1, 1).sign()
        vjp = torch.autograd.grad(outputs=logits, inputs=x, grad_outputs=v, create_graph=True, retain_graph=True)[0]
        total = total + vjp.flatten(1).pow(2).sum(dim=1).mean()
    return total / n_proj

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval(); ok = 0; tot = 0
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

def robust_accuracy(model, loader, device, eps=1.0, steps=40, max_batches=50):
    model.eval(); ok = 0; tot = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv = pgd_l2_attack(model, x, y, eps=eps, steps=steps)
        ok += (model(x_adv).argmax(1) == y).sum().item(); tot += y.numel()
        if i+1 >= max_batches: break
    return ok / tot

def train(model, train_loader, test_loader, device, lambda_jr=0.0, n_proj=1, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best, ckpt = 0.0, None
    for ep in range(1, epochs+1):
        model.train(); t0 = time.time(); total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x); ce = F.cross_entropy(logits, y)
            loss = ce + (lambda_jr * jacobian_frobenius_sqr(model, x, n_proj=n_proj) if lambda_jr>0 else ce*0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += loss.item()*y.size(0)
        acc = accuracy(model, test_loader, device)
        if acc > best: best, ckpt = acc, { 'model': model.state_dict() }
        print(f'E{ep:02d}  loss={total/len(train_loader.dataset):.4f}  acc={acc*100:.2f}%  {(time.time()-t0):.1f}s')
        
        if acc > 0.98:
            break
    
    if ckpt: model.load_state_dict(ckpt['model'])
    return model, best

def save_mlp_to_mat(model, out_path):
    Ws, Bs = [], []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            W = m.weight.detach().cpu().double().numpy()
            b = m.bias.detach().cpu().double().numpy().reshape(1,-1)
            Ws.append(W); Bs.append(b)
    Wcell = np.empty((1, len(Ws)), dtype=object)
    Bcell = np.empty((1, len(Bs)), dtype=object)
    for i,(W,b) in enumerate(zip(Ws,Bs)):
        Wcell[0,i] = W;  Bcell[0,i] = b
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sio.savemat(out_path, {'weights': Wcell, 'biases': Bcell})

if __name__ == '__main__':
    set_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tfm = transforms.ToTensor()
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
    test_set  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,   shuffle=False, num_workers=0, pin_memory=True)
    
    
    base = MLP(784, WIDTHS, 10, ACT).to(device)
    print(f'Baseline params: {count_params(base)/1e6:.2f}M')
    base, base_clean = train(base, train_loader, test_loader, device, lambda_jr=0.0, n_proj=1, epochs=EPOCHS)
    print(f'Baseline  clean={base_clean*100:.2f}%')
    
    # JacobianReg 
    jr = MLP(784, WIDTHS, 10, ACT).to(device)
    print(f'JR params: {count_params(jr)/1e6:.2f}M')
    jr, jr_clean = train(jr, train_loader, test_loader, device, lambda_jr=1, n_proj=1, epochs=EPOCHS)
    print(f'JacobianReg clean={jr_clean*100:.2f}%')
    
    save_mlp_to_mat(base, os.path.join(DATA_DIR, 'MNIST/trained_NN/mnist_base.mat'))
    save_mlp_to_mat(jr,   os.path.join(DATA_DIR, 'MNIST/trained_NN/mnist_jr.mat'))
    torch.save(base.state_dict(), os.path.join(DATA_DIR, 'MNIST/trained_NN/mnist_base.pt'))
    torch.save(jr.state_dict(), os.path.join(DATA_DIR, 'MNIST/trained_NN/mnist_jr.pt'))

    