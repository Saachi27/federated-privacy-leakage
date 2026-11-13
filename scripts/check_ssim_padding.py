"""Check how much the zero-padded SSIM in src/metrics.py differs from standard SSIM."""
import sys, warnings, numpy as np, torch, torch.nn.functional as F
warnings.filterwarnings("ignore"); torch.set_num_threads(1)
sys.path.insert(0, ".")
from multiprocessing import Pool
from scipy.optimize import linear_sum_assignment
from src import metrics, attack
from scripts.evaluate import load_images
from scripts.attack_noise_sum import noisy_mean_gradient
from src.evaluation import build_attack_model

def ssim_valid(a, b):
    a, b = a.clamp(0,1)[None].float(), b.clamp(0,1)[None].float()
    w = metrics._gaussian_window(11, 1.5, 1, a.device, a.dtype)
    mu_a, mu_b = F.conv2d(a, w), F.conv2d(b, w)
    sa = F.conv2d(a*a, w)-mu_a**2; sb = F.conv2d(b*b, w)-mu_b**2; sab = F.conv2d(a*b, w)-mu_a*mu_b
    return (((2*mu_a*mu_b+1e-4)*(2*sab+9e-4))/((mu_a**2+mu_b**2+1e-4)*(sa+sb+9e-4))).mean().item()

def both(recon, originals):
    recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=0.0)
    out = {}
    for name, f in (("ours", lambda r, o: metrics.ssim(r.clamp(0,1), o)), ("std", ssim_valid)):
        S = np.array([[f(recon[i], originals[j]) for j in range(len(originals))] for i in range(len(recon))])
        r, c = linear_sum_assignment(S, maximize=True); out[name] = S[r, c].tolist()
    return out

imgs, meta = load_images("mnist", 66); imgs = imgs[50:66]
X = [i[0] for i in imgs]; Y = [i[1] for i in imgs]

def job(t):
    kind, k, g, sigma, att = t
    model = build_attack_model(meta, 0)
    xs, ys = X[g*k:(g+1)*k], Y[g*k:(g+1)*k]
    per = [attack.compute_gradient(model, x[None], torch.tensor([y])) for x, y in zip(xs, ys)]
    obs, _ = noisy_mean_gradient(per, 30.0, sigma, seed=g)
    shape = (k,) + tuple(xs[0].shape)
    if att == "basic":
        rec, _ = attack.reconstruct(model, obs, shape, 10, label=ys, iters=300, seed=0)
    else:
        rec, _ = attack.reconstruct_cosine(model, obs, shape, ys, iters=3000, tv_weight=1e-3, seed=0)
    return (k, sigma, att), both(rec, xs)

if __name__ == '__main__':
    tasks = [("sum", 8, g, 0.0, "basic") for g in range(2)] + [("sum", 8, g, 0.0, "cosine") for g in range(2)] \
          + [("sum", 1, g, 1e-3, "cosine") for g in range(16)] + [("sum", 1, g, 1e-3, "basic") for g in range(16)] \
          + [("sum", 4, g, 0.0, "cosine") for g in range(4)]
    res = {}
    with Pool(8) as p:
        for key, v in p.imap_unordered(job, tasks):
            d = res.setdefault(key, {"ours": [], "std": []}); d["ours"] += v["ours"]; d["std"] += v["std"]
    # Baselines under both
    black = torch.zeros_like(X[0])
    print(f"BASELINE all black: ours {np.mean([metrics.ssim(black, x) for x in X]):.3f}  std {np.mean([ssim_valid(black, x) for x in X]):.3f}")
    _, test, _ = __import__('src.data', fromlist=['x']).load_datasets("mnist", normalize=False)
    rng = np.random.default_rng(1)
    for k in (1, 4, 8):
        o, s = [], []
        for g in range(16 // k):
            others = torch.stack([test[int(i)][0] for i in rng.choice(np.arange(100, 10000), k, replace=False)])
            b = both(others, X[g*k:(g+1)*k]); o += b["ours"]; s += b["std"]
        print(f"BASELINE unrelated best-matched K={k}: ours {np.mean(o):.3f}  std {np.mean(s):.3f}")
    for key in sorted(res):
        d = res[key]; o, s = np.array(d["ours"]), np.array(d["std"])
        print(f"K={key[0]} sigma={key[1]:g} {key[2]:6s}: ours {o.mean():.3f} (>=0.9: {(o>=0.9).mean()*100:.0f}%)   std {s.mean():.3f} (>=0.9: {(s>=0.9).mean()*100:.0f}%)")
