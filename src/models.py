"""FT-Transformer / TabMixer / GLU-MLP + SAM(AMP-safe) + 학습 루프 (v16 이식)."""
import math
import time

import numpy as np

from .utils import ensure_prob_finite

_HAVE_TORCH = True
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    _HAVE_TORCH = False


if _HAVE_TORCH:

    class FTTransformerTiny(nn.Module):
        def __init__(self, F_in, C, d_model=160, heads=8, layers=4, dropout=0.25, mlp=2.0, feat_drop=0.10):
            super().__init__()
            self.feat_drop = feat_drop
            self.f_w = nn.Parameter(torch.randn(F_in, d_model) * 0.02)
            self.f_b = nn.Parameter(torch.zeros(F_in, d_model))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads,
                dim_feedforward=int(d_model * mlp),
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, C))

        def forward(self, x):
            tok = x.unsqueeze(-1) * self.f_w.unsqueeze(0) + self.f_b.unsqueeze(0)
            if self.training and self.feat_drop > 0:
                mask = (torch.rand(tok.shape[:2], device=tok.device) >= self.feat_drop).float().unsqueeze(-1)
                tok = tok * mask
            h = self.encoder(tok).mean(dim=1)
            return self.head(h)

    class MixerBlock(nn.Module):
        def __init__(self, F_in, D, tok_exp=2.0, chan_exp=2.0, p=0.25):
            super().__init__()
            self.ln_tok = nn.LayerNorm(D)
            self.tok_fc1 = nn.Linear(F_in, int(F_in * tok_exp))
            self.tok_fc2 = nn.Linear(int(F_in * tok_exp), F_in)
            self.drop = nn.Dropout(p)
            self.act = nn.GELU()
            self.ln_chn = nn.LayerNorm(D)
            self.chn_fc1 = nn.Linear(D, int(D * chan_exp))
            self.chn_fc2 = nn.Linear(int(D * chan_exp), D)

        def forward(self, x):  # x: [B,F,D]
            y = self.ln_tok(x)
            y = y.transpose(1, 2)
            y = self.tok_fc2(self.drop(self.act(self.tok_fc1(y))))
            y = y.transpose(1, 2)
            x = x + self.drop(y)
            z = self.ln_chn(x)
            z = self.chn_fc2(self.drop(self.act(self.chn_fc1(z))))
            x = x + self.drop(z)
            return x

    class TabMixer(nn.Module):
        def __init__(self, F_in, C, d_model=160, layers=5, tok_exp=2.0, chan_exp=2.0, p=0.25, feat_drop=0.10):
            super().__init__()
            self.feat_drop = feat_drop
            self.f_w = nn.Parameter(torch.randn(F_in, d_model) * 0.02)
            self.f_b = nn.Parameter(torch.zeros(F_in, d_model))
            self.blocks = nn.ModuleList([MixerBlock(F_in, d_model, tok_exp, chan_exp, p) for _ in range(layers)])
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, C))

        def forward(self, x):
            tok = x.unsqueeze(-1) * self.f_w.unsqueeze(0) + self.f_b.unsqueeze(0)
            if self.training and self.feat_drop > 0:
                mask = (torch.rand(tok.shape[:2], device=tok.device) >= self.feat_drop).float().unsqueeze(-1)
                tok = tok * mask
            h = tok
            for blk in self.blocks:
                h = blk(h)
            h = h.mean(dim=1)
            return self.head(h)

    class GEGLU(nn.Module):
        def __init__(self, d_in, d_hid, p=0.25):
            super().__init__()
            self.ln = nn.LayerNorm(d_in)
            self.fc = nn.Linear(d_in, d_hid * 2)
            self.proj = nn.Linear(d_hid, d_in)
            self.drop = nn.Dropout(p)

        def forward(self, x):
            z = self.ln(x)
            a, g = self.fc(z).chunk(2, dim=-1)
            a = a * F.gelu(g)
            a = self.drop(a)
            a = self.proj(a)
            a = self.drop(a)
            return x + a

    class GLUMLP(nn.Module):
        def __init__(self, F_in, C, width=512, depth=6, p=0.25):
            super().__init__()
            self.stem = nn.Sequential(nn.LayerNorm(F_in), nn.Linear(F_in, width), nn.SiLU(), nn.Dropout(p))
            self.blocks = nn.ModuleList([GEGLU(width, width * 2, p) for _ in range(depth)])
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, C))

        def forward(self, x):
            h = self.stem(x)
            for blk in self.blocks:
                h = blk(h)
            return self.head(h)

    # ------------- losses -------------
    def cross_entropy_ls(logits, target, eps=0.0, class_weights=None):
        logp = F.log_softmax(logits, dim=-1)
        n = logits.size(-1)
        with torch.no_grad():
            td = torch.zeros_like(logp)
            td.fill_(eps / (n - 1))
            td.scatter_(1, target.unsqueeze(1), 1 - eps)
        loss = -torch.sum(td * logp, dim=1)
        if class_weights is not None:
            loss = loss * class_weights[target]
        return torch.mean(loss)

    def focal_loss(logits, target, gamma=2.0, class_weights=None):
        logp = F.log_softmax(logits, dim=-1)
        p = torch.exp(logp)
        logpt = torch.gather(logp, 1, target.unsqueeze(1)).squeeze(1)
        pt = torch.gather(p, 1, target.unsqueeze(1)).squeeze(1)
        loss = -((1 - pt) ** gamma) * logpt
        if class_weights is not None:
            loss = loss * class_weights[target]
        return torch.mean(loss)

    def symmetric_kl(p, q):
        p = F.log_softmax(p, dim=-1)
        q = F.log_softmax(q, dim=-1)
        kl1 = F.kl_div(p, q.exp(), reduction="batchmean", log_target=False)
        kl2 = F.kl_div(q, p.exp(), reduction="batchmean", log_target=False)
        return 0.5 * (kl1 + kl2)

    # ------------- SAM (AMP-safe) -------------
    class SAM(torch.optim.Optimizer):
        def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
            assert rho >= 0.0, "rho should be non-negative"
            defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
            super().__init__(params, defaults)
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
            self.param_groups = self.base_optimizer.param_groups

        @torch.no_grad()
        def first_step(self, zero_grad=True):
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                    p.add_(e_w)
                    self.state[p]["e_w"] = e_w
            if zero_grad:
                self.zero_grad()

        @torch.no_grad()
        def second_step(self, zero_grad=True, scaler=None):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.sub_(self.state[p]["e_w"])
            if scaler is None:
                self.base_optimizer.step()
            else:
                scaler.step(self.base_optimizer)
            if zero_grad:
                self.zero_grad()

        def step(self, closure=None):
            raise NotImplementedError("Use first_step and second_step instead.")

        def zero_grad(self):
            self.base_optimizer.zero_grad()

        def _grad_norm(self):
            shared_device = self.param_groups[0]["params"][0].device
            return torch.norm(
                torch.stack([
                    p.grad.norm(p=2).to(shared_device)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]), p=2)


ARCHITECTURES = {
    "ft": dict(cls="FTTransformerTiny",
               kwargs=dict(d_model=160, heads=8, layers=4, dropout=0.25, mlp=2.0, feat_drop=0.10)),
    "mixer": dict(cls="TabMixer",
                  kwargs=dict(d_model=160, layers=5, tok_exp=2.0, chan_exp=2.0, p=0.25, feat_drop=0.10)),
    "glu": dict(cls="GLUMLP", kwargs=dict(width=512, depth=6, p=0.25)),
}


def build_model(arch, n_features, n_classes):
    """이름으로 아키텍처를 재구성한다 (체크포인트 복원에 사용)."""
    if arch not in ARCHITECTURES:
        raise KeyError(f"알 수 없는 아키텍처: {arch} (가능: {sorted(ARCHITECTURES)})")
    spec = ARCHITECTURES[arch]
    cls = globals()[spec["cls"]]
    return cls(n_features, n_classes, **spec["kwargs"])


class TorchPredictor:
    """학습된 모델 + 정규화 통계 + 온도를 함께 들고 다니는 호출 가능 예측기.

    학습 코드와 추론 코드가 같은 객체를 쓰므로 전처리가 갈라질 수 없다.
    `temperature`는 학습 시 내부 홀드아웃에서 정하고, 예측마다 자동 적용된다.
    """

    def __init__(self, model, mu, std, arch=None, n_features=None, n_classes=None,
                 temperature=1.0, batch_size=512):
        self.model = model
        self.mu = np.asarray(mu, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.arch = arch
        self.n_features = n_features
        self.n_classes = n_classes
        self.temperature = float(temperature)
        self.batch_size = batch_size

    @property
    def device(self):
        return next(self.model.parameters()).device

    def __call__(self, Xt, mc_passes=1, enable_dropout=False, tta_noise_std=0.0,
                 apply_temperature=True):
        device = self.device
        Z = np.nan_to_num((np.asarray(Xt) - self.mu) / self.std, posinf=0.0, neginf=0.0)
        Xt_t = torch.tensor(Z, dtype=torch.float32, device=device)
        n_pass = mc_passes if (enable_dropout and mc_passes > 1) else 1
        self.model.train() if n_pass > 1 else self.model.eval()
        preds = []
        with torch.no_grad():
            for _ in range(n_pass):
                out = []
                for i in range(0, Xt_t.size(0), self.batch_size):
                    xb = Xt_t[i:i + self.batch_size]
                    if tta_noise_std > 0:
                        xb = xb + torch.randn_like(xb) * tta_noise_std
                    logits = self.model(xb).float()
                    if apply_temperature and self.temperature != 1.0:
                        logits = logits / self.temperature
                    out.append(F.softmax(logits, dim=-1).cpu().numpy())
                preds.append(np.vstack(out))
        self.model.eval()
        return ensure_prob_finite(np.mean(preds, axis=0))

    def save(self, path):
        if self.arch is None:
            raise ValueError("arch 가 없는 예측기는 저장할 수 없습니다.")
        torch.save({
            "arch": self.arch,
            "n_features": int(self.n_features),
            "n_classes": int(self.n_classes),
            "temperature": float(self.temperature),
            "mu": self.mu,
            "std": self.std,
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
        }, path)

    @staticmethod
    def load(path, device=None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(ckpt["arch"], ckpt["n_features"], ckpt["n_classes"])
        model.load_state_dict(ckpt["state_dict"])
        device = device or resolve_device()
        model.to(device).eval()
        return TorchPredictor(model, ckpt["mu"], ckpt["std"], arch=ckpt["arch"],
                              n_features=ckpt["n_features"], n_classes=ckpt["n_classes"],
                              temperature=ckpt["temperature"])


def resolve_device():
    if not _HAVE_TORCH:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fit_torch_model(
    model, Xtr, ytr, Xva, yva,
    lr=3e-3, wd=1e-4, epochs=120, bs=256,
    warm=8, patience=18, ls=0.05, live=False, name="DL",
    class_weights=None, loss_mode="ce", focal_gamma=2.0,
    mixup_alpha=0.0, rdrop_alpha=0.0, use_sam=False, sam_rho=0.05,
    mc_dropout=False, tta_noise_std=0.0, amp_enabled=True,
    arch=None, n_classes=None,
):
    """v16의 학습 루프. best-F1 스냅샷 복원 + predict 클로저 반환."""
    assert _HAVE_TORCH, "PyTorch not available."
    device = resolve_device()
    model = model.to(device)
    if live:
        print(f"[{name}] epochs={epochs}, bs={bs}, device={device}, amp={amp_enabled}, sam={use_sam}")

    mu = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    Ztr = np.nan_to_num((Xtr - mu) / std, posinf=0.0, neginf=0.0)
    Zva = np.nan_to_num((Xva - mu) / std, posinf=0.0, neginf=0.0)

    Xtr_t = torch.tensor(Ztr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Zva, dtype=torch.float32, device=device)

    if use_sam:
        opt = SAM(model.parameters(), torch.optim.AdamW, lr=lr, weight_decay=wd, rho=sam_rho)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    cw_tensor = None
    if class_weights is not None:
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    def criterion(logits, target):
        if loss_mode == "focal":
            return focal_loss(logits, target, gamma=focal_gamma, class_weights=cw_tensor)
        return cross_entropy_ls(logits, target, eps=ls, class_weights=cw_tensor)

    def loader(X, Y, shuffle=True):
        idx = torch.arange(X.size(0), device=device)
        if shuffle:
            idx = idx[torch.randperm(idx.numel(), device=device)]
        for i in range(0, X.size(0), bs):
            j = idx[i:i + bs]
            yield X[j], Y[j]

    def set_lr(base_lr, ep):
        if ep < warm:
            lr_now = base_lr * (ep + 1) / max(1, warm)
        else:
            t = (ep - warm) / max(1, (epochs - warm))
            lr_now = 0.5 * base_lr * (1 + math.cos(math.pi * t))
        for g in opt.param_groups:
            g["lr"] = lr_now

    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and device == "cuda"))
    from sklearn.metrics import f1_score as _f1

    best_f1, best = -1.0, None
    noimp = 0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        set_lr(lr, ep)
        tr_loss, nb = 0.0, 0

        for xb, yb in loader(Xtr_t, ytr_t, True):
            if mixup_alpha and mixup_alpha > 0.0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(xb.size(0), device=device)
                xb_in = lam * xb + (1 - lam) * xb[idx]

                def loss_fn(out):
                    ce = lambda o, t: cross_entropy_ls(o, t, eps=0.0, class_weights=cw_tensor)
                    return lam * ce(out, yb) + (1 - lam) * ce(out, yb[idx])
            else:
                xb_in = xb

                def loss_fn(out):
                    return criterion(out, yb)

            if use_sam:
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    out1 = model(xb_in)
                    loss1 = loss_fn(out1)
                scaler.scale(loss1).backward()
                opt.first_step(zero_grad=True)

                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    out2 = model(xb_in)
                    loss2 = loss_fn(out2)
                    if rdrop_alpha and rdrop_alpha > 0.0:
                        loss2 = loss2 + rdrop_alpha * symmetric_kl(out1.detach(), out2)
                scaler.scale(loss2).backward()
                scaler.unscale_(opt.base_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.second_step(zero_grad=True, scaler=scaler)
                scaler.update()
                cur_loss = 0.5 * (loss1.detach() + loss2.detach()).item()
            else:
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    out = model(xb_in)
                    loss = loss_fn(out)
                    if rdrop_alpha and rdrop_alpha > 0.0:
                        out_b = model(xb_in)
                        loss = loss + rdrop_alpha * symmetric_kl(out, out_b)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                cur_loss = loss.detach().item()

            tr_loss += cur_loss
            nb += 1

        # validation
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, Xva_t.size(0), 512):
                with torch.amp.autocast("cuda", enabled=False):
                    logits = model(Xva_t[i:i + 512])
                outs.append(F.softmax(logits, dim=-1).cpu().numpy())
        pva = ensure_prob_finite(np.vstack(outs))
        f1 = _f1(yva, pva.argmax(1), average="macro")

        if live:
            print(f"[{name}][ep {ep:03d}] train_loss={tr_loss/max(1,nb):.4f} | val_f1={f1:.4f} | {time.time()-t0:.1f}s")

        if f1 > best_f1 + 1e-6:
            best_f1 = f1
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            noimp = 0
        else:
            noimp += 1
        if noimp >= patience:
            break

    if best is None:
        best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best)

    predictor = TorchPredictor(model, mu, std, arch=arch, n_features=Xtr.shape[1],
                               n_classes=int(np.max(ytr)) + 1 if n_classes is None else n_classes)
    return model, predictor


DISPLAY_NAMES = {"ft": "FT-Trans", "mixer": "TabMixer", "glu": "GLU-MLP"}


def fit_architecture(arch, Xtr, ytr, Xva, yva, K, **kw):
    """이름으로 아키텍처를 만들고 학습한다. (model, TorchPredictor) 반환."""
    model = build_model(arch, Xtr.shape[1], K)
    return fit_torch_model(model, Xtr, ytr, Xva, yva, name=DISPLAY_NAMES.get(arch, arch),
                           arch=arch, n_classes=K, **kw)


def fit_ft(Xtr, ytr, Xva, yva, K, **kw):
    return fit_architecture("ft", Xtr, ytr, Xva, yva, K, **kw)


def fit_tabmixer(Xtr, ytr, Xva, yva, K, **kw):
    return fit_architecture("mixer", Xtr, ytr, Xva, yva, K, **kw)


def fit_glumlp(Xtr, ytr, Xva, yva, K, **kw):
    return fit_architecture("glu", Xtr, ytr, Xva, yva, K, **kw)
