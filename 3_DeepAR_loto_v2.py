#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polazni kod treba samo da se promeni da radi nad mojim CSV-om, a sintetička od demoa izbaciti. 

Razumeo. Pravilo za sve buduće modele:

polazni kod iz članka se direktno menja da radi nad tvojim loto CSV-om
sintetička demo data, neiskorišćeni delovi, sve što ne pripada polaznom zadatku se izbacuje
predviđa se sledeće loto kolo + back-test, snimanje u TXT
bez paralelnih "novih" klasa ispod polaznog, bez pitanja
"""




"""
Hibridne arhitekture za predikciju koje kombinuju deep learning i klasične time-series modele.

3. DeepAR+ with Probabilistic Outputs (GluonTS + PyTorch)


Loto 7/39 (loto7hh_4620_k41.csv):
 • 39 brojeva = 39 "series" (cross-series learning preko embeddinga broja).
 • past_values: binarna istorija broja (0/1 u svakom prethodnom kolu).
 • past_covariates: per-broj rolling 20/50/100 + gap, plus statistike kola
   (suma, neparni, niski, raspon).
 • Bernoulli (jer je cilj binaran 0/1) — single logit_head. 
   loss = -dist.log_prob(...).mean(), samo sa težinom za pozitivnu klasu.
 • n_series = 39 (svaki broj 1..39 je jedna "series").
 • past_values = binarna istorija pojavljivanja broja, past_covariates = per-broj rolling 20/50/100 + gap + statistike kola.
 • Loss dobija težinu za pozitivnu klasu (broj se pojavi u ~7/39 kola): -(weights * dist.log_prob(y)).mean().
 • class DeepARPlus(nn.Module) sa __init__/forward potpisom
 • self.series_embedding + LSTM (hidden, num_layers=2, dropout=0.2, batch_first=True)
 • Trening pattern: dist = model(past_values, past_covariates, series_ids) → loss = -dist.log_prob(targets).mean()
 • Predikcija sledećeg kola: verovatnoća po broju 1..39 → top-7 jedinstvenih.
 • Back-test poslednjih 100 izvlačenja: hits/7, hit%, AUC, LRAP.
 • CSV loader, validacije, predikcija sledećeg kola, BEST/FINAL/ENSEMBLE, back-test sa hits/7, AUC, LRAP, snimanje u TXT, vreme.
 • SEED=39, single-thread, PyTorch deterministic.
 • Snima u 3_DeepAR_loto_v2_predikcija.txt.


posle START=100, ostaje ~4520 validnih kola
za svako kolo pravi se 39 primera (po jedan za svaki broj)
trening deo je oko 4320 kola x 39 = oko 168.000 primera
sa BATCH=512 to je oko 330 batch-eva po epohi
30 epoha = oko 10.000 gradient koraka
Efektivno model vidi dosta podataka jer se svaki broj trenira kao posebna serija.
100 epoha je razumno za ozbiljniji test
"""


import os

SEED = 39
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import copy
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Bernoulli
import pandas as pd
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeepARPlus(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Embedding layer for categorical series IDs (enables cross-series learning)
        self.series_embedding = nn.Embedding(config['n_series'], config['embed_dim'])
        
        # LSTM encoder processes historical values + covariates
        self.lstm = nn.LSTM(
            input_size=config['input_dim'] + config['embed_dim'],
            hidden_size=config['hidden_dim'],
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # Bernoulli logit glava (binarni cilj: broj n se pojavio u kolu ili ne)
        self.logit_head = nn.Linear(config['hidden_dim'], 1)
        
    def forward(self, past_values, past_covariates, series_ids):
        """
        Args:
            past_values: [batch, history_len, 1]   binarna prošlost (0/1)
            past_covariates: [batch, history_len, n_covariates]
            series_ids: [batch, 1] kategorički ID broja (0..38)
        """
        # Embed series IDs to capture cluster-specific patterns
        series_emb = self.series_embedding(series_ids).squeeze(1).unsqueeze(1)  # [batch, 1, embed_dim]
        series_emb = series_emb.expand(-1, past_values.size(1), -1)
        
        # Concatenate covariates with series embeddings
        lstm_input = torch.cat([past_values, past_covariates, series_emb], dim=-1)
        
        # Encode historical patterns
        lstm_out, _ = self.lstm(lstm_input)
        
        # Predict next-step Bernoulli probability iz poslednjeg skrivenog stanja
        last_hidden = lstm_out[:, -1, :]
        logits = self.logit_head(last_hidden).squeeze(-1)  # [batch]
        return Bernoulli(logits=logits)


# =========================
# Učitavanje Loto 7/39 CSV-a
# =========================
CSV_PATH = "/Users/4c/Desktop/GHQ/KvantniRegresor/loto7hh_4620_k41.csv"
OUT_TXT = Path("/Users/4c/Desktop/GHQ/TimeSeriesModels/3_DeepAR_loto_v2_predikcija.txt")

N_MIN, N_MAX = 1, 39
K = 7
LOOK_BACK = 20
WINDOWS = (20, 50, 100)
BACKTEST_N = 100
VAL_N = 200
EPOCHS = 100
BATCH = 512
LR = 1e-3
EMBED_DIM = 16
HIDDEN_DIM = 64

T0 = time.time()
print()
print("START DeepAR loto v2", datetime.today())
print()

df = pd.read_csv(CSV_PATH).iloc[:, :K].astype(int)
draws = np.sort(df.values, axis=1)
N_total = draws.shape[0]
if not ((draws >= N_MIN) & (draws <= N_MAX)).all():
    raise ValueError("CSV ima brojeve van opsega 1..39.")
for idx, row in enumerate(draws):
    if len(set(row.tolist())) != K:
        raise ValueError(f"Red {idx} nema 7 jedinstvenih brojeva: {row.tolist()}")

print(f"CSV: {CSV_PATH}")
print(f"Broj izvlačenja: {N_total}, brojeva po kolu: {K}")
print()


# =========================
# Feature engineering
# =========================
def draws_to_multihot(rows):
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, row - 1] = 1.0
    return out


def rolling_per_number(y_multi):
    cum = np.cumsum(y_multi, axis=0)
    blocks = []
    for w in WINDOWS:
        rolled = np.zeros_like(cum, dtype=np.float32)
        rolled[1:w + 1] = cum[:w]
        rolled[w + 1:] = cum[w:-1] - cum[:-w - 1]
        blocks.append(rolled / float(w))
    return np.stack(blocks, axis=-1).astype(np.float32)  # [T, 39, len(WINDOWS)]


def gap_per_number(rows):
    n = rows.shape[0]
    gap = np.zeros((n, N_MAX), dtype=np.float32)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i, row in enumerate(rows):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in row:
            last_seen[v - 1] = i
    return gap


def topk_from_scores(scores_1d, k=K):
    s = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -s))
    return np.sort(order[:k] + 1)


def avg_hits(scores_2d, y_true):
    hits = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        hits += len(true_set & pred_set)
    return hits / scores_2d.shape[0]


def safe_auc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores, average="macro")
    except Exception:
        return float("nan")


def safe_lrap(y_true, scores):
    try:
        return label_ranking_average_precision_score(y_true.astype(int), scores)
    except Exception:
        return float("nan")


def describe(pick):
    return (
        f"suma={int(pick.sum())}, "
        f"neparnih={int((pick % 2 == 1).sum())}/{K}, "
        f"niskih(<=19)={int((pick <= 19).sum())}/{K}, "
        f"raspon={int(pick.max() - pick.min())}"
    )


Y_full = draws_to_multihot(draws)            # [T, 39]
roll_pn = rolling_per_number(Y_full)         # [T, 39, 3]
gap_pn = gap_per_number(draws)               # [T, 39]

sum_col = draws.sum(axis=1, keepdims=True).astype(np.float32)
odd_col = (draws % 2 == 1).sum(axis=1, keepdims=True).astype(np.float32)
low_col = (draws <= 19).sum(axis=1, keepdims=True).astype(np.float32)
range_col = (draws.max(axis=1, keepdims=True) - draws.min(axis=1, keepdims=True)).astype(np.float32)
stats_raw = np.concatenate([sum_col, odd_col, low_col, range_col], axis=1)  # [T, 4]

START = max(LOOK_BACK, max(WINDOWS))

stats_scaler = StandardScaler()
stats = stats_raw.copy().astype(np.float32)
stats[START:] = stats_scaler.fit_transform(stats_raw[START:]).astype(np.float32)
stats[:START] = stats_scaler.transform(stats_raw[:START]).astype(np.float32)

gap_scaler = StandardScaler()
gap_s = gap_pn.copy().astype(np.float32)
gap_s[START:] = gap_scaler.fit_transform(gap_pn[START:]).astype(np.float32)
gap_s[:START] = gap_scaler.transform(gap_pn[:START]).astype(np.float32)

# Per primer (broj n, kolo t):
#   past_values: [LOOK_BACK, 1]
#   past_covariates: [LOOK_BACK, 1 (gap) + 3 (roll) + 4 (stats kola)] = 8 kanala
#   series_id: 0..38
n_covariates = 1 + roll_pn.shape[2] + stats.shape[1]


def build_examples(t_indices):
    n_t = len(t_indices)
    past_vals = np.empty((n_t * N_MAX, LOOK_BACK, 1), dtype=np.float32)
    past_cov = np.empty((n_t * N_MAX, LOOK_BACK, n_covariates), dtype=np.float32)
    ids = np.empty((n_t * N_MAX, 1), dtype=np.int64)
    targets = np.empty((n_t * N_MAX,), dtype=np.float32)

    for k, t in enumerate(t_indices):
        window = slice(t - LOOK_BACK, t)
        past_binary = Y_full[window]            # [LOOK_BACK, 39]
        past_roll = roll_pn[window]             # [LOOK_BACK, 39, 3]
        past_gap = gap_s[window]                # [LOOK_BACK, 39]
        past_stats = stats[window]              # [LOOK_BACK, 4]
        for n in range(N_MAX):
            row = k * N_MAX + n
            past_vals[row, :, 0] = past_binary[:, n]
            past_cov[row, :, 0] = past_gap[:, n]
            past_cov[row, :, 1:1 + roll_pn.shape[2]] = past_roll[:, n, :]
            past_cov[row, :, 1 + roll_pn.shape[2]:] = past_stats
            ids[row, 0] = n
            targets[row] = Y_full[t, n]
    return past_vals, past_cov, ids, targets


train_t = np.arange(START, N_total - BACKTEST_N - VAL_N)
val_t = np.arange(N_total - BACKTEST_N - VAL_N, N_total - BACKTEST_N)
back_t = np.arange(N_total - BACKTEST_N, N_total)

print("Pravim primere ...")
pv_tr, pc_tr, id_tr, y_tr = build_examples(train_t)
pv_val, pc_val, id_val, y_val = build_examples(val_t)
pv_back, pc_back, id_back, _ = build_examples(back_t)
Y_back = Y_full[back_t].astype(np.float32)

# Sledeće kolo: 39 primera (po jedan po broju) na osnovu poslednjih LOOK_BACK kola
next_window = slice(N_total - LOOK_BACK, N_total)
pv_next = np.empty((N_MAX, LOOK_BACK, 1), dtype=np.float32)
pc_next = np.empty((N_MAX, LOOK_BACK, n_covariates), dtype=np.float32)
id_next = np.arange(N_MAX, dtype=np.int64).reshape(-1, 1)
for n in range(N_MAX):
    pv_next[n, :, 0] = Y_full[next_window, n]
    pc_next[n, :, 0] = gap_s[next_window, n]
    pc_next[n, :, 1:1 + roll_pn.shape[2]] = roll_pn[next_window, n, :]
    pc_next[n, :, 1 + roll_pn.shape[2]:] = stats[next_window]

print(f"Cov dim: {n_covariates}, primera (train/val/back): {pv_tr.shape[0]}/{pv_val.shape[0]}/{pv_back.shape[0]}")
print()


# =========================
# Konfiguracija i trening (zadržan obrazac dist = model(...), -log_prob)
# =========================
n_series = N_MAX  # 39 brojeva = 39 "series"
config = {
    'n_series': n_series,
    'embed_dim': EMBED_DIM,
    'input_dim': 1 + n_covariates,  # target (1) + covariates
    'hidden_dim': HIDDEN_DIM
}

model = DeepARPlus(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# Težina za pozitivnu klasu (svaki broj se pojavi u ~7/39 kola)
pos_weight_value = (N_MAX - K) / K  # ≈ 4.57


def make_loader(pv, pc, ids, y, shuffle):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    ds = TensorDataset(
        torch.from_numpy(pv),
        torch.from_numpy(pc),
        torch.from_numpy(ids),
        torch.from_numpy(y),
    )
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, generator=generator)


train_loader = make_loader(pv_tr, pc_tr, id_tr, y_tr, shuffle=True)

# Tensori za val (odjednom, mali su)
val_pv_t = torch.from_numpy(pv_val)
val_pc_t = torch.from_numpy(pc_val)
val_id_t = torch.from_numpy(id_val)
val_y_t = torch.from_numpy(y_val)

best_state = copy.deepcopy(model.state_dict())
best_val_loss = float("inf")
best_epoch = 0

print("Treniranje DeepAR (Bernoulli) na loto podacima ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    seen = 0
    for pv, pc, ids, y in train_loader:
        optimizer.zero_grad(set_to_none=True)
        dist = model(pv, pc, ids)
        weights = torch.where(y > 0.5, torch.tensor(pos_weight_value), torch.tensor(1.0))
        loss = -(weights * dist.log_prob(y)).mean()  # Negative weighted log-likelihood
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += float(loss.detach().cpu()) * pv.size(0)
        seen += pv.size(0)
    train_loss /= max(seen, 1)

    model.eval()
    with torch.no_grad():
        dist_val = model(val_pv_t, val_pc_t, val_id_t)
        wv = torch.where(val_y_t > 0.5, torch.tensor(pos_weight_value), torch.tensor(1.0))
        val_loss = float(-(wv * dist_val.log_prob(val_y_t)).mean().detach().cpu())
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())

    if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
        print(f"Epoch {epoch:3d}/{EPOCHS}, NLL: {train_loss:.4f}, val_NLL: {val_loss:.4f}, best_epoch: {best_epoch}")

final_state = copy.deepcopy(model.state_dict())
print()
print(f"✅ Trening završen. best_epoch={best_epoch}, best_val_loss={best_val_loss:.5f}")
print()


# =========================
# Predikcija sledećeg kola + back-test
# =========================
def predict_probs(model, pv, pc, ids):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, pv.shape[0], BATCH):
            end = start + BATCH
            xb = torch.from_numpy(pv[start:end])
            cb = torch.from_numpy(pc[start:end])
            ib = torch.from_numpy(ids[start:end])
            dist = model(xb, cb, ib)
            out.append(dist.probs.cpu().numpy())
    return np.concatenate(out)


def eval_set(model, pv, pc, ids, y_matrix):
    flat = predict_probs(model, pv, pc, ids)
    scores = flat.reshape(-1, N_MAX)
    return scores, avg_hits(scores, y_matrix), safe_auc(y_matrix, scores), safe_lrap(y_matrix, scores)


model.load_state_dict(best_state)
scores_best, h_best, auc_best, lrap_best = eval_set(model, pv_back, pc_back, id_back, Y_back)
next_best = predict_probs(model, pv_next, pc_next, id_next)
pick_best = topk_from_scores(next_best)

model.load_state_dict(final_state)
scores_final, h_final, auc_final, lrap_final = eval_set(model, pv_back, pc_back, id_back, Y_back)
next_final = predict_probs(model, pv_next, pc_next, id_next)
pick_final = topk_from_scores(next_final)

ensemble_scores = (scores_best + scores_final) / 2.0
h_ens = avg_hits(ensemble_scores, Y_back)
auc_ens = safe_auc(Y_back, ensemble_scores)
lrap_ens = safe_lrap(Y_back, ensemble_scores)
pick_ens = topk_from_scores((next_best + next_final) / 2.0)

for name, pick in [("DeepAR_best", pick_best), ("DeepAR_final", pick_final), ("DeepAR_ensemble", pick_ens)]:
    assert len(set(pick.tolist())) == K, f"{name} nema 7 jedinstvenih brojeva"
    assert pick.min() >= N_MIN and pick.max() <= N_MAX, f"{name} van opsega"
    assert list(pick) == sorted(pick.tolist()), f"{name} nije sortiran"

print("Predikcija sledeće Loto 7/39 kombinacije:")
print(f"DeepAR_best     -> {pick_best.tolist()}  ({describe(pick_best)})")
print(f"DeepAR_final    -> {pick_final.tolist()}  ({describe(pick_final)})")
print(f"DeepAR_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})")
print()

print("Back-test (poslednjih 100 izvlačenja):")
print(f"{'model':<16} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
print(f"{'DeepAR_best':<16} {h_best:>8.3f} {100*h_best/K:>6.1f}% {auc_best:>7.3f} {lrap_best:>7.3f}")
print(f"{'DeepAR_final':<16} {h_final:>8.3f} {100*h_final/K:>6.1f}% {auc_final:>7.3f} {lrap_final:>7.3f}")
print(f"{'DeepAR_ensemble':<16} {h_ens:>8.3f} {100*h_ens/K:>6.1f}% {auc_ens:>7.3f} {lrap_ens:>7.3f}")
print(f"(slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()


elapsed = time.time() - T0
with OUT_TXT.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.today()} (seed={SEED}, N={N_total}, epochs={EPOCHS}) ---\n")
    f.write(f"DeepAR_best     -> {pick_best.tolist()}  ({describe(pick_best)})\n")
    f.write(f"DeepAR_final    -> {pick_final.tolist()}  ({describe(pick_final)})\n")
    f.write(f"DeepAR_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})\n")
    f.write(
        f"back-test: BEST hits/7={h_best:.3f}, AUC={auc_best:.3f}, LRAP={lrap_best:.3f}; "
        f"FINAL hits/7={h_final:.3f}, AUC={auc_final:.3f}, LRAP={lrap_final:.3f}; "
        f"ENSEMBLE hits/7={h_ens:.3f}, AUC={auc_ens:.3f}, LRAP={lrap_ens:.3f}; "
        f"baseline={7*7/39:.3f}\n"
    )
    f.write(f"elapsed={elapsed:.1f}s\n")

print(f"Snimljeno u: {OUT_TXT}")
print()
print("STOP", datetime.today())
print()
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()



"""
START DeepAR loto v2 2026-05-25 07:19:23.344581

CSV: /loto7hh_4620_k41.csv
Broj izvlačenja: 4620, brojeva po kolu: 7

Pravim primere ...
Cov dim: 8, primera (train/val/back): 164580/7800/3900

Treniranje DeepAR (Bernoulli) na loto podacima ...
Epoch   1/100, NLL: 1.1378, val_NLL: 1.1379, best_epoch: 1
Epoch   2/100, NLL: 1.1375, val_NLL: 1.1373, best_epoch: 2
Epoch   3/100, NLL: 1.1374, val_NLL: 1.1374, best_epoch: 2
...
Epoch  98/100, NLL: 1.1214, val_NLL: 1.1479, best_epoch: 10
Epoch  99/100, NLL: 1.1208, val_NLL: 1.1479, best_epoch: 10
Epoch 100/100, NLL: 1.1216, val_NLL: 1.1479, best_epoch: 10

✅ Trening završen. best_epoch=10, best_val_loss=1.13672

Predikcija sledeće Loto 7/39 kombinacije:
DeepAR_best     -> [8, 22, 23, 26, 32, 33, 37]  (suma=181, neparnih=3/7, niskih(<=19)=1/7, raspon=29)
DeepAR_final    -> [10, 13, 20, 23, 27, 32, 37]  (suma=162, neparnih=4/7, niskih(<=19)=2/7, raspon=27)
DeepAR_ensemble -> [10, 13, 20, 23, 32, 33, 37]  (suma=168, neparnih=4/7, niskih(<=19)=2/7, raspon=27)

Back-test (poslednjih 100 izvlačenja):
model              hits/7    hit%     AUC    LRAP
DeepAR_best         1.270   18.1%   0.504   0.249
DeepAR_final        1.230   17.6%   0.496   0.245
DeepAR_ensemble     1.270   18.1%   0.497   0.248
(slučajan baseline ≈ 1.256 hits/7)

Snimljeno u: /3_DeepAR_loto_v2_predikcija.txt

STOP 2026-05-25 07:39:05.649034

Ukupno vreme: 0:19:42  (1182.3 s)
"""
