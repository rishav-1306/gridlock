
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

SEED = 42
np.random.seed(SEED)

print("[1] Loading data...")
train = pd.read_csv('e88186124ec611f1/dataset/train.csv')
test  = pd.read_csv('e88186124ec611f1/dataset/test.csv')
print(f"    Train: {train.shape} | Test: {test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
print("[2] Decoding geohashes...")
def decode_geohash(geohash):
    b32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    lat_r = [-90.0, 90.0]; lon_r = [-180.0, 180.0]; is_even = True
    for c in geohash:
        v = b32.find(c)
        for i in range(4, -1, -1):
            bit = (v >> i) & 1
            if is_even:
                mid = sum(lon_r)/2
                lon_r = [mid, lon_r[1]] if bit else [lon_r[0], mid]
            else:
                mid = sum(lat_r)/2
                lat_r = [mid, lat_r[1]] if bit else [lat_r[0], mid]
            is_even = not is_even
    return sum(lat_r)/2, sum(lon_r)/2

train_ghs = set(train['geohash'].unique())
test_ghs  = set(test['geohash'].unique())
all_ghs   = list(train_ghs | test_ghs)
gh_coords = {gh: decode_geohash(gh) for gh in all_ghs}

train_gh_list = list(train_ghs)
train_coords  = np.array([gh_coords[g] for g in train_gh_list])
gh_mapping = {}
for gh in (test_ghs - train_ghs):
    lat, lon = gh_coords[gh]
    dists = np.sqrt((train_coords[:,0]-lat)**2 + (train_coords[:,1]-lon)**2)
    gh_mapping[gh] = train_gh_list[np.argmin(dists)]
    print(f"    Mapped unseen {gh} -> {gh_mapping[gh]}")

train['lookup_gh'] = train['geohash'].map(lambda x: gh_mapping.get(x, x))
test['lookup_gh']  = test['geohash'].map(lambda x: gh_mapping.get(x, x))

train['lat'] = train['geohash'].map(lambda x: gh_coords[x][0])
train['lon'] = train['geohash'].map(lambda x: gh_coords[x][1])
test['lat']  = test['geohash'].map(lambda x: gh_coords[x][0])
test['lon']  = test['geohash'].map(lambda x: gh_coords[x][1])

# ─────────────────────────────────────────────────────────────────────────────
print("[3] Spatial clusters & neighbors...")
km = KMeans(n_clusters=20, random_state=SEED, n_init=10)
all_coords = np.array([gh_coords[g] for g in all_ghs])
km.fit(all_coords)
gh_cluster = {gh: km.predict([gh_coords[gh]])[0] for gh in all_ghs}
train['spatial_cluster'] = train['geohash'].map(gh_cluster)
test['spatial_cluster']  = test['geohash'].map(gh_cluster)

spatial_neighbors = {}
for gh in all_ghs:
    lat, lon = gh_coords[gh]
    dists = np.sqrt((train_coords[:,0]-lat)**2 + (train_coords[:,1]-lon)**2)
    idxs  = np.argsort(dists)
    nbrs  = [train_gh_list[i] for i in idxs if train_gh_list[i] != gh][:5]
    spatial_neighbors[gh] = nbrs

# ─────────────────────────────────────────────────────────────────────────────
print("[4] Parsing timestamps...")
def parse_ts(df):
    df = df.copy()
    parts = df['timestamp'].str.split(':', expand=True)
    df['hour']      = parts[0].astype(int)
    df['minute']    = parts[1].astype(int)
    df['time_mins'] = df['hour']*60 + df['minute']
    return df

train = parse_ts(train)
test  = parse_ts(test)

# ─────────────────────────────────────────────────────────────────────────────
print("[5] Building pivot tables (day 48 ONLY — matches test reality)...")
# NOTE: We deliberately do NOT use day49 early-hour data as a feature pivot.
# day49 train timestamps = 0-120 min; test timestamps = 135-825 min → zero overlap.
# Using d49e features causes a severe NaN-distribution mismatch between
# train and test, which is why v2 scored ~86 despite high OOF.
day48   = train[train['day']==48][['geohash','time_mins','demand']].copy()
pivot48 = day48.pivot_table(index='geohash', columns='time_mins', values='demand')

TEMP_MEAN = train['Temperature'].mean()

# Per-geohash stats (day48)
gh_stats = day48.groupby('geohash')['demand'].agg(
    ['mean','std','median','max','min']).reset_index()
gh_stats.columns = ['geohash','gh_mean48','gh_std48','gh_median48','gh_max48','gh_min48']

# Per-timestamp stats (day48)
time_stats = day48.groupby('time_mins')['demand'].agg(['mean','std']).reset_index()
time_stats.columns = ['time_mins','ts_mean48','ts_std48']

# Per-(geohash, hour) stats (day48) — captures intra-day rhythm per location
day48['hour'] = day48['time_mins'] // 60
gh_hour_stats = day48.groupby(['geohash','hour'])['demand'].agg(
    ['mean','std']).reset_index()
gh_hour_stats.columns = ['geohash','hour','gh_hour_mean48','gh_hour_std48']

# Per-(cluster, timestamp) stats (day48)
d48_full = train[train['day']==48].copy()
d48_full['hour'] = d48_full['time_mins'] // 60
cluster_stats = d48_full.groupby(
    ['spatial_cluster','time_mins'])['demand'].mean().reset_index()
cluster_stats.columns = ['spatial_cluster','time_mins','cluster_ts_mean']

# Day48 sorted time slots (for rolling features)
ALL_TIMES = sorted(day48['time_mins'].unique())

# ─────────────────────────────────────────────────────────────────────────────
print("[6] Building expanded lag features (day48 only)...")

# Lags to compute: wider range, no d49e
LAGS = [0, -15, 15, -30, 30, -60, 60, -90, 90, -120, 120, -180, 180, -240, 240]

def add_lag_features(df, pivot48):
    df = df.copy()
    # Day48 lags
    for lag in LAGS:
        tgt = df['time_mins'] + lag
        vals = []
        for gh, t in zip(df['lookup_gh'], tgt):
            if gh in pivot48.index and t in pivot48.columns:
                vals.append(pivot48.loc[gh, t])
            else:
                vals.append(np.nan)
        df[f'd48_lag{lag}'] = vals

    # Spatial spillover (5 nearest neighbors, day48 lag0)
    spill_mean, spill_max = [], []
    for gh, t in zip(df['lookup_gh'], df['time_mins']):
        nbrs = spatial_neighbors.get(gh, [])
        vals = [pivot48.loc[n,t] for n in nbrs
                if n in pivot48.index and t in pivot48.columns
                and not np.isnan(pivot48.loc[n,t])]
        spill_mean.append(np.mean(vals) if vals else np.nan)
        spill_max.append(np.max(vals) if vals else np.nan)
    df['d48_spatial_mean'] = spill_mean
    df['d48_spatial_max']  = spill_max

    # Rolling 3-slot window (±1 slot around lag0, day48)
    roll_vals = []
    for gh, t in zip(df['lookup_gh'], df['time_mins']):
        window = [t - 15, t, t + 15]
        vs = [pivot48.loc[gh, w] for w in window
              if gh in pivot48.index and w in pivot48.columns
              and not np.isnan(pivot48.loc[gh, w])]
        roll_vals.append(np.mean(vs) if vs else np.nan)
    df['d48_roll3_mean'] = roll_vals

    # Previous hour anchor (day48) — demand 60 min before at same geohash
    prev_hour_vals = []
    for gh, t in zip(df['lookup_gh'], df['time_mins']):
        t_prev = t - 60
        if gh in pivot48.index and t_prev in pivot48.columns:
            prev_hour_vals.append(pivot48.loc[gh, t_prev])
        else:
            prev_hour_vals.append(np.nan)
    df['d48_prev_hour'] = prev_hour_vals

    return df


def build_features(df):
    df = df.copy()
    df['hour_sin']   = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']   = np.cos(2*np.pi*df['hour']/24)
    df['min_sin']    = np.sin(2*np.pi*df['time_mins']/1440)
    df['min_cos']    = np.cos(2*np.pi*df['time_mins']/1440)
    df['is_rush']      = df['hour'].isin([7,8,9,17,18,19]).astype(int)
    df['is_night']     = df['hour'].isin([0,1,2,3,4,5]).astype(int)
    df['is_afternoon'] = df['hour'].isin([13,14,15,16]).astype(int)
    df['is_morning']   = df['hour'].isin([6,7,8,9,10,11]).astype(int)

    road_map = {'Residential':0,'Street':1,'Highway':2}
    df['RoadType_enc']      = df['RoadType'].map(road_map).fillna(-1)
    df['LargeVehicles_enc'] = (df['LargeVehicles']=='Allowed').astype(int)
    df['Landmarks_enc']     = (df['Landmarks']=='Yes').astype(int)
    weather_map = {'Sunny':0,'Cloudy':1,'Foggy':2,'Rainy':3,'Snowy':4}
    df['Weather_enc']       = df['Weather'].map(weather_map).fillna(-1)
    df['Temperature']       = df['Temperature'].fillna(TEMP_MEAN)
    df['temp_extreme']      = ((df['Temperature']<5)|(df['Temperature']>35)).astype(int)
    df['temp_weather_interaction'] = df['Temperature'] * df['Weather_enc']

    df['gh_prefix3'] = df['geohash'].str[:3]
    df['gh_prefix4'] = df['geohash'].str[:4]

    df = add_lag_features(df, pivot48)

    # Aggregate lag features
    lag48_cols = [c for c in df.columns if c.startswith('d48_lag')]
    df['lag48_mean']   = df[lag48_cols].mean(axis=1)
    df['lag48_max']    = df[lag48_cols].max(axis=1)
    df['lag48_std']    = df[lag48_cols].std(axis=1)
    df['lag48_median'] = df[lag48_cols].median(axis=1)
    df['lag48_min']    = df[lag48_cols].min(axis=1)
    df['lag48_range']  = df['lag48_max'] - df['lag48_min']

    # Trend within day48 itself (lag0 vs lag-60)
    df['d48_trend_1h'] = df['d48_lag0'] - df['d48_lag-60']  # 1-hour momentum
    df['d48_trend_2h'] = df['d48_lag0'] - df['d48_lag-120'] # 2-hour momentum
    df['d48_trend_4h'] = df['d48_lag0'] - df.get('d48_lag-240', np.nan)

    # Merge look-up tables
    df = df.merge(gh_stats,    left_on='lookup_gh', right_on='geohash', how='left',
                  suffixes=('','_gs'))
    if 'geohash_gs' in df.columns: df.drop(columns=['geohash_gs'], inplace=True)

    df = df.merge(time_stats,  on='time_mins', how='left')
    df = df.merge(cluster_stats, on=['spatial_cluster','time_mins'], how='left')
    df = df.merge(gh_hour_stats, left_on=['lookup_gh','hour'],
                  right_on=['geohash','hour'], how='left', suffixes=('','_ghs'))
    if 'geohash_ghs' in df.columns: df.drop(columns=['geohash_ghs'], inplace=True)

    # Interaction / deviation features
    df['lag0_x_ts']         = df['d48_lag0'].fillna(df['gh_mean48']) * df['ts_mean48']
    df['lanes_highway']     = df['NumberofLanes'] * (df['RoadType_enc']==2).astype(int)
    df['gh_ts_deviation']   = df['d48_lag0'] - df['ts_mean48']
    df['lag0_vs_cluster']   = df['d48_lag0'] - df['cluster_ts_mean']
    df['pct_of_day']        = df['time_mins'] / 1440.0
    df['lag0_vs_gh_mean']   = df['d48_lag0'] - df['gh_mean48']
    df['lag0_vs_gh_hour']   = df['d48_lag0'] - df['gh_hour_mean48']

    return df


print("    Building train features...")
train_fe = build_features(train)
print("    Building test features...")
test_fe  = build_features(test)

# Encode categoricals
for src, dst in [('geohash','geohash_enc'),('gh_prefix3','gh3_enc'),
                 ('gh_prefix4','gh4_enc'),('lookup_gh','lookup_gh_enc')]:
    le = LabelEncoder().fit(pd.concat([train_fe[src], test_fe[src]]))
    train_fe[dst] = le.transform(train_fe[src])
    test_fe[dst]  = le.transform(test_fe[src])

DROP = {'Index','geohash','timestamp','day','demand','gh_prefix3','gh_prefix4',
        'RoadType','LargeVehicles','Landmarks','Weather','lookup_gh'}
FEATURES = [c for c in train_fe.columns if c not in DROP and train_fe[c].dtype != 'object']
print(f"    Total features: {len(FEATURES)}")

X      = train_fe[FEATURES].values.astype(np.float32)
y      = train_fe['demand'].values.astype(np.float32)
X_test = test_fe[FEATURES].values.astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
print("[7] Setting up GroupKFold on geohash groups...")
N_FOLDS = 5
groups  = train_fe['geohash_enc'].values
gkf     = GroupKFold(n_splits=N_FOLDS)
folds   = list(gkf.split(X, y, groups))

# ─────────────────────────────────────────────────────────────────────────────
print("[8] Training Neural Network (Residual MLP)...")
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True

    class ResidualBlock(nn.Module):
        def __init__(self, dim, dropout=0.15):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim), nn.BatchNorm1d(dim),
            )
            self.act = nn.GELU()
        def forward(self, x):
            return self.act(x + self.net(x))

    class TrafficMLP(nn.Module):
        def __init__(self, n_features, hidden=256, n_blocks=4, dropout=0.15):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
            )
            self.blocks = nn.ModuleList(
                [ResidualBlock(hidden, dropout) for _ in range(n_blocks)])
            self.head = nn.Sequential(
                nn.Linear(hidden, 64), nn.GELU(),
                nn.Linear(64, 1), nn.Sigmoid()
            )
        def forward(self, x):
            x = self.input_proj(x)
            for blk in self.blocks:
                x = blk(x)
            return self.head(x).squeeze(-1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Using device: {device}")

    scaler_nn = StandardScaler()
    X_nn      = scaler_nn.fit_transform(np.nan_to_num(X, nan=0.0))
    X_test_nn = scaler_nn.transform(np.nan_to_num(X_test, nan=0.0))

    oof_nn  = np.zeros(len(X))
    pred_nn = np.zeros(len(X_test))

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        print(f"    NN Fold {fold_idx+1}/{N_FOLDS}...")
        X_tr = torch.tensor(X_nn[tr_idx],  dtype=torch.float32).to(device)
        y_tr = torch.tensor(y[tr_idx],     dtype=torch.float32).to(device)
        X_vl = torch.tensor(X_nn[val_idx], dtype=torch.float32).to(device)
        y_vl = torch.tensor(y[val_idx],    dtype=torch.float32).to(device)
        Xt   = torch.tensor(X_test_nn,     dtype=torch.float32).to(device)

        ds_tr = TensorDataset(X_tr, y_tr)
        dl_tr = DataLoader(ds_tr, batch_size=2048, shuffle=True)

        model   = TrafficMLP(X_nn.shape[1], hidden=256, n_blocks=4, dropout=0.15).to(device)
        opt     = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        loss_fn = nn.HuberLoss(delta=0.1)

        best_val, best_ep, patience = 1e9, 0, 15
        best_state = None
        for epoch in range(150):
            model.train()
            for xb, yb in dl_tr:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(X_vl).cpu().numpy()
                val_loss = np.mean((val_pred - y[val_idx])**2)
            if val_loss < best_val:
                best_val = val_loss; best_ep = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif epoch - best_ep > patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof_nn[val_idx]  = model(X_vl).cpu().numpy()
            pred_nn         += model(Xt).cpu().numpy() / N_FOLDS

        fold_score = max(0, 100*r2_score(y[val_idx], oof_nn[val_idx]))
        print(f"    NN Fold {fold_idx+1}: {fold_score:.4f}")

    nn_score = max(0, 100*r2_score(y, oof_nn))
    print(f"    NN Overall OOF: {nn_score:.4f}")

except ImportError:
    HAS_TORCH = False
    print("    PyTorch not found — skipping NN.")
    oof_nn  = None
    pred_nn = None

# ─────────────────────────────────────────────────────────────────────────────
print("[9] Training LightGBM...")
lgb_params = {
    'objective': 'regression', 'metric': 'rmse',
    'n_estimators': 5000, 'learning_rate': 0.02,
    'num_leaves': 255, 'min_child_samples': 15,
    'feature_fraction': 0.75, 'bagging_fraction': 0.75, 'bagging_freq': 1,
    'lambda_l1': 0.05, 'lambda_l2': 0.05,
    'verbose': -1, 'random_state': SEED, 'n_jobs': -1,
}
oof_lgb  = np.zeros(len(X))
pred_lgb = np.zeros(len(X_test))
for i, (tr_idx, val_idx) in enumerate(folds):
    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X[tr_idx], y[tr_idx],
          eval_set=[(X[val_idx], y[val_idx])],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    oof_lgb[val_idx]  = m.predict(X[val_idx])
    pred_lgb         += m.predict(X_test) / N_FOLDS
    print(f"    Fold {i+1}: {max(0,100*r2_score(y[val_idx],oof_lgb[val_idx])):.4f}  "
          f"(iter: {m.best_iteration_})")
lgb_score = max(0, 100*r2_score(y, oof_lgb))
print(f"    LightGBM OOF: {lgb_score:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("[10] Training XGBoost...")
xgb_params = {
    'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'n_estimators': 5000, 'learning_rate': 0.02,
    'max_depth': 8, 'min_child_weight': 3,
    'subsample': 0.75, 'colsample_bytree': 0.75,
    'reg_alpha': 0.05, 'reg_lambda': 0.05,
    'tree_method': 'hist', 'random_state': SEED, 'n_jobs': -1,
    'verbosity': 0, 'early_stopping_rounds': 200,
}
oof_xgb  = np.zeros(len(X))
pred_xgb = np.zeros(len(X_test))
for i, (tr_idx, val_idx) in enumerate(folds):
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
    oof_xgb[val_idx]  = m.predict(X[val_idx])
    pred_xgb         += m.predict(X_test) / N_FOLDS
    print(f"    Fold {i+1}: {max(0,100*r2_score(y[val_idx],oof_xgb[val_idx])):.4f}")
xgb_score = max(0, 100*r2_score(y, oof_xgb))
print(f"    XGBoost OOF: {xgb_score:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("[11] Training CatBoost...")
cat_params = {
    'iterations': 5000, 'learning_rate': 0.02,
    'depth': 8, 'loss_function': 'RMSE',
    'random_seed': SEED, 'verbose': False,
    'early_stopping_rounds': 200, 'thread_count': -1,
    'l2_leaf_reg': 3.0,
}
oof_cat    = np.zeros(len(X))
pred_cat   = np.zeros(len(X_test))
X_cat      = np.nan_to_num(X,      nan=-999)
X_test_cat = np.nan_to_num(X_test, nan=-999)
for i, (tr_idx, val_idx) in enumerate(folds):
    m = CatBoostRegressor(**cat_params)
    m.fit(X_cat[tr_idx], y[tr_idx],
          eval_set=(X_cat[val_idx], y[val_idx]), use_best_model=True)
    oof_cat[val_idx]  = m.predict(X_cat[val_idx])
    pred_cat         += m.predict(X_test_cat) / N_FOLDS
    print(f"    Fold {i+1}: {max(0,100*r2_score(y[val_idx],oof_cat[val_idx])):.4f}")
cat_score = max(0, 100*r2_score(y, oof_cat))
print(f"    CatBoost OOF: {cat_score:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("[12] Stacking with Ridge meta-learner...")
if HAS_TORCH and oof_nn is not None:
    meta_X      = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
    meta_X_test = np.column_stack([pred_lgb, pred_xgb, pred_cat, pred_nn])
else:
    meta_X      = np.column_stack([oof_lgb, oof_xgb, oof_cat])
    meta_X_test = np.column_stack([pred_lgb, pred_xgb, pred_cat])

meta_oof  = np.zeros(len(y))
meta_pred = np.zeros(len(pred_lgb))
kf_meta   = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for tr_i, val_i in kf_meta.split(meta_X):
    ridge = Ridge(alpha=0.5)
    ridge.fit(meta_X[tr_i], y[tr_i])
    meta_oof[val_i]  = ridge.predict(meta_X[val_i])
    meta_pred       += ridge.predict(meta_X_test) / N_FOLDS

meta_score = max(0, 100*r2_score(y, meta_oof))

print("\n" + "="*55)
print("  PIPELINE RESULTS  (v3 — no d49e leakage)")
print("="*55)
print(f"  LightGBM OOF:     {lgb_score:.4f}")
print(f"  XGBoost  OOF:     {xgb_score:.4f}")
print(f"  CatBoost OOF:     {cat_score:.4f}")
if HAS_TORCH and oof_nn is not None:
    print(f"  NeuralNet OOF:    {nn_score:.4f}")
print(f"  Stacked  OOF:     {meta_score:.4f}  <- final")
print("="*55)

final_pred = np.clip(meta_pred, 0.0, 1.0)
submission = pd.DataFrame({'Index': test['Index'], 'demand': final_pred})
assert submission.shape == (41778, 2), f"Shape error: {submission.shape}"
submission.to_csv('submission_v3.csv', index=False)
print(f"\nSaved submission_v3.csv  ({submission.shape[0]} rows)")
print(submission.head())
