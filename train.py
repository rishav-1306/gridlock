import pandas as pd
import numpy as np
import os
import pygeohash as pgh
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("Starting Gridlock Traffic Demand Prediction pipeline... START")

# 1. Load Data
print("\n[Step 1] Loading datasets...")
train_path = 'e88186124ec611f1/dataset/train.csv'
test_path  = 'e88186124ec611f1/dataset/test.csv'
sub_path   = 'e88186124ec611f1/dataset/sample_submission.csv'

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
sub   = pd.read_csv(sub_path)
print(f"Loaded Train: {train.shape} | Test: {test.shape}")

# 2. Geohash Decoding and Unseen Geohash Mapping
print("\n[Step 2] Decoding geohashes and mapping unseen ones...")
def decode_geohash(geohash):
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    is_even = True
    for char in geohash:
        val = base32.find(char)
        if val == -1:
            return None, None
        for i in range(4, -1, -1):
            bit = (val >> i) & 1
            if is_even:
                mid = (lon_interval[0] + lon_interval[1]) / 2
                if bit:
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if bit:
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            is_even = not is_even
    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon

train_ghs = set(train['geohash'].unique())
test_ghs = set(test['geohash'].unique())
unseen_ghs = test_ghs - train_ghs
print(f"Unseen test geohashes count: {len(unseen_ghs)}")

# Pre-decode coordinates for all unique geohashes
all_unique_ghs = list(train_ghs | test_ghs)
gh_coords = np.array([decode_geohash(gh) for gh in all_unique_ghs])
gh_to_coord = {gh: coord for gh, coord in zip(all_unique_ghs, gh_coords)}

# Find spatial nearest neighbor in train for any unseen test geohash
train_gh_list = list(train_ghs)
train_coords = np.array([gh_to_coord[gh] for gh in train_gh_list])

gh_mapping = {}
for gh in unseen_ghs:
    lat, lon = gh_to_coord[gh]
    dists = np.sqrt((train_coords[:, 0] - lat)**2 + (train_coords[:, 1] - lon)**2)
    nearest_idx = np.argmin(dists)
    gh_mapping[gh] = train_gh_list[nearest_idx]
    print(f"  Mapped unseen {gh} -> Nearest train neighbor: {gh_mapping[gh]} (dist: {dists.min():.5f})")

# Create lookup_geohash column
train['lookup_geohash'] = train['geohash'].apply(lambda x: gh_mapping.get(x, x))
test['lookup_geohash']  = test['geohash'].apply(lambda x: gh_mapping.get(x, x))

# Precompute the 3 nearest spatial training neighbors for all geohashes (used for spillover feature)
print("Precomputing 3 nearest spatial training neighbors for all geohashes...")
spatial_neighbors_map = {}
for gh in all_unique_ghs:
    lat, lon = gh_to_coord[gh]
    dists = np.sqrt((train_coords[:, 0] - lat)**2 + (train_coords[:, 1] - lon)**2)
    sorted_indices = np.argsort(dists)
    neighbors = []
    for idx in sorted_indices:
        n_gh = train_gh_list[idx]
        if n_gh != gh: # Exclude itself
            neighbors.append(n_gh)
        if len(neighbors) == 3:
            break
    spatial_neighbors_map[gh] = neighbors

# 3. Time Parsing and Core Setup
print("\n[Step 3] Parsing timestamps...")
def parse_ts(df):
    df = df.copy()
    parts = df['timestamp'].str.split(':', expand=True)
    df['hour']      = parts[0].astype(int)
    df['minute']    = parts[1].astype(int)
    df['time_mins'] = df['hour'] * 60 + df['minute']
    return df

train = parse_ts(train)
test  = parse_ts(test)

# Add coordinates to main dataframes
train['lat'] = train['geohash'].apply(lambda x: gh_to_coord[x][0])
train['lon'] = train['geohash'].apply(lambda x: gh_to_coord[x][1])
test['lat']  = test['geohash'].apply(lambda x: gh_to_coord[x][0])
test['lon']  = test['geohash'].apply(lambda x: gh_to_coord[x][1])

# Setup historical pivot tables using train set
day48       = train[train['day']==48][['geohash','time_mins','demand']].copy()
day49_early = train[train['day']==49][['geohash','time_mins','demand']].copy()

pivot48  = day48.pivot_table(index='geohash', columns='time_mins', values='demand')
pivot49e = day49_early.pivot_table(index='geohash', columns='time_mins', values='demand')

# Geohash aggregate stats (baseline locations)
gh_stats = day48.groupby('geohash')['demand'].agg(['mean','std','median','max']).reset_index()
gh_stats.columns = ['geohash','gh_mean48','gh_std48','gh_median48','gh_max48']

# Time average stats
hour_avg = day48.copy()
hour_avg['hour'] = hour_avg['time_mins'] // 60
ts_avg = hour_avg.groupby('hour')['demand'].mean().reset_index().rename(columns={'demand':'ts_avg48'})

TEMP_MEAN = train['Temperature'].mean()

# 4. Feature Engineering function
print("\n[Step 4] Engineering features...")
def add_hist_features_fast(df, pivot48, pivot49e, spatial_neighbors_map, lags=[0,-15,15,-30,30,-60,60,-90,90]):
    df = df.copy()
    for lag in lags:
        target_times = df['time_mins'] + lag
        vals_48, vals_49e = [], []
        for gh, t in zip(df['lookup_geohash'], target_times):
            v48  = pivot48.loc[gh, t]  if (gh in pivot48.index  and t in pivot48.columns)  else np.nan
            v49e = pivot49e.loc[gh, t] if (gh in pivot49e.index and t in pivot49e.columns) else np.nan
            vals_48.append(v48)
            vals_49e.append(v49e)
        df[f'd48_lag{lag}']  = vals_48
        df[f'd49e_lag{lag}'] = vals_49e
        
    # Spatial neighbor spillover for lag 0
    spatial_vals = []
    for gh, t in zip(df['lookup_geohash'], df['time_mins']):
        neighbors = spatial_neighbors_map.get(gh, [])
        neighbor_vals = []
        for n in neighbors:
            if n in pivot48.index and t in pivot48.columns:
                val = pivot48.loc[n, t]
                if not np.isnan(val):
                    neighbor_vals.append(val)
        if neighbor_vals:
            spatial_vals.append(np.mean(neighbor_vals))
        else:
            spatial_vals.append(np.nan)
    df['d48_lag0_spatial_mean'] = spatial_vals
    
    return df

def build_features(df):
    df = df.copy()
    
    # Time cyclical encodings
    df['hour_sin']     = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']     = np.cos(2*np.pi*df['hour']/24)
    df['is_rush']      = df['hour'].isin([7,8,9,10,11,12]).astype(int)
    df['is_night']     = df['hour'].isin([0,1,2,3,4,5]).astype(int)
    df['is_afternoon'] = df['hour'].isin([13,14,15,16]).astype(int)
    
    # Road characteristics Encodings
    df['RoadType_enc']      = df['RoadType'].map({'Residential':0,'Street':1,'Highway':2}).fillna(-1)
    df['LargeVehicles_enc'] = (df['LargeVehicles']=='Allowed').astype(int)
    df['Landmarks_enc']     = (df['Landmarks']=='Yes').astype(int)
    df['Weather_enc']       = df['Weather'].map({'Sunny':0,'Cloudy':1,'Foggy':2,'Rainy':3,'Snowy':4}).fillna(-1)
    
    # Smart Imputation for Temperature based on Weather average
    df['Temperature']       = df['Temperature'].fillna(TEMP_MEAN)
    df['temp_extreme']      = ((df['Temperature']<5)|(df['Temperature']>35)).astype(int)
    
    # Geohash prefixes for location pooling
    df['gh_prefix3'] = df['geohash'].str[:3]
    df['gh_prefix4'] = df['geohash'].str[:4]
    
    # Add historical lag features (including spatial neighboring lag)
    df = add_hist_features_fast(df, pivot48, pivot49e, spatial_neighbors_map)
    
    lag48_cols  = [c for c in df.columns if c.startswith('d48_lag')]
    lag49e_cols = [c for c in df.columns if c.startswith('d49e_lag')]
    
    df['lag48_mean']   = df[lag48_cols].mean(axis=1)
    df['lag48_max']    = df[lag48_cols].max(axis=1)
    df['lag48_std']    = df[lag48_cols].std(axis=1)
    df['lag48_median'] = df[lag48_cols].median(axis=1)
    df['lag49e_mean']  = df[lag49e_cols].mean(axis=1)
    df['lag49e_max']   = df[lag49e_cols].max(axis=1)
    
    # Merge with location-level statistics using lookup_geohash
    df = df.merge(gh_stats, left_on='lookup_geohash', right_on='geohash', how='left', suffixes=('', '_y'))
    # Clean up duplicate geohash columns from merge if any
    if 'geohash_y' in df.columns:
        df = df.drop(columns=['geohash_y'])
        
    df = df.merge(ts_avg, on='hour', how='left')
    
    # Interaction features
    df['lag0_x_ts'] = df['d48_lag0'].fillna(df['gh_mean48']) * df['ts_avg48']
    df['lanes_x_highway'] = df['NumberofLanes'] * (df['RoadType_enc']==2).astype(int)
    
    return df

print("Building features for train...")
train_fe = build_features(train)
print("Building features for test...")
test_fe  = build_features(test)

# Label encode categorical location strings
for col_src, col_dst in [('geohash','geohash_enc'),('gh_prefix3','gh3_enc'),('gh_prefix4','gh4_enc')]:
    le = LabelEncoder().fit(pd.concat([train_fe[col_src], test_fe[col_src]]))
    train_fe[col_dst] = le.transform(train_fe[col_src])
    test_fe[col_dst]  = le.transform(test_fe[col_src])

# Identify final features
DROP = {'Index','geohash','timestamp','day','demand','gh_prefix3','gh_prefix4',
        'RoadType','LargeVehicles','Landmarks','Weather','lookup_geohash'}
FEATURES = [c for c in train_fe.columns if c not in DROP and train_fe[c].dtype != 'object']
print(f"Total features created: {len(FEATURES)}")
print(f"Features: {FEATURES}")

X      = train_fe[FEATURES].values.astype(np.float32)
y      = train_fe['demand'].values.astype(np.float32)
X_test = test_fe[FEATURES].values.astype(np.float32)

# 5. Train Stacking Ensemble (LGBM + XGBoost + CatBoost)
print("\n[Step 5] Setting up 5-Fold Cross-Validation ensembling...")
N_FOLDS  = 5
kf       = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb  = np.zeros(len(X))
pred_lgb = np.zeros(len(X_test))

oof_xgb  = np.zeros(len(X))
pred_xgb = np.zeros(len(X_test))

oof_cat  = np.zeros(len(X))
pred_cat = np.zeros(len(X_test))

# --- LightGBM ---
print("\n--- Training LightGBM ---")
lgb_params = {
    'objective':         'regression',
    'metric':            'rmse',
    'n_estimators':      4000,
    'learning_rate':     0.03,
    'num_leaves':        127,
    'min_child_samples': 20,
    'feature_fraction':  0.8,
    'bagging_fraction':  0.8,
    'bagging_freq':      1,
    'lambda_l1':         0.1,
    'lambda_l2':         0.1,
    'verbose':           -1,
    'random_state':      SEED,
    'n_jobs':            -1,
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X[tr_idx], y[tr_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)]
    )
    oof_lgb[val_idx] = model_lgb.predict(X[val_idx])
    pred_lgb += model_lgb.predict(X_test) / N_FOLDS
    score = max(0, 100 * r2_score(y[val_idx], oof_lgb[val_idx]))
    print(f"  Fold {fold+1}: {score:.4f}  (best iter: {model_lgb.best_iteration_})")

lgb_score = max(0, 100 * r2_score(y, oof_lgb))
print(f"LightGBM Overall OOF Score: {lgb_score:.4f}")

# --- XGBoost ---
print("\n--- Training XGBoost ---")
xgb_params = {
    'objective':         'reg:squarederror',
    'eval_metric':       'rmse',
    'n_estimators':      4000,
    'learning_rate':     0.03,
    'max_depth':         7,
    'min_child_weight':  5,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
    'reg_alpha':         0.1,
    'reg_lambda':        0.1,
    'tree_method':       'hist',
    'random_state':      SEED,
    'n_jobs':            -1,
    'verbosity':         0,
    'early_stopping_rounds': 150,
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X[val_idx])
    pred_xgb += model_xgb.predict(X_test) / N_FOLDS
    score = max(0, 100 * r2_score(y[val_idx], oof_xgb[val_idx]))
    print(f"  Fold {fold+1}: {score:.4f}  (best iter: {model_xgb.best_iteration})")

xgb_score = max(0, 100 * r2_score(y, oof_xgb))
print(f"XGBoost Overall OOF Score: {xgb_score:.4f}")

# --- CatBoost ---
print("\n--- Training CatBoost ---")
cat_params = {
    'iterations':        4000,
    'learning_rate':     0.03,
    'depth':             7,
    'loss_function':     'RMSE',
    'eval_metric':       'RMSE',
    'random_seed':       SEED,
    'verbose':           False,
    'early_stopping_rounds': 150,
    'thread_count':      -1,
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(X[tr_idx], y[tr_idx], eval_set=(X[val_idx], y[val_idx]), use_best_model=True)
    oof_cat[val_idx] = model_cat.predict(X[val_idx])
    pred_cat += model_cat.predict(X_test) / N_FOLDS
    score = max(0, 100 * r2_score(y[val_idx], oof_cat[val_idx]))
    print(f"  Fold {fold+1}: {score:.4f}  (best iter: {model_cat.get_best_iteration()})")

cat_score = max(0, 100 * r2_score(y, oof_cat))
print(f"CatBoost Overall OOF Score: {cat_score:.4f}")

# 6. Ridge Stacking Meta-Learner
print("\n[Step 6] Stacking models using Ridge Regression Meta-Learner...")
meta_X      = np.column_stack([oof_lgb,  oof_xgb,  oof_cat])
meta_X_test = np.column_stack([pred_lgb, pred_xgb, pred_cat])

meta_oof  = np.zeros(len(y))
meta_pred = np.zeros(len(pred_lgb))

for tr_i, val_i in kf.split(meta_X):
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_X[tr_i], y[tr_i])
    meta_oof[val_i] = meta_model.predict(meta_X[val_i])
    meta_pred += meta_model.predict(meta_X_test) / N_FOLDS

meta_score = max(0, 100 * r2_score(y, meta_oof))

print("\n================== PIPELINE RESULTS ==================")
print(f"  LightGBM OOF Score:     {lgb_score:.4f}")
print(f"  XGBoost OOF Score:      {xgb_score:.4f}")
print(f"  CatBoost OOF Score:     {cat_score:.4f}")
print(f"  Stacked Meta-Learner:   {meta_score:.4f} SUCCESS")
print("=======================================================")

# Post-processing: predictions must be valid demand values [0, 1]
final_pred = np.clip(meta_pred, 0.0, 1.0)

# 7. Create Submission File
print("\n[Step 7] Generating submission file...")
submission = pd.DataFrame({'Index': test['Index'], 'demand': final_pred})
assert submission.shape == (41778, 2), f"Error: Submission shape {submission.shape} is incorrect!"
submission.to_csv('submission.csv', index=False)
print("submission.csv successfully generated and validated! DONE")
print(submission.head())
