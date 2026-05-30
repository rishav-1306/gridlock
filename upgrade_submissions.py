import shutil, os, pandas as pd

src = 'd:/gridlock/submission_v4.csv'
targets = ['d:/gridlock/submission.csv', 'd:/gridlock/submission_v2.csv']

if not os.path.exists(src):
    print("ERROR: submission_v4.csv not found. Train v4 may not have finished.")
    exit(1)

df = pd.read_csv(src)
print(f"submission_v4.csv  rows={len(df)}  demand_mean={df['demand'].mean():.6f}  NaNs={df['demand'].isna().sum()}")

for t in targets:
    shutil.copy2(src, t)
    print(f"Upgraded: {t}")

print("Done — submission.csv and submission_v2.csv now contain v4 predictions.")
