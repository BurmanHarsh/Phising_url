import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

file_path = "PhiUSIIL_raw.csv" if os.path.exists("PhiUSIIL_raw.csv") else "assignment/PhiUSIIL_raw.csv"
df = pd.read_csv(file_path)

print("Initial Data Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna()

print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

fig1, axes1 = plt.subplots(1, 3, figsize=(13, 4.2))

counts = df['label'].value_counts().sort_index() 
total = len(df)
classes = ['Phishing (0)', 'Legitimate (1)']
bar_colors = ['#4a7298', '#d9534f']
bars1 = axes1[0].bar(classes, [counts[0], counts[1]], color=bar_colors, width=0.6)
axes1[0].set_title('Class balance', fontsize=12, pad=10, fontweight='bold')
axes1[0].set_ylim(0, max(counts) * 1.15)
for bar in bars1:
    h = bar.get_height()
    pct = (h / total) * 100
    axes1[0].text(bar.get_x() + bar.get_width() / 2, h + total * 0.01, f'{int(h):,} ({pct:.1f}%)',
                  ha='center', va='bottom', fontsize=9.5)
ht_group = df.groupby('HasTitle')['label'].apply(lambda s: (s == 0).mean() * 100)
bars2 = axes1[1].bar(['No', 'Yes'], [ht_group.get(0, 0), ht_group.get(1, 0)], color='#4a7298', width=0.6)
axes1[1].set_title('Phishing rate by HasTitle', fontsize=12, pad=10, fontweight='bold')
axes1[1].set_ylabel('% Phishing', fontsize=10)
axes1[1].set_ylim(0, max(ht_group) * 1.15)
for bar in bars2:
    h = bar.get_height()
    axes1[1].text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=9.5)

top_tlds = ['com', 'net', 'org']
tld_group = df[df['TLD'].isin(top_tlds)].groupby('TLD')['label'].apply(lambda s: (s == 0).mean() * 100)
bars3 = axes1[2].bar(top_tlds, [tld_group.get(t, 0) for t in top_tlds], color='#4a7298', width=0.6)
axes1[2].set_title('Phishing rate by TLD', fontsize=12, pad=10, fontweight='bold')
axes1[2].set_ylim(0, max(tld_group) * 1.18)
for bar in bars3:
    h = bar.get_height()
    axes1[2].text(bar.get_x() + bar.get_width() / 2, h + 0.8, f'{h:.1f}%', ha='center', va='bottom', fontsize=9.5)

for ax in axes1:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=10)

plt.figtext(0.5, 0.01, 'Figure 1: Class balance and phishing rate by HasTitle and TLD (training data).',
            ha='center', fontsize=11, style='italic', color='#333333')
plt.tight_layout(rect=[0, 0.05, 1, 1])
out_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else "."
fig1_path = os.path.join(out_dir, 'phishing_figure1.png')
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
print("Figure 1 saved to:", fig1_path)
plt.show()

# ----------------- FIGURE 2 -----------------
skewed_features = ['DomainLength', 'LetterRatioInURL', 'SpacialCharRatioInURL', 'URLCharProb']
fig2, axes2 = plt.subplots(1, 4, figsize=(14, 4.2))

for ax, feat in zip(axes2, skewed_features):
    ax.boxplot(
        df[feat].dropna(),
        patch_artist=True,
        widths=0.35,
        boxprops=dict(facecolor='#8ecae6', edgecolor='black', linewidth=1.2),
        medianprops=dict(color='#e67e22', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
        flierprops=dict(marker='o', markersize=3, markerfacecolor='#e74c3c', markeredgecolor='#e74c3c', alpha=0.7)
    )
    ax.set_title(feat, fontsize=11, pad=8)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=9.5)

plt.figtext(0.5, 0.01, 'Figure 2: Box plots of skewed numeric features; red points are IQR outliers (retained).',
            ha='center', fontsize=11, style='italic', color='#333333')
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig2_path = os.path.join(out_dir, 'phishing_figure2.png')
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
print("Figure 2 saved to:", fig2_path)
plt.show()

drop_cols = ['FILENAME', 'URL', 'Domain', 'Title']
for col in drop_cols:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

target_col = 'label' if 'label' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

selector = SelectKBest(score_func=f_classif, k=15)
selector.fit(X, y)
selected_features = X.columns[selector.get_support()]

print("\nSelected Top 15 Features:")
print(selected_features.tolist())

save_path = "PhiUSIIL_cleaned.csv" if os.path.exists("PhiUSIIL_raw.csv") else "assignment/PhiUSIIL_cleaned.csv"
df_cleaned = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
df_cleaned.to_csv(save_path, index=False)

print("\nCleaned dataset saved successfully to:", save_path)
print("Final Shape:", df_cleaned.shape)
