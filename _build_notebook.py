"""Generate notebook riset DSCO 2025 (deteksi URL phishing, metrik ROC AUC) dan script runner Kaggle.

Output:
- kaggle-notebook/dsco-2025-phishing-url.ipynb  (notebook bersih, di-push sebagai notebook Kaggle)
- kaggle/run_on_kaggle.py                        (script kernel terpisah untuk mengambil notebook ber-output)
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
NB_NAME = "dsco-2025-phishing-url.ipynb"
FINDINGS = json.loads((ROOT / "_findings.json").read_text()) if (ROOT / "_findings.json").exists() else {}


def T(key: str) -> str:
    return FINDINGS.get(key, "_(temuan diisi setelah run pertama)_")


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ════════════════════════════════════════════════════════════════════ Header
md(
    f"""
# Mendeteksi URL Phishing: Apa yang Sebenarnya Dipelajari Model?

| | |
| --- | --- |
| **Kompetisi** | [Data Science Olympiads (DSCO) 2025](https://www.kaggle.com/competitions/dsco) (Data Science Club, Kaggle, November–Desember 2025) |
| **Tujuan** | Membedakan URL *legitimate* (1) dan *phishing* (0) dari string URL dan 16 fitur leksikal (metrik ROC AUC) |
| **Data** | 80.974 URL berlabel dan 20.244 URL uji, turunan dataset LegitPhish |
| **Penulis** | Fauzan Ahsanudin Alfikri · [Kaggle: fauzan2](https://www.kaggle.com/fauzan2) |
| **Tanggal** | Disusun ulang November 2026, menggantikan notebook kompetisi Desember 2025 |
| **Lingkungan** | Dijalankan penuh di Kaggle (CPU) |

**Pertanyaan penelitian.** Skor AUC di atas 0,99 mudah dicapai di kompetisi ini. Apakah itu berarti model memahami
phishing, atau hanya mengenali cara dataset ini disusun?

**Ringkasan temuan.**
{T("summary")}

**Alur:** 1. Setup → 2. Memahami data → 3. EDA → 4. Preprocessing → 5. Baseline → 6. Model →
7. Interpretasi dan analisis error → 8. Submission → 9. Kesimpulan
"""
)

md(
    """
## 1. Setup

Semua import, parameter, dan path ada di sel ini. Fitur yang dihitung ulang dari URL ada di `src/features`, dan
cross-validation ada di `src/models`.
"""
)

code(
    """
import os
import shutil
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SRC_ROOT = Path.cwd().parent
for r, _, f in os.walk("/kaggle/input", followlinks=True):
    if Path(r).name == "features" and "url_features.py" in f:
        pkg = Path(r).parent
        if pkg.name == "src":
            SRC_ROOT = pkg.parent
        else:
            SRC_ROOT = Path("/tmp/project")
            shutil.copytree(pkg, SRC_ROOT / "src", dirs_exist_ok=True)
        break
sys.path.insert(0, str(SRC_ROOT))
from src.data.load_data import ID, PROVIDED, TARGET, find_data_dir, load
from src.features.url_features import mismatch, recompute, url_type
from src.models.train_model import auc_by_seed, cv_predict, lgb_model, logreg_model, rank_mean, tfidf_model
from src.visualization.plots import PALETTE, pct_axis, save_fig, set_style

# ── Parameter eksperimen
SEEDS = range(3)                 # Stratified 5-Fold diulang dengan 3 seed

# ── Path
DATA_DIR = find_data_dir(local="../data/raw")
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("../reports")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
set_style()
C1, C2, C3, CN = PALETTE["normal"], PALETTE["fraud"], PALETTE["accent"], PALETTE["neutral"]
T0 = time.time()
print("DATA_DIR:", DATA_DIR, "| OUT_DIR:", OUT_DIR)
"""
)

# ════════════════════════════════════════════════════════════════════ Data
md(
    """
## 2. Memahami data

Setiap baris adalah satu URL beserta 16 fitur leksikal yang sudah dihitung penyelenggara: panjang URL, jumlah titik,
entropi, jumlah token, panjang path, jumlah digit, dan seterusnya. Label `ClassLabel` bernilai 1 untuk URL legitimate dan
0 untuk phishing.
"""
)

code(
    """
train, test, sample = load(DATA_DIR)
y = train[TARGET].values
print(f"Train: {train.shape} | Test: {test.shape} | nilai kosong train/test: {train.isna().sum().sum()}/{test.isna().sum().sum()}")
print(f"Legitimate: {y.mean():.1%} | URL duplikat di train: {train.URL.duplicated().sum()} | URL test yang juga ada di train: {test.URL.isin(train.URL).mean():.2%}")
pd.concat([train[train[TARGET] == k].sample(5, random_state=1)[["URL", TARGET]] for k in [1, 0]], ignore_index=True)
"""
)

md("**Temuan.** " + T("data"))

# ════════════════════════════════════════════════════════════════════ EDA
md("## 3. Exploratory Data Analysis\n\n### 3.1 Dua tipe URL mendominasi dataset")

code(
    """
TYPE_TR, TYPE_TE = url_type(train.URL), url_type(test.URL)
ORDER = ["host IP", "beranda https://www", "lainnya"]
tt = pd.DataFrame({"porsi train": TYPE_TR.value_counts(normalize=True), "porsi test": TYPE_TE.value_counts(normalize=True),
                   "proporsi legit": train.groupby(TYPE_TR)[TARGET].mean(), "jumlah": TYPE_TR.value_counts()}).loc[ORDER]
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].barh(ORDER[::-1], tt["porsi train"][::-1], color=C1)
pct_axis(ax[0], 0, axis="x"); ax[0].set(xlabel="porsi URL di train", title="Tipe URL")
ax[1].barh(ORDER[::-1], tt["proporsi legit"][::-1], color=[C3, C1, C2])
pct_axis(ax[1], 0, axis="x"); ax[1].set(xlabel="proporsi legitimate", title="Proporsi legitimate per tipe URL")
for i, v in enumerate(tt["proporsi legit"][::-1]):
    ax[1].text(v + 0.01, i, f"{v:.1%}", va="center")
save_fig(FIG_DIR, "01_url_types")
rule_auc = roc_auc_score(y, (TYPE_TR == "beranda https://www").astype(int))
print(f"AUC aturan 'beranda https://www = legitimate': {rule_auc:.4f}")
tt.style.format({"porsi train": "{:.1%}", "porsi test": "{:.1%}", "proporsi legit": "{:.1%}", "jumlah": "{:,.0f}"})
"""
)

md("**Temuan.** " + T("types"))

md("### 3.2 Fitur bawaan per kelas")

code(
    """
show = ["url_length", "url_entropy", "path_length", "percentage_numeric_chars", "token_count", "domain_name_length"]
fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))
for a, c in zip(axes.flat, show):
    lo, hi = train[c].quantile([0.005, 0.995])
    for k, col, lab in [(1, C1, "legitimate"), (0, C2, "phishing")]:
        a.hist(train.loc[train[TARGET] == k, c].clip(lo, hi), bins=50, alpha=0.55, density=True, color=col, label=lab)
    a.set(title=c, ylabel="densitas")
axes[0, 0].legend()
save_fig(FIG_DIR, "02_provided_features")
single = pd.Series({c: max(roc_auc_score(y, train[c]), 1 - roc_auc_score(y, train[c])) for c in PROVIDED}).sort_values(ascending=False)
single.head(8).to_frame("AUC satu fitur").style.format("{:.3f}")
"""
)

md("**Temuan.** " + T("features"))

md(
    """
### 3.3 Apakah fitur bawaan sesuai dengan URL-nya?

Sebagian besar fitur bisa dihitung ulang langsung dari string URL. Jika fitur bawaan tidak sama dengan hasil hitung ulang,
berarti nilainya diubah oleh penyusun data.
"""
)

code(
    """
CLEAN_TR, CLEAN_TE = recompute(train.URL), recompute(test.URL)
MIS_TR, MIS_TE = mismatch(train, CLEAN_TR), mismatch(test, CLEAN_TE)
print("Cocok persis dengan hasil hitung ulang: url_length", np.mean(train.url_length == CLEAN_TR.u_len).round(3),
      "| number_of_digits", np.mean(train.number_of_digits == CLEAN_TR.u_digits).round(3),
      "| dot_count", np.mean(train.dot_count == train.URL.str.count(r"\\.")).round(3))
rate = MIS_TR.groupby([TYPE_TR, train[TARGET].map({1: "legit", 0: "phishing"})]).mean().T
fig, ax = plt.subplots(1, 2, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
sns.heatmap(rate[[c for c in rate.columns if c[0] in ORDER]], annot=True, fmt=".0%", cmap="Blues", cbar=False, ax=ax[0])
ax[0].set(xlabel="tipe URL, label", ylabel="", title="Porsi nilai yang berbeda dari hasil hitung ulang")
m = MIS_TR.beda_entropy == 1
ax[1].scatter(CLEAN_TR.u_entropy[~m], train.url_entropy[~m], s=2, alpha=0.2, color=C1, label="sama")
ax[1].scatter(CLEAN_TR.u_entropy[m], train.url_entropy[m], s=2, alpha=0.2, color=C2, label="berbeda")
ax[1].set(xlabel="entropi dihitung dari URL", ylabel="url_entropy bawaan", title="Entropi: bawaan vs dihitung ulang"); ax[1].legend(markerscale=6)
save_fig(FIG_DIR, "03_feature_noise")
print("Porsi baris dengan minimal satu nilai berbeda: train", (MIS_TR.sum(1) > 0).mean().round(3), "| test", (MIS_TE.sum(1) > 0).mean().round(3))
"""
)

md("**Temuan.** " + T("noise"))

md("### 3.4 Label yang berlawanan dengan tipe URL-nya")

code(
    """
odd_ip = (TYPE_TR == "host IP") & (train[TARGET] == 1)
odd_home = (TYPE_TR == "beranda https://www") & (train[TARGET] == 0)
print(f"Host IP berlabel legitimate: {odd_ip.sum():,} ({odd_ip.sum() / (TYPE_TR == 'host IP').sum():.1%} dari host IP)")
print(f"Beranda https://www berlabel phishing: {odd_home.sum():,} ({odd_home.sum() / (TYPE_TR == 'beranda https://www').sum():.1%} dari beranda)")
dups = train[train.URL.duplicated(keep=False)].groupby("URL")[TARGET].nunique()
print(f"URL duplikat dengan label berbeda: {(dups > 1).sum()} dari {len(dups)}")
# Di dalam host IP: seberapa bisa label dibedakan oleh entropi bawaan vs entropi yang dihitung ulang?
ipm = (TYPE_TR == "host IP").values
cv = StratifiedKFold(5, shuffle=True, random_state=0)
ent_auc = {}
for name, col in [("url_entropy bawaan", train.url_entropy), ("entropi dihitung dari URL", CLEAN_TR.u_entropy)]:
    p = cross_val_predict(lgb_model(0, n_estimators=200), col[ipm].to_frame(), y[ipm], cv=cv, method="predict_proba")[:, 1]
    ent_auc[name] = roc_auc_score(y[ipm], p)
print("AUC di dalam host IP, satu fitur:", {k: round(v, 3) for k, v in ent_auc.items()})
pd.DataFrame({"host IP berlabel legitimate": train.URL[odd_ip].sample(6, random_state=0).values,
              "beranda berlabel phishing": train.URL[odd_home].sample(6, random_state=0).values})
"""
)

md("**Temuan.** " + T("labels"))

md("### 3.5 Apakah test berasal dari distribusi yang sama?")

code(
    """
adv_X = pd.concat([train[PROVIDED], test[PROVIDED]], ignore_index=True)
adv_y = np.r_[np.zeros(len(train)), np.ones(len(test))]
adv_p = cross_val_predict(lgb_model(0, n_estimators=200), adv_X, adv_y, cv=cv, method="predict_proba")[:, 1]
print(f"Adversarial validation AUC train vs test: {roc_auc_score(adv_y, adv_p):.3f}")
"""
)

md("**Temuan.** " + T("adversarial"))

# ════════════════════════════════════════════════════════════════════ Preprocessing
md(
    """
## 4. Preprocessing

Keputusan yang diambil dari EDA:

- **Fitur bawaan dipakai apa adanya**, karena kompetisi menilai prediksi pada data yang disusun dengan cara yang sama.
- **Fitur bersih dihitung ulang dari URL** (20 fitur: panjang host, kedalaman path, port, jumlah huruf besar, ekstensi
  berisiko, entropi asli, dan lain-lain). Model yang hanya memakai fitur bersih menunjukkan seberapa jauh kita bisa tanpa
  bergantung pada nilai yang diubah penyusun data.
- **Penanda perbedaan** (4 kolom) disiapkan untuk diuji, bukan langsung dipakai.
- **String URL** dipakai langsung oleh model teks TF-IDF karakter 2–5.
"""
)

code(
    """
X_prov, X_clean = train[PROVIDED], CLEAN_TR
X_both = pd.concat([train[PROVIDED], CLEAN_TR], axis=1)
X_all = pd.concat([X_both, MIS_TR], axis=1)
print({k: v.shape[1] for k, v in {"bawaan": X_prov, "bersih": X_clean, "bawaan + bersih": X_both, "+ penanda": X_all}.items()})
RESULTS, OOF = {}, {}

def record(name, oof):
    s = auc_by_seed(oof, y)
    OOF[name] = np.atleast_2d(oof).mean(0)
    RESULTS[name] = {"AUC OOF": np.mean(s), "std antar seed": np.std(s)}
    return f"{name}: AUC {np.mean(s):.5f} ± {np.std(s):.5f}"
"""
)

# ════════════════════════════════════════════════════════════════════ Baseline
md(
    """
## 5. Baseline

Semua skor adalah AUC out-of-fold dari Stratified 5-Fold yang diulang dengan 3 seed. Baseline: aturan satu baris dari
bagian 3.1, dan regresi logistik pada 16 fitur bawaan.
"""
)

code(
    """
print(record("aturan beranda https://www", np.tile((TYPE_TR == "beranda https://www").astype(float).values, (len(SEEDS), 1))))
print(record("regresi logistik, fitur bawaan", cv_predict(logreg_model, X_prov, y, SEEDS)))
"""
)

md("**Temuan.** " + T("baseline"))

# ════════════════════════════════════════════════════════════════════ Models
md("## 6. Model\n\nLightGBM pada empat set fitur, model teks TF-IDF pada string URL, dan rata-rata peringkat keduanya.")

code(
    """
t = time.time()
for name, X in [("LightGBM, fitur bersih dari URL", X_clean), ("LightGBM, fitur bawaan", X_prov),
                ("LightGBM, bawaan + bersih", X_both), ("LightGBM, bawaan + bersih + penanda", X_all)]:
    print(record(name, cv_predict(lgb_model, X, y, SEEDS)), f"({(time.time() - t) / 60:.1f} menit)")
print(record("TF-IDF karakter + regresi logistik", cv_predict(tfidf_model, train.URL, y, SEEDS)), f"({(time.time() - t) / 60:.1f} menit)")
OOF["rata-rata peringkat: LightGBM bawaan + bersih & TF-IDF"] = rank_mean(OOF["LightGBM, bawaan + bersih"], OOF["TF-IDF karakter + regresi logistik"])
RESULTS["rata-rata peringkat: LightGBM bawaan + bersih & TF-IDF"] = {"AUC OOF": roc_auc_score(y, OOF["rata-rata peringkat: LightGBM bawaan + bersih & TF-IDF"]), "std antar seed": np.nan}
"""
)

code(
    """
res = pd.DataFrame(RESULTS).T.sort_values("AUC OOF")
fig, ax = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
cols = [C2 if "bawaan" in n and "LightGBM" in n else (C3 if "bersih dari URL" in n or "TF-IDF" in n else C1) for n in res.index]
ax[0].barh(res.index, res["AUC OOF"], xerr=res["std antar seed"].fillna(0), color=cols)
ax[0].set_xlim(res["AUC OOF"].min() - 0.005, 1.0005); ax[0].set(xlabel="AUC OOF", title="Perbandingan model")
for n in ["LightGBM, fitur bersih dari URL", "LightGBM, fitur bawaan", "LightGBM, bawaan + bersih", "TF-IDF karakter + regresi logistik"]:
    fpr, tpr, _ = roc_curve(y, OOF[n])
    ax[1].plot(fpr, tpr, label=f"{n} ({RESULTS[n]['AUC OOF']:.4f})")
ax[1].set(xscale="log", xlim=(1e-3, 1), xlabel="false positive rate (log)", ylabel="true positive rate", title="Kurva ROC (OOF)")
ax[1].legend(fontsize=8, loc="lower right")
save_fig(FIG_DIR, "04_model_comparison")
res.style.format({"AUC OOF": "{:.5f}", "std antar seed": "{:.5f}"})
"""
)

md("**Temuan.** " + T("models"))

md(
    """
### 6.1 Model akhir

Model dipilih dengan aturan parsimoni: model paling sederhana yang AUC-nya tidak lebih buruk dari model terbaik lebih dari
dua standar deviasi antar seed. Urutan kesederhanaan: LightGBM fitur bawaan, bawaan + bersih, + penanda, lalu
rata-rata peringkat dengan TF-IDF.
"""
)

code(
    """
best = res["AUC OOF"].idxmax()
tol = 2 * np.nanmax([res.loc[best, "std antar seed"], 1e-5])
candidates = ["LightGBM, fitur bawaan", "LightGBM, bawaan + bersih", "LightGBM, bawaan + bersih + penanda",
              "rata-rata peringkat: LightGBM bawaan + bersih & TF-IDF"]
FINAL_NAME = next(n for n in candidates if res.loc[n, "AUC OOF"] >= res.loc[best, "AUC OOF"] - tol)
print(f"Terbaik: {best} ({res.loc[best, 'AUC OOF']:.5f}) | Model akhir: {FINAL_NAME} ({res.loc[FINAL_NAME, 'AUC OOF']:.5f})")
"""
)

# ════════════════════════════════════════════════════════════════════ Interpretation
md("## 7. Interpretasi dan analisis error\n\n### 7.1 Fitur apa yang dipakai model?")

code(
    """
imp_model = lgb_model(0).fit(X_both, y)
imp = pd.Series(imp_model.booster_.feature_importance("gain"), index=X_both.columns)
imp = (imp / imp.sum()).sort_values().tail(15)
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.barh(imp.index, imp.values, color=[C2 if c in PROVIDED else C3 for c in imp.index])
pct_axis(ax, 0, axis="x"); ax.set(xlabel="porsi total gain", title="15 fitur terpenting (oranye = bawaan, hijau = dihitung dari URL)")
save_fig(FIG_DIR, "05_importance")
print(f"Porsi gain fitur bawaan: {imp_model.booster_.feature_importance('gain')[:len(PROVIDED)].sum() / imp_model.booster_.feature_importance('gain').sum():.0%}")
"""
)

md("**Temuan.** " + T("importance"))

md("### 7.2 Di mana model salah?")

code(
    """
p_fin = OOF[FINAL_NAME]
err = pd.DataFrame({"URL": train.URL, "tipe": TYPE_TR, "label": y, "skor": p_fin})
err["salah"] = (err.skor > 0.5).astype(int) != err.label
by = err.groupby(["tipe", "label"]).agg(baris=("skor", "size"), salah=("salah", "mean"), skor_median=("skor", "median"))
by["porsi seluruh kesalahan"] = err.groupby(["tipe", "label"]).salah.sum() / err.salah.sum()
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), sharey=True)
for a, tp in zip(axes, ORDER):
    for k, col, lab in [(1, C1, "legitimate"), (0, C2, "phishing")]:
        s = err.skor[(err.tipe == tp) & (err.label == k)]
        a.hist(s, bins=np.linspace(0, 1, 41), color=col, alpha=0.6, label=f"{lab} (n={len(s):,})", log=True)
    a.set(title=tp, xlabel="skor model (peluang legitimate)"); a.legend(fontsize=8)
axes[0].set_ylabel("jumlah URL (log)")
save_fig(FIG_DIR, "06_errors")
display(by.style.format({"baris": "{:,.0f}", "salah": "{:.1%}", "skor_median": "{:.3f}", "porsi seluruh kesalahan": "{:.0%}"}))
worst = err.assign(loss=np.abs(err.label - err.skor)).sort_values("loss", ascending=False)
worst.head(15)[["URL", "tipe", "label", "skor"]]
"""
)

md("**Temuan.** " + T("errors"))

# ════════════════════════════════════════════════════════════════════ Submission
md("## 8. Submission\n\nModel akhir dilatih ulang pada seluruh data latih. Untuk LightGBM, prediksi dirata-rata dari 3 seed.")

code(
    """
X_te = {"LightGBM, fitur bawaan": test[PROVIDED], "LightGBM, bawaan + bersih": pd.concat([test[PROVIDED], CLEAN_TE], axis=1),
        "LightGBM, bawaan + bersih + penanda": pd.concat([test[PROVIDED], CLEAN_TE, MIS_TE], axis=1)}
X_trn = {"LightGBM, fitur bawaan": X_prov, "LightGBM, bawaan + bersih": X_both, "LightGBM, bawaan + bersih + penanda": X_all}
key = FINAL_NAME if FINAL_NAME in X_te else "LightGBM, bawaan + bersih"
p_lgb = np.mean([lgb_model(s).fit(X_trn[key], y).predict_proba(X_te[key])[:, 1] for s in SEEDS], axis=0)
if FINAL_NAME.startswith("rata-rata"):
    p_txt = tfidf_model(0).fit(train.URL, y).predict_proba(test.URL)[:, 1]
    p_test = rank_mean(p_lgb, p_txt)
else:
    p_test = p_lgb
target_col = sample.columns[1]
sub = sample[[ID]].merge(pd.DataFrame({ID: test[ID], target_col: p_test}), on=ID, how="left")
assert sub[target_col].notna().all() and len(sub) == len(sample)
sub.to_csv(OUT_DIR / "submission.csv", index=False)
res.to_csv(OUT_DIR / "cv_results.csv"); by.to_csv(OUT_DIR / "errors_by_type.csv")
print(f"Model: {FINAL_NAME} | proporsi skor > 0,5 di test: {(p_test > 0.5).mean():.1%} (train legit {y.mean():.1%})")
"""
)

md("**Temuan.** " + T("submission"))

md("## 9. Kesimpulan\n\n" + T("conclusion") + "\n\n### Jejak lingkungan")

code(
    """
import platform
pkgs = ["numpy", "pandas", "scikit-learn", "lightgbm", "matplotlib", "seaborn"]
print(" | ".join([f"python {platform.python_version()}"] + [f"{p} {version(p)}" for p in pkgs]))
print(f"Waktu eksekusi total: {(time.time() - T0) / 60:.1f} menit")
"""
)

nb["cells"] = cells
nb_path = ROOT / "kaggle-notebook" / NB_NAME
nb_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, nb_path)
print("ditulis:", nb_path)

src_files = {str(p.relative_to(ROOT)): p.read_text() for p in sorted((ROOT / "src").rglob("*.py"))}
runner = f'''"""Runner Kaggle untuk {NB_NAME} (di-generate oleh _build_notebook.py; jangan diedit manual)."""
import json, shutil, subprocess, sys
from pathlib import Path

SRC_FILES = {src_files!r}
NOTEBOOK = json.loads({nbf.writes(nb)!r})

proj = Path("/kaggle/working/project")
for rel, text in SRC_FILES.items():
    p = proj / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
(proj / "notebooks").mkdir(parents=True, exist_ok=True)
nb_in = proj / "notebooks" / "{NB_NAME}"
nb_in.write_text(json.dumps(NOTEBOOK))
cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", str(nb_in),
       "--output", "{NB_NAME}", "--output-dir", "/kaggle/working", "--ExecutePreprocessor.timeout=-1"]
ret = subprocess.run(cmd, cwd=proj / "notebooks")
shutil.rmtree(proj, ignore_errors=True)
sys.exit(ret.returncode)
'''
(ROOT / "kaggle").mkdir(exist_ok=True)
(ROOT / "kaggle" / "run_on_kaggle.py").write_text(runner)
print("ditulis:", ROOT / "kaggle" / "run_on_kaggle.py")
