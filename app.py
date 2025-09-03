import os, io, joblib, tempfile, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import mne
from mne.io import read_raw_edf
from scipy.signal import welch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# ---------- CONFIG BÁSICA ----------
st.set_page_config(page_title="Sono XAI", layout="wide")
st.title("💤 IA Interpretável para Estadiamento do Sono — Sleep-EDF / Upload / PDF / LOSO")

CLASSES5 = ["W","N1","N2","N3","REM"]
CLASSES4 = ["W","SonoLeve","N3","REM"]
BANDS = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,12),"sigma":(12,16),"beta":(16,30)}

STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": None,
    "Movement time": None,
}

# ---------- FUNÇÕES DE SINAL ----------
def bandpower(x, sf, fmin, fmax):
    freqs, psd = welch(x, sf, nperseg=int(sf*4), noverlap=int(sf*2))
    idx = (freqs>=fmin)&(freqs<fmax)
    return float(np.trapz(psd[idx], freqs[idx])) if np.any(idx) else 0.0

def pick_channels(chs):
    eeg = [c for c in ["EEG Fpz-Cz","Fpz-Cz","EEG Pz-Oz","Pz-Oz"] if c in chs][:2]
    eog = [c for c in ["EOG horizontal","EOG","EOG L","EOG R"] if c in chs][:1]
    emg = [c for c in ["EMG submental","EMG"] if c in chs][:1]
    return eeg, eog, emg

def epoch_labels_from_annotations(raw, epoch_len=30.0):
    ann = raw.annotations
    tmax = raw.times[-1] if raw.times.size else 0.0
    n_epochs = int(tmax // epoch_len)
    labels = []
    for i in range(n_epochs):
        t0 = i*epoch_len
        desc=None
        for onset, dur, des in zip(ann.onset, ann.duration, ann.description):
            if (t0>=onset) and (t0<onset+dur):
                desc=des; break
        labels.append(STAGE_MAP.get(desc, None))
    return np.array(labels, dtype=object)

def detect_spindles_K_simple(raw, epoch_len=30.0):
    """Detecção heurística para IC: densidade de fusos (12–16 Hz) e K (0.7–2 Hz) por época."""
    sf = raw.info["sfreq"]
    eeg_chs, _, _ = pick_channels(raw.ch_names)
    if not eeg_chs:
        return None, None
    ch = eeg_chs[0]
    x = raw.get_data(picks=[ch])[0]

    from scipy.signal import butter, filtfilt
    def bp(sig, lo, hi): 
        b,a = butter(4, [lo/(sf/2), hi/(sf/2)], btype="band")
        return filtfilt(b,a,sig)

    x_sigma = bp(x, 12, 16)
    thr_sigma = np.percentile(np.abs(x_sigma), 95)
    spindle_mask = np.abs(x_sigma) > thr_sigma

    x_k = bp(x, 0.7, 2.0)
    thr_k = np.percentile(np.abs(x_k), 95)
    k_mask = np.abs(x_k) > thr_k

    n_epochs = int(raw.times[-1] // epoch_len)
    sp_density, k_density = [], []
    for i in range(n_epochs):
        t0 = int(i*epoch_len*sf); t1 = int((i+1)*epoch_len*sf)
        sp_density.append(float(spindle_mask[t0:t1].mean()))
        k_density.append(float(k_mask[t0:t1].mean()))
    return np.array(sp_density), np.array(k_density)

def extract_features_from_raw(raw, epoch_len=30.0, resample_hz=100):
    raw.filter(0.5, 40., fir_design="firwin", verbose=False)
    raw.resample(resample_hz, npad="auto", verbose=False)

    labels = epoch_labels_from_annotations(raw, epoch_len=epoch_len)
    eeg_chs, eog_chs, emg_chs = pick_channels(raw.ch_names)
    sf = raw.info["sfreq"]

    feat_names = []
    for ch in eeg_chs:
        for b in BANDS:
            feat_names.append(f"{ch}__bp_{b}")
    if eog_chs: feat_names.append(f"{eog_chs[0]}__var")
    if emg_chs: feat_names.append(f"{emg_chs[0]}__rms")
    feat_names += ["spindle_density","kcomplex_density"]

    X_rows = []
    sp_dens, k_dens = detect_spindles_K_simple(raw, epoch_len=epoch_len)
    n_epochs = len(labels)

    for i in range(n_epochs):
        t0 = i*epoch_len; t1 = t0+epoch_len
        try:
            seg = raw.copy().crop(t0, t1).get_data()
        except Exception:
            break
        row=[]
        for ch in eeg_chs:
            idx = raw.ch_names.index(ch); x = seg[idx]
            for (lo,hi) in BANDS.values():
                row.append(bandpower(x, sf, lo, hi))
        if eog_chs:
            idx = raw.ch_names.index(eog_chs[0]); row.append(float(np.var(seg[idx])))
        if emg_chs:
            idx = raw.ch_names.index(emg_chs[0]); x = seg[idx]; row.append(float(np.sqrt(np.mean(x**2))))
        row.append(float(sp_dens[i]) if sp_dens is not None else 0.0)
        row.append(float(k_dens[i]) if k_dens is not None else 0.0)
        X_rows.append(row)

    X = np.array(X_rows, dtype=float)
    y = labels[:len(X_rows)]
    mask = y != None
    return X[mask], y[mask], feat_names

# ---------- IA ----------
def group_labels(y, group=False):
    if not group:
        return y, CLASSES5
    y2 = np.where(np.isin(y,["N1","N2"]), "SonoLeve", y)
    return y2, CLASSES4

def compute_sleep_stats(y_seq, epoch_len=30.0, classes=None):
    """Estatísticas simples de arquitetura de sono para o relatório."""
    if classes is None: classes = sorted(pd.unique(y_seq))
    total_epochs = len(y_seq)
    tib_sec = total_epochs * epoch_len
    # TST = tudo que não é W
    sleep_mask = y_seq != "W"
    tst_sec = int(np.sum(sleep_mask) * epoch_len)
    eff = tst_sec / tib_sec if tib_sec > 0 else 0.0
    # latência até primeiro sono (primeiro != W)
    try:
        lat_idx = np.where(y_seq != "W")[0][0]
        latency_sec = int(lat_idx * epoch_len)
    except Exception:
        latency_sec = None

    perc = {}
    for c in classes:
        perc[c] = float(np.mean(y_seq==c))*100.0

    return {
        "epochs_total": int(total_epochs),
        "tib_min": tib_sec/60.0,
        "tst_min": tst_sec/60.0,
        "efficiency": eff*100.0,
        "latency_min": (latency_sec/60.0) if latency_sec is not None else None,
        "percent_by_stage": perc
    }

def plot_hypnogram(y_true, y_pred, classes, fname="hypnogram.png", title="Hipnograma (Real x Previsto)"):
    # mapa para níveis
    mapping = {c:i for i,c in enumerate(classes)}
    t = np.arange(len(y_true))
    plt.figure(figsize=(10,3))
    plt.plot(t, [mapping[c] for c in y_true], drawstyle="steps-mid", label="Real")
    plt.plot(t, [mapping[c] for c in y_pred], drawstyle="steps-mid", alpha=0.6, label="Previsto")
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Épocas (30s cada)"); plt.title(title); plt.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()
    return fname

def export_pdf(rep_dict, cm, classes, imp_fig_path, hyp_fig_path, stats_dict, outpath="report.pdf"):
    doc = SimpleDocTemplate(outpath)
    styles = getSampleStyleSheet()
    elems = [Paragraph("Relatório Sono XAI", styles["Title"]), Spacer(1,12)]

    # Estatísticas clínicas
    s = stats_dict
    lines_stats = [
        f"Épocas totais: {s['epochs_total']}",
        f"Tempo na cama (TIB): {s['tib_min']:.1f} min",
        f"Tempo total de sono (TST): {s['tst_min']:.1f} min",
        f"Eficiência do sono: {s['efficiency']:.1f}%",
        f"Latência do sono: {s['latency_min']:.1f} min" if s['latency_min'] is not None else "Latência do sono: n/d",
        "Distribuição por estágio: " + ", ".join([f"{k}={v:.1f}%" for k,v in s["percent_by_stage"].items()])
    ]
    elems.append(Paragraph("<br/>".join(lines_stats), styles["Normal"]))
    elems.append(Spacer(1,12))

    # Tabela de métricas em texto
    lines = []
    for k,v in rep_dict.items():
        if isinstance(v, dict):
            lines.append(f"[{k}] precision={v.get('precision',0):.3f}  recall={v.get('recall',0):.3f}  f1={v.get('f1-score',0):.3f}  support={v.get('support',0)}")
    elems.append(Paragraph("<br/>".join(lines), styles["Code"]))
    elems.append(Spacer(1,12))

    # Matriz de confusão
    plt.figure(figsize=(5,4))
    plt.imshow(cm); plt.title("Confusion Matrix"); plt.xticks(range(len(classes)),classes); plt.yticks(range(len(classes)),classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j,i,cm[i,j],ha="center",va="center",color="red",fontsize=8)
    plt.tight_layout(); plt.savefig("cm.png", dpi=150); plt.close()
    elems.append(RLImage("cm.png", width=380, height=300))
    elems.append(Spacer(1,12))

    # Hipnograma
    elems.append(RLImage(hyp_fig_path, width=500, height=160))
    elems.append(Spacer(1,12))

    # Importâncias
    elems.append(RLImage(imp_fig_path, width=500, height=360))

    doc.build(elems)
    with open(outpath, "rb") as f:
        return f.read()

def train_holdout(X, y, classes):
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(Xtr_s, ytr); pred = rf.predict(Xte_s)
    rep = classification_report(yte, pred, labels=classes, zero_division=0, output_dict=True)
    cm  = confusion_matrix(yte, pred, labels=classes)
    importances = rf.feature_importances_
    return rf, scaler, rep, cm, importances, (yte, pred)

def train_loso(subj_data, classes):
    """subj_data = list of dicts: {'X':..., 'y':..., 'feat_names':...}"""
    all_reports = []
    cms = []
    importances_all = []
    folds_info = []

    for i in range(len(subj_data)):
        X_te = subj_data[i]['X']; y_te = subj_data[i]['y']
        X_tr = np.vstack([d['X'] for j,d in enumerate(subj_data) if j!=i])
        y_tr = np.concatenate([d['y'] for j,d in enumerate(subj_data) if j!=i])

        scaler = StandardScaler(); Xtr_s = scaler.fit_transform(X_tr); Xte_s = scaler.transform(X_te)
        rf = RandomForestClassifier(n_estimators=300, random_state=42)
        rf.fit(Xtr_s, y_tr); pred = rf.predict(Xte_s)

        rep = classification_report(y_te, pred, labels=classes, zero_division=0, output_dict=True)
        cm  = confusion_matrix(y_te, pred, labels=classes)

        all_reports.append(rep); cms.append(cm)
        importances_all.append(rf.feature_importances_)
        folds_info.append({"y_true": y_te, "y_pred": pred})

    # médias
    mean_cm = np.mean(np.stack(cms), axis=0).round(1)
    mean_imp = np.mean(np.stack(importances_all), axis=0)
    # média de classification_report
    keys = list(all_reports[0].keys())
    mean_report = {}
    for k in keys:
        if isinstance(all_reports[0][k], dict):
            mean_report[k] = {}
            for m in all_reports[0][k].keys():
                mean_report[k][m] = float(np.mean([r[k][m] for r in all_reports]))
        else:
            mean_report[k] = float(np.mean([r[k] for r in all_reports]))

    # usa o último fold para hipnograma ilustrativo
    last_fold = folds_info[-1]
    return mean_report, mean_cm, mean_imp, last_fold

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Configurações")
    mode = st.radio("Modo de dados:", ["Sleep-EDF (baixar)", "Arquivos locais (upload)"])
    group = st.checkbox("Agrupar N1+N2 como Sono Leve", value=True)
    epoch_len = st.number_input("Duração da época (s)", 10.0, 60.0, 30.0, 5.0)
    resample_hz = st.number_input("Reamostragem (Hz)", 50.0, 200.0, 100.0, 10.0)
    val_mode = st.radio("Modo de validação:", ["Holdout (75/25)", "LOSO (por sujeito)"])

    if mode == "Sleep-EDF (baixar)":
        subjects_text = st.text_input("IDs de sujeitos (ex.: 0 1)", "0")
        recording = st.number_input("Recording (geralmente 1)", 1, 2, 1, 1)
        start_btn = st.button("⬇️ Baixar & Treinar")
    else:
        psg_files = st.file_uploader("Envie PSG (EDF) — 1 ou mais", type=["edf"], accept_multiple_files=True)
        hyp_files = st.file_uploader("Envie Hipnogramas (EDF/XML) — mesmo número", type=["edf","xml"], accept_multiple_files=True)
        start_btn = st.button("📊 Processar & Treinar")

status = st.empty()

# ---------- EXECUÇÃO ----------
if start_btn:
    try:
        subj_data = []  # lista por sujeito: dict com X,y,feat_names
        fnames = None

        if mode == "Sleep-EDF (baixar)":
            try:
                from mne.datasets.sleep_physionet.age import fetch_data
            except Exception:
                from mne.datasets.sleep_physionet import fetch_data
            subs = [int(s) for s in subjects_text.strip().split() if s.strip().isdigit()]
            status.info("Baixando Sleep-EDF (pode levar alguns minutos)…")
            paths = fetch_data(subjects=subs, recording=[int(recording)])
            for psg_path, hyp_path in paths:
                raw = read_raw_edf(psg_path, preload=True, verbose=False)
                ann = mne.read_annotations(hyp_path)
                raw.set_annotations(ann, emit_warning=False)
                X_, y_, fn = extract_features_from_raw(raw, epoch_len=epoch_len, resample_hz=resample_hz)
                y_, classes_used = group_labels(y_, group=group)
                if fnames is None: fnames = fn
                subj_data.append({"X":X_, "y":y_})
        else:
            if not psg_files or not hyp_files:
                st.error("Envie PSG(s) e hipnograma(s) com a mesma quantidade."); st.stop()
            if len(psg_files) != len(hyp_files):
                st.error("O número de PSGs e hipnogramas deve ser o mesmo."); st.stop()
            for psg_up, hyp_up in zip(psg_files, hyp_files):
                raw = read_raw_edf(psg_up, preload=True, verbose=False)
                ann = mne.read_annotations(hyp_up)
                raw.set_annotations(ann, emit_warning=False)
                X_, y_, fn = extract_features_from_raw(raw, epoch_len=epoch_len, resample_hz=resample_hz)
                y_, classes_used = group_labels(y_, group=group)
                if fnames is None: fnames = fn
                subj_data.append({"X":X_, "y":y_})

        # Se for holdout, junta tudo; se LOSO, usa fold por sujeito
        if val_mode.startswith("Holdout"):
            X = np.vstack([d['X'] for d in subj_data]); y = np.concatenate([d['y'] for d in subj_data])
            status.success(f"Dados prontos: {X.shape[0]} épocas, {X.shape[1]} features. Treinando (Holdout)…")
            rf, scaler, rep, cm, importances, (yte, pred) = train_holdout(X, y, classes_used)

            # Tabelas e gráficos na UI
            st.subheader("📈 Relatório de Classificação (Holdout)")
            st.dataframe(pd.DataFrame(rep).T.round(3))

            st.subheader("🧩 Matriz de Confusão")
            fig_cm, ax = plt.subplots()
            ax.imshow(cm); ax.set_xticks(range(len(classes_used))); ax.set_xticklabels(classes_used)
            ax.set_yticks(range(len(classes_used))); ax.set_yticklabels(classes_used)
            ax.set_title("Confusion Matrix")
            for i in range(len(classes_used)):
                for j in range(len(classes_used)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="red", fontsize=8)
            st.pyplot(fig_cm)

            st.subheader("🔍 Importância das Features")
            order = np.argsort(importances)[::-1]
            topk = min(25, len(importances))
            fig_imp, ax2 = plt.subplots(figsize=(8, 6))
            ax2.barh(range(topk), importances[order[:topk]][::-1])
            ax2.set_yticks(range(topk)); ax2.set_yticklabels([fnames[i] for i in order[:topk]][::-1], fontsize=8)
            ax2.set_xlabel("Gini importance"); ax2.set_title("Top features")
            st.pyplot(fig_imp)

            # Hipnograma + stats (usando a sequência do conjunto de teste)
            hyp_path = plot_hypnogram(yte, pred, classes_used, fname="hyp_holdout.png", title="Hipnograma (Holdout — teste)")
            stats = compute_sleep_stats(yte, epoch_len=epoch_len, classes=classes_used)

            # PDF
            fig_imp.savefig("feature_importance.png", dpi=150)
            pdf_bytes = export_pdf(rep, cm, classes_used, "feature_importance.png", hyp_path, stats, outpath="report.pdf")
            st.download_button("⬇️ Baixar Relatório PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")

            # Features CSV (todas, não apenas teste)
            df_feats = pd.DataFrame(X, columns=fnames); df_feats["stage"] = y
            st.download_button("Baixar features.csv", data=df_feats.to_csv(index=False).encode("utf-8"),
                               file_name="features.csv", mime="text/csv")

            # Salvar modelo (para reuso)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmpf:
                joblib.dump({"model":rf, "scaler":scaler, "classes":classes_used, "features":fnames}, tmpf.name)
                with open(tmpf.name, "rb") as f:
                    st.download_button("💾 Baixar modelo treinado (.joblib)", data=f.read(),
                                       file_name="sono_xai_model.joblib", mime="application/octet-stream")

            st.success("Concluído! ✅")

        else:
            # LOSO
            if len(subj_data) < 2:
                st.error("LOSO requer pelo menos 2 sujeitos."); st.stop()

            status.success(f"{len(subj_data)} sujeitos carregados. Executando LOSO…")
            mean_report, mean_cm, mean_imp, last_fold = train_loso(subj_data, classes_used)

            st.subheader("📈 Relatório de Classificação (média LOSO)")
            st.dataframe(pd.DataFrame(mean_report).T.round(3))

            st.subheader("🧩 Matriz de Confusão (média LOSO)")
            fig_cm, ax = plt.subplots()
            ax.imshow(mean_cm); ax.set_xticks(range(len(classes_used))); ax.set_xticklabels(classes_used)
            ax.set_yticks(range(len(classes_used))); ax.set_yticklabels(classes_used)
            ax.set_title("Confusion Matrix — média (validação por sujeito)")
            for i in range(len(classes_used)):
                for j in range(len(classes_used)):
                    ax.text(j, i, int(mean_cm[i, j]), ha="center", va="center", color="red", fontsize=8)
            st.pyplot(fig_cm)

            st.subheader("🔍 Importância das Features (média LOSO)")
            order = np.argsort(mean_imp)[::-1]
            topk = min(25, len(mean_imp))
            fig_imp, ax2 = plt.subplots(figsize=(8, 6))
            ax2.barh(range(topk), mean_imp[order[:topk]][::-1])
            ax2.set_yticks(range(topk)); ax2.set_yticklabels([fnames[i] for i in order[:topk]][::-1], fontsize=8)
            ax2.set_xlabel("Gini importance"); ax2.set_title("Top features (média LOSO)")
            st.pyplot(fig_imp)

            # Hipnograma para o último fold (sujeito de teste)
            hyp_path = plot_hypnogram(last_fold["y_true"], last_fold["y_pred"], classes_used,
                                      fname="hyp_loso.png", title="Hipnograma (sujeito de teste — LOSO)")
            stats = compute_sleep_stats(last_fold["y_true"], epoch_len=epoch_len, classes=classes_used)

            # PDF (médias LOSO + hipnograma do sujeito de teste)
            fig_imp.savefig("feature_importance.png", dpi=150)
            pdf_bytes = export_pdf(mean_report, mean_cm, classes_used, "feature_importance.png", hyp_path, stats, outpath="report.pdf")
            st.download_button("⬇️ Baixar Relatório PDF (LOSO)", data=pdf_bytes, file_name="report_loso.pdf", mime="application/pdf")

            # Exportar features concatenadas (útil p/ análises futuras)
            X_all = np.vstack([d['X'] for d in subj_data]); y_all = np.concatenate([d['y'] for d in subj_data])
            df_feats = pd.DataFrame(X_all, columns=fnames); df_feats["stage"] = y_all
            st.download_button("Baixar features.csv (todos os sujeitos)", data=df_feats.to_csv(index=False).encode("utf-8"),
                               file_name="features_all.csv", mime="text/csv")

            st.info("LOSO finalizado. Para reuso de modelo LOSO (treino em N-1), rode salvando manualmente a cada fold caso necessário.")
            st.success("Concluído! ✅")

    except Exception as e:
        st.error(f"Erro: {e}")


