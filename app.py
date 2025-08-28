import os, io
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

st.set_page_config(page_title="Sono XAI", layout="wide")
st.title("💤 IA Interpretável para Estadiamento do Sono — Upload / Sleep-EDF / PDF / Agrupamento N1+N2")

CLASSES5 = ["W","N1","N2","N3","REM"]
CLASSES4 = ["W","SonoLeve","N3","REM"]

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
BANDS = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,12),"sigma":(12,16),"beta":(16,30)}

def bandpower(x, sf, fmin, fmax):
    from scipy.signal import welch
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
    """Detecção simples (heurística) p/ IC: densidade de fusos e K-Complex por época."""
    sf = raw.info["sfreq"]
    eeg_chs, _, _ = pick_channels(raw.ch_names)
    if not eeg_chs:
        return None, None
    ch = eeg_chs[0]
    x = raw.get_data(picks=[ch])[0]

    # Filtrar faixas típicas
    from scipy.signal import butter, filtfilt
    def bp(sig, lo, hi): 
        b,a = butter(4, [lo/(sf/2), hi/(sf/2)], btype="band")
        return filtfilt(b,a,sig)

    # Fusos ~12–16 Hz
    x_sigma = bp(x, 12, 16)
    thr_sigma = np.percentile(np.abs(x_sigma), 95)
    spindle_mask = np.abs(x_sigma) > thr_sigma

    # K-complex ~ onda lenta negativa + rebote: aqui usamos banda 0.7–2 Hz como proxy
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
        # eventos
        if sp_dens is not None: row.append(float(sp_dens[i]))
        else: row.append(0.0)
        if k_dens is not None: row.append(float(k_dens[i]))
        else: row.append(0.0)
        X_rows.append(row)

    X = np.array(X_rows, dtype=float)
    y = labels[:len(X_rows)]
    mask = y != None
    return X[mask], y[mask], feat_names

def train_and_eval(X,y,group=False):
    if group:
        y = np.where(np.isin(y,["N1","N2"]), "SonoLeve", y)
        classes = CLASSES4
    else:
        classes = CLASSES5

    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(Xtr_s, ytr); pred = rf.predict(Xte_s)
    rep = classification_report(yte, pred, labels=classes, zero_division=0, output_dict=True)
    cm  = confusion_matrix(yte, pred, labels=classes)
    importances = rf.feature_importances_
    return rf, rep, cm, importances, classes

def export_pdf(rep_dict, cm, classes, imp_fig_path, outpath="report.pdf"):
    doc = SimpleDocTemplate(outpath)
    styles = getSampleStyleSheet()
    elems = [Paragraph("Relatório Sono XAI", styles["Title"]), Spacer(1,12)]

    # Tabela de métricas em texto simples
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

    # Importâncias (imagem já gerada fora)
    elems.append(RLImage(imp_fig_path, width=400, height=300))

    doc.build(elems)
    with open(outpath, "rb") as f:
        return f.read()

with st.sidebar:
    st.header("⚙️ Configurações")
    mode = st.radio("Modo de dados:", ["Sleep-EDF (baixar)", "Arquivos locais (upload)"])
    group = st.checkbox("Agrupar N1+N2 como Sono Leve")
    epoch_len = st.number_input("Duração da época (s)", 10.0, 60.0, 30.0, 5.0)
    resample_hz = st.number_input("Reamostragem (Hz)", 50.0, 200.0, 100.0, 10.0)

    if mode == "Sleep-EDF (baixar)":
        subjects_text = st.text_input("IDs de sujeitos (ex.: 0 1)", "0")
        recording = st.number_input("Recording (geralmente 1)", 1, 2, 1, 1)
        start_btn = st.button("⬇️ Baixar & Treinar")
    else:
        psg_files = st.file_uploader("Envie PSG (EDF) — 1 ou mais", type=["edf"], accept_multiple_files=True)
        hyp_files = st.file_uploader("Envie Hipnogramas (EDF/XML) — mesmo número", type=["edf","xml"], accept_multiple_files=True)
        start_btn = st.button("📊 Processar & Treinar")

status = st.empty()

if start_btn:
    try:
        all_X, all_y, fnames = [], [], None

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
                if fnames is None: fnames = fn
                all_X.append(X_); all_y.append(y_)

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
                if fnames is None: fnames = fn
                all_X.append(X_); all_y.append(y_)

        X = np.vstack(all_X); y = np.concatenate(all_y)
        status.success(f"Dados prontos: {X.shape[0]} épocas, {X.shape[1]} features. Treinando…")

        rf, rep, cm, importances, classes = train_and_eval(X, y, group=group)

        st.subheader("📈 Relatório de Classificação")
        df_rep = pd.DataFrame(rep).T.round(3)
        st.dataframe(df_rep)

        st.subheader("🧩 Matriz de Confusão")
        fig_cm, ax = plt.subplots()
        ax.imshow(cm); ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        ax.set_title("Confusion Matrix")
        for i in range(len(classes)):
            for j in range(len(classes)):
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

        # Salvar figura de importâncias para o PDF
        imp_fig_path = "feature_importance.png"
        fig_imp.savefig(imp_fig_path, dpi=150)

        pdf_bytes = export_pdf(rep, cm, classes, imp_fig_path, outpath="report.pdf")
        st.download_button("⬇️ Baixar Relatório PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")

        # Também oferecer features e CSV
        df_feats = pd.DataFrame(X, columns=fnames); df_feats["stage"] = y
        st.download_button("Baixar features.csv", data=df_feats.to_csv(index=False).encode("utf-8"),
                           file_name="features.csv", mime="text/csv")

        st.success("Concluído! ✅")

    except Exception as e:
        st.error(f"Erro: {e}")
