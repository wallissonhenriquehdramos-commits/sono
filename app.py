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
st.title("💤 IA Interpretável para Estadiamento do Sono — com Fusos, Complexos K, PDF e Agrupamento N1+N2")

CLASSES = ["W","N1","N2","N3","REM"]

def bandpower(data, sf, fmin, fmax):
    freqs, psd = welch(data, sf, nperseg=int(sf*4), noverlap=int(sf*2))
    idx = (freqs >= fmin) & (freqs < fmax)
    return float(np.trapz(psd[idx], freqs[idx])) if np.any(idx) else 0.0

def detect_spindles_and_k(raw, sf):
    # ⚠️ Aqui está simplificado: na prática você usaria detecção real de fusos/K.
    n_epochs = int(raw.times[-1]//30)
    spindles = np.random.rand(n_epochs) # placeholder para fusos
    ks = np.random.rand(n_epochs)       # placeholder para complexos K
    return spindles, ks

def extract_features(raw, epoch_len=30.0, resample_hz=100):
    raw.filter(0.5,40., fir_design="firwin", verbose=False)
    raw.resample(resample_hz, npad="auto", verbose=False)
    sf = raw.info["sfreq"]
    labels = np.array(["W","N1","N2","N3","REM"]*int(raw.times[-1]//(5*epoch_len)))[:int(raw.times[-1]//epoch_len)] 
    spindles, ks = detect_spindles_and_k(raw,sf)
    feats = np.vstack([spindles, ks]).T
    return feats, labels, ["spindles","Kcomplex"]

def train_and_eval(X,y,classes):
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)
    scaler=StandardScaler(); Xtr_s=scaler.fit_transform(Xtr); Xte_s=scaler.transform(Xte)
    rf=RandomForestClassifier(n_estimators=100,random_state=42)
    rf.fit(Xtr_s,ytr); pred=rf.predict(Xte_s)
    rep=classification_report(yte,pred,labels=classes,output_dict=True)
    cm=confusion_matrix(yte,pred,labels=classes)
    return rf,rep,cm

def export_pdf(rep, cm, classes, outpath="report.pdf"):
    doc=SimpleDocTemplate(outpath)
    styles=getSampleStyleSheet()
    elems=[]
    elems.append(Paragraph("Relatório Sono XAI", styles["Title"]))
    elems.append(Spacer(1,12))
    elems.append(Paragraph(str(rep), styles["Normal"]))
    elems.append(Spacer(1,12))
    plt.imshow(cm); plt.title("Confusion Matrix"); plt.savefig("cm.png"); plt.close()
    elems.append(RLImage("cm.png",width=400,height=300))
    doc.build(elems)
    with open(outpath,"rb") as f: return f.read()

with st.sidebar:
    st.header("Configurações")
    group=st.checkbox("Agrupar N1+N2 como Sono Leve")
    start=st.button("Treinar IA (demo)")

if start:
    class FakeRaw:
        def __init__(self): self.times=np.linspace(0,1800,1800*100)
        def filter(self,*a,**k): return self
        def resample(self,*a,**k): return self
        @property
        def info(self): return {"sfreq":100}
    raw=FakeRaw()
    X,y,fnames=extract_features(raw)
    if group:
        y=np.where(np.isin(y,["N1","N2"]),"SonoLeve",y)
        classes=["W","SonoLeve","N3","REM"]
    else:
        classes=CLASSES
    rf,rep,cm=train_and_eval(X,y,classes)
    st.subheader("Relatório")
    st.write(rep)
    st.subheader("Matriz de Confusão")
    st.write(cm)
    pdf_bytes=export_pdf(rep,cm,classes)
    st.download_button("⬇️ Baixar Relatório PDF", data=pdf_bytes, file_name="report.pdf")
