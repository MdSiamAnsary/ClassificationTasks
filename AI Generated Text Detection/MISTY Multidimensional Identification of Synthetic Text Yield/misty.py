# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

pip install torch torchvision sentence-transformers transformers textstat nltk scikit-learn tqdm datasets

import os
import argparse
import math
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

import textstat
import nltk
from nltk import word_tokenize, pos_tag, sent_tokenize

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DATA_CSV = "dataset.csv"

OUT_DIR = "out"
EPOCHS_PROJ = 4
EPOCHS_CLF = 4
LR = 1e-3
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

DISCOURSE_CONNECTIVES = {
    "however", "moreover", "furthermore", "therefore",
    "consequently", "thus", "despite", "although", "meanwhile", "instead"
}

def perplexity_gpt2(text, tok, model, device=DEVICE):
    enc = tok(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = enc.input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    loss = out.loss.item()
    return math.exp(loss) if loss < 700 else float('inf')

def readability_feats(text):
    try:
        return [
            textstat.flesch_reading_ease(text),
            textstat.automated_readability_index(text),
            textstat.dale_chall_readability_score(text)
        ]
    except Exception:
        return [0.0, 0.0, 0.0]

def discourse_feats(text):
    sents = sent_tokenize(text)
    lens = [len(word_tokenize(s)) for s in sents] or [0]
    return [float(np.var(lens)), float(np.mean(lens)), sum(text.lower().count(c) for c in DISCOURSE_CONNECTIVES)]

def pos_feats(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    total = max(1, len(tags))
    base = {"NN":0,"VB":0,"JJ":0,"RB":0}
    for _,t in tags:
        key = t[:2]
        if key in base: base[key]+=1
    return [base[k]/total for k in base]

def lexical_entropy(text):
    toks = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    if not toks: return 0.0
    vals = np.array(list({t:toks.count(t) for t in set(toks)}.values()))
    p = vals / len(toks)
    return float(-np.sum(p*np.log2(p+1e-12)))

def handcrafted(text, tok, model):
    feats = [perplexity_gpt2(text, tok, model)]
    feats += readability_feats(text)
    feats += discourse_feats(text)
    feats += pos_feats(text)
    feats.append(lexical_entropy(text))
    return np.array(feats, dtype=np.float32)

class TextData(Dataset):
    def __init__(self, df, embedder, tok, model):
        self.df = df.reset_index(drop=True)
        self.embedder, self.tok, self.model = embedder, tok, model
        self.cache = {}
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        txt, label = str(row["text"]), int(row["label"])
        if i not in self.cache:
            emb = self.embedder.encode([txt], show_progress_bar=False)[0]
            hf = handcrafted(txt, self.tok, self.model)
            self.cache[i] = (emb.astype(np.float32), hf)
        emb, hf = self.cache[i]
        return {"embedding": emb, "handcrafted": hf, "label": label}

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hid=256, out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,hid), nn.ReLU(), nn.Linear(hid,out))
    def forward(self,x): return self.net(x)

class SupConLoss(nn.Module):
    def __init__(self,temp=0.07): super().__init__(); self.t=temp
    def forward(self,z,y):
        z = nn.functional.normalize(z, dim=1)
        sim = torch.matmul(z,z.T)/self.t
        mask = torch.eq(y.unsqueeze(1), y.unsqueeze(0)).float()
        exp = torch.exp(sim)
        log_prob = sim - torch.log(exp.sum(1,keepdim=True))
        mean_log_prob = (mask*log_prob).sum(1)/mask.sum(1)
        return -mean_log_prob.mean()

class Classifier(nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid,1)
        )
    def forward(self,x): return self.net(x).squeeze(-1)

def train_proj(model, data, epochs=EPOCHS_PROJ):
    dl = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = SupConLoss()
    model.to(DEVICE)
    for e in range(epochs):
        model.train(); total=0
        for b in tqdm(dl, desc=f"Projection epoch {e+1}/{epochs}"):
            x = torch.tensor(np.stack([i["embedding"] for i in b]), device=DEVICE)
            y = torch.tensor([i["label"] for i in b], device=DEVICE)
            opt.zero_grad(); out = model(x); loss = crit(out,y); loss.backward(); opt.step()
            total += loss.item()*x.size(0)
        print(f"Epoch {e+1} loss={total/len(data):.4f}")
    return model

def evaluate(model, dl):
    model.eval(); preds=[]; trues=[]
    with torch.no_grad():
        for b in dl:
            x=torch.tensor(np.stack([i["fused"] for i in b]),device=DEVICE)
            y=np.array([i["label"] for i in b])
            p=(torch.sigmoid(model(x))>=0.5).int().cpu().numpy()
            preds.extend(p); trues.extend(y)
    acc=accuracy_score(trues,preds)
    p,r,f,_=precision_recall_fscore_support(trues,preds,average="binary",zero_division=0)
    return acc,p,r,f

def train_classifier(model, train,val,proj):
    def fuse(dataset):
        fused=[]
        for i in tqdm(dataset,desc="Fusing"):
            with torch.no_grad():
                p=proj(torch.tensor(i["embedding"],device=DEVICE).unsqueeze(0)).cpu().numpy()[0]
            fused.append({"fused":np.concatenate([p,i["handcrafted"]]),"label":i["label"]})
        return fused
    tr,va=fuse(train),fuse(val)
    dl_tr = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)
    dl_va = DataLoader(va, batch_size=BATCH_SIZE, collate_fn=lambda x: x)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    crit=nn.BCEWithLogitsLoss()
    model.to(DEVICE)
    best=0
    for e in range(EPOCHS_CLF):
        model.train(); tot=0
        for b in tqdm(dl_tr,desc=f"Classifier epoch {e+1}/{EPOCHS_CLF}"):
            x=torch.tensor(np.stack([i["fused"] for i in b]),device=DEVICE)
            y=torch.tensor([i["label"] for i in b],dtype=torch.float32,device=DEVICE)
            opt.zero_grad(); loss=crit(model(x),y); loss.backward(); opt.step()
            tot+=loss.item()*x.size(0)
        acc,p,r,f=evaluate(model,dl_va)
        print(f"Epoch {e+1}: loss={tot/len(tr):.4f} valF1={f:.4f}")
        if f>best:
            best=f; torch.save(model.state_dict(), os.path.join(OUT_DIR,"best_classifier.pt"))
    return model

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Device:", DEVICE)
    df = pd.read_csv(DATA_CSV)
    assert {"text","label"} <= set(df.columns), "CSV must have 'text' and 'label'"
    tr,te=train_test_split(df,test_size=0.2,stratify=df["label"],random_state=42)
    tr,val=train_test_split(tr,test_size=0.1,stratify=tr["label"],random_state=42)

    print("Loading models...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()

    train_ds, val_ds, test_ds = TextData(tr,embedder,tok,gpt2), TextData(val,embedder,tok,gpt2), TextData(te,embedder,tok,gpt2)

    emb_dim = embedder.get_sentence_embedding_dimension()
    proj = ProjectionHead(emb_dim)
    proj = train_proj(proj, train_ds)


    sample_hf = handcrafted("sample text", tok, gpt2)
    clf = Classifier(128 + len(sample_hf))
    clf = train_classifier(clf, train_ds, val_ds, proj)


    print("Evaluating on test set")
    fused=[]
    for i in tqdm(test_ds):
        with torch.no_grad():
            p=proj(torch.tensor(i["embedding"],device=DEVICE).unsqueeze(0)).cpu().numpy()[0]
        fused.append({"fused":np.concatenate([p,i["handcrafted"]]),"label":i["label"]})
    acc,p,r,f=evaluate(clf,DataLoader(fused, batch_size=BATCH_SIZE, collate_fn=lambda x: x))
    print(f"TEST RESULTS: Acc={acc:.4f}  Prec={p:.4f}  Rec={r:.4f}  F1={f:.4f}")
    torch.save(proj.state_dict(), os.path.join(OUT_DIR,"projection_head.pt"))
    meta={"emb_dim":emb_dim,"proj_out":128,"handcrafted_dim":len(sample_hf)}
    json.dump(meta, open(os.path.join(OUT_DIR,"meta.json"),"w"), indent=2)
    print("All done. Models saved to", OUT_DIR)

import nltk
nltk.download('punkt_tab')

import nltk
nltk.download('averaged_perceptron_tagger_eng')

if __name__ == "__main__":
    main()