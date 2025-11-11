import os, sys

def resource_path(rel_path: str) -> str:
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, rel_path)

# tenta .db e .dB (independente do que subiu)
def pick_db_path():
    for name in ["sorteio_carros100k.db", "sorteio_carros100k.dB"]:
        p = resource_path(name)
        if os.path.exists(p):
            return p
    return resource_path("sorteio_carros100k.db")

DB_PATH    = pick_db_path()
MODELS_DIR = resource_path("models_gui")


# -*- coding: utf-8 -*-
"""
estimador_gui.py
Interface Tkinter para estimar tempo de venda (dias) por 3 abordagens:
1) Empírico direto do banco (quantis por filtros adaptativos)
2) KNN (vizinhos mais próximos em Ano e KM, com reforço por cor e perfil automático)
3) Modelo de Quantis (GradientBoostingRegressor: 10/50/90%)

Banco padrão: sorteio_carros100k.db (tabela: sorteio)
Menu "Arquivo -> Abrir banco..." permite trocar o .db na hora.
"""

import os, json, sqlite3, warnings, math, csv
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# -------------------- CONFIG --------------------
DEFAULT_DB = "sorteio_carros100k.db"
TABLE      = "sorteio"
MODELS_DIR = "models_gui"
ANO_ATUAL  = 2025
QUANTIS    = [0.1, 0.5, 0.9]     # ≈ min, med, max
K_DEFAULT  = 50                  # vizinhos para KNN
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------- ESTADO GLOBAL --------------------
DF_BASE = None
LIST_MARCAS = []
LIST_CORES  = []
LIST_PERFIS = []
FEATURE_COLS_CACHE = None
MODEL_CACHE = None  # (pre, models, meta)
LAST_RESULTS = None
CURRENT_DB_PATH = os.path.abspath(DEFAULT_DB)

# ----------- UTIL: km/ano e perfil automático -----------
def km_ano_e_perfil(ano: int, km_total: int, ano_atual: int = ANO_ATUAL):
    """
    Perfil automático:
      ≤ 7.000  -> Pouco Rodado
      ≤ 12.000 -> Normal
      ≤ 20.000 -> Alto
      > 20.000 -> Super Alto
    """
    idade = max(0, int(ano_atual) - int(ano))
    km_ano = float(km_total) / (idade if idade > 0 else 0.5)
    if km_ano <= 7000:
        perfil = "Pouco Rodado"
    elif km_ano <= 12000:
        perfil = "Normal"
    elif km_ano <= 20000:
        perfil = "Alto"
    else:
        perfil = "Super Alto"
    return km_ano, perfil

def map_estado_para_nota(estado: str) -> float:
    if not estado: return 4.0
    e = estado.strip().lower()
    mapa = {
        "ruim": 3.0, "fraco": 3.0, "fraca": 3.0,
        "medio": 3.5, "médio": 3.5, "regular": 3.5,
        "bom": 4.0, "boa": 4.0,
        "otimo": 4.5, "ótimo": 4.5, "muito bom": 4.5,
        "excelente": 4.9
    }
    return float(mapa.get(e, 4.0))

# -------------- CARREGAR DADOS BASE --------------
def load_table(db_path: str, table: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Não encontrei o banco: {db_path}")
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", con)
    finally:
        con.close()
    if "Dias_para_Venda" not in df.columns:
        raise ValueError(f"A tabela {table} precisa ter 'Dias_para_Venda'.")
    return df

def prepare_df_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tmp = df.apply(lambda r: km_ano_e_perfil(r["Ano"], r["Quilometragem_Estimada"]), axis=1)
    df["KM_Ano_auto"] = tmp.apply(lambda x: x[0])
    df["Perfil_Rodagem_auto"] = tmp.apply(lambda x: x[1])
    return df

def refresh_lists_from_df():
    global LIST_MARCAS, LIST_CORES, LIST_PERFIS
    LIST_MARCAS = sorted(DF_BASE["Marca"].dropna().unique().tolist())
    LIST_CORES  = sorted(DF_BASE["Cor"].dropna().unique().tolist())
    LIST_PERFIS = sorted(DF_BASE["Perfil_Rodagem_auto"].dropna().unique().tolist())

# --------- PREPROCESSAMENTO E TREINO (MODELO QUANTIS) ---------
def build_preprocessor(categorical_cols, numeric_cols):
    # compatível com várias versões do scikit-learn
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    pre = ColumnTransformer(
        transformers=[
            ("cat", ohe, categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )
    return pre

def build_features(df: pd.DataFrame):
    df = df.copy()
    df["Idade"] = (ANO_ATUAL - df["Ano"]).clip(lower=0)
    # garante colunas automáticas
    if "KM_Ano_auto" not in df.columns or "Perfil_Rodagem_auto" not in df.columns:
        tmp = df.apply(lambda r: km_ano_e_perfil(r["Ano"], r["Quilometragem_Estimada"]), axis=1)
        df["KM_Ano_auto"] = tmp.apply(lambda x: x[0])
        df["Perfil_Rodagem_auto"] = tmp.apply(lambda x: x[1])

    for c in ["Nota_Confianca_Marca","Nota_Confianca_Modelo","Nota_Aparencia"]:
        if c in df.columns:
            df[c] = df[c].fillna(3.0)

    feature_cols = [
        "Marca","Modelo","Cor","Perfil_Rodagem_auto",
        "Nota_Confianca_Marca","Nota_Confianca_Modelo","Nota_Aparencia",
        "Quilometragem_Estimada","Ano","Idade","KM_Ano_auto"
    ]
    return df[feature_cols + ["Dias_para_Venda"]], feature_cols

def compute_sample_weights(df_like_X_with_y: pd.DataFrame) -> np.ndarray:
    key = df_like_X_with_y["Marca"].astype(str) + "||" + df_like_X_with_y["Modelo"].astype(str)
    counts = key.value_counts()
    med = counts.median()
    w = key.map(lambda k: float(med) / float(counts[k]))
    return w.clip(lower=0.25, upper=4.0).values

def fit_quantile_models(X, y, sample_weight, quantis=QUANTIS, random_state=42):
    models = {}
    for q in quantis:
        model = GradientBoostingRegressor(
            loss="quantile", alpha=q,
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=random_state
        )
        model.fit(X, y, sample_weight=sample_weight)
        models[q] = model
    return models

def train_models(df_raw: pd.DataFrame, models_dir: str = MODELS_DIR):
    global FEATURE_COLS_CACHE
    df, feature_cols = build_features(df_raw)
    X = df[feature_cols]
    y = df["Dias_para_Venda"].astype(float).values

    cat_cols = ["Marca","Modelo","Cor","Perfil_Rodagem_auto"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = build_preprocessor(cat_cols, num_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    pre.fit(X_train)
    Xt_train = pre.transform(X_train)
    Xt_test  = pre.transform(X_test)

    w_all = compute_sample_weights(X.assign(Dias_para_Venda=y))
    w_train = w_all[X_train.index]

    models = fit_quantile_models(Xt_train, y_train, sample_weight=w_train)
    y_pred_med = models[0.5].predict(Xt_test)
    mae = mean_absolute_error(y_test, y_pred_med)

    # salvar artefatos
    joblib.dump(pre, os.path.join(models_dir, "preprocessor.pkl"))
    for q, m in models.items():
        joblib.dump(m, os.path.join(models_dir, f"gbr_q{int(q*100)}.pkl"))
    meta = {"feature_cols": feature_cols, "quantis": QUANTIS}
    with open(os.path.join(models_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    FEATURE_COLS_CACHE = feature_cols
    return mae

def load_artifacts(models_dir: str = MODELS_DIR):
    pre = joblib.load(os.path.join(models_dir, "preprocessor.pkl"))
    with open(os.path.join(models_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    models = {q: joblib.load(os.path.join(models_dir, f"gbr_q{int(q*100)}.pkl")) for q in meta["quantis"]}
    return pre, models, meta

def ensure_model_trained():
    global MODEL_CACHE, FEATURE_COLS_CACHE
    try:
        MODEL_CACHE = load_artifacts(MODELS_DIR)
        FEATURE_COLS_CACHE = MODEL_CACHE[2]["feature_cols"]
    except Exception:
        mae = train_models(DF_BASE, MODELS_DIR)
        MODEL_CACHE = load_artifacts(MODELS_DIR)
        FEATURE_COLS_CACHE = MODEL_CACHE[2]["feature_cols"]
        messagebox.showinfo("Modelo", f"Treino concluído.\nMAE (mediana, teste): {mae:.2f} dias")

# ----------- MÉTODO A: EMPÍRICO DIRETO DO BANCO -----------
def estimativa_empirica(df_base: pd.DataFrame, marca, modelo, ano, km,
                        janela_anos=2, janela_km_perc=0.20, min_amostra=30):
    df = df_base[(df_base["Marca"] == marca) & (df_base["Modelo"] == modelo)].copy()
    if df.empty:
        return None

    ano_min, ano_max = ano - janela_anos, ano + janela_anos
    km_min, km_max   = km * (1 - janela_km_perc), km * (1 + janela_km_perc)
    sub = df[(df["Ano"].between(ano_min, ano_max)) &
             (df["Quilometragem_Estimada"].between(km_min, km_max))].copy()

    widen = 0
    while len(sub) < min_amostra and widen < 3:
        widen += 1
        ano_min, ano_max = ano - (janela_anos + widen), ano + (janela_anos + widen)
        p2 = janela_km_perc + 0.10*widen
        km_min, km_max = km * (1 - p2), km * (1 + p2)
        sub = df[(df["Ano"].between(ano_min, ano_max)) &
                 (df["Quilometragem_Estimada"].between(km_min, km_max))].copy()

    if sub.empty:
        return None

    dias = sub["Dias_para_Venda"].astype(float)
    q10, q50, q90 = dias.quantile([0.10, 0.50, 0.90])
    media, desvio = dias.mean(), dias.std(ddof=1)

    return {
        "label": "Empírico (filtros)",
        "n": int(len(sub)),
        "min": round(q10,1), "med": round(q50,1), "max": round(q90,1),
        "mean_std": f"{media:.1f} ± {desvio:.1f}"
    }

# ----------- MÉTODO B: KNN -----------
def estimativa_knn_auto(df_base: pd.DataFrame, marca, modelo, ano, km, cor=None, k=K_DEFAULT):
    df = df_base[(df_base["Marca"] == marca) & (df_base["Modelo"] == modelo)].copy()
    if df.empty:
        return None

    # garante colunas automáticas
    if "KM_Ano_auto" not in df.columns or "Perfil_Rodagem_auto" not in df.columns:
        tmp = df.apply(lambda r: km_ano_e_perfil(r["Ano"], r["Quilometragem_Estimada"]), axis=1)
        df["KM_Ano_auto"] = tmp.apply(lambda x: x[0])
        df["Perfil_Rodagem_auto"] = tmp.apply(lambda x: x[1])

    X = df[["Ano","Quilometragem_Estimada"]].astype(float).values
    xq = np.array([[float(ano), float(km)]])

    # Cor (opcional)
    if cor is not None:
        df["match_cor"] = (df["Cor"]==cor).astype(int)
        X = np.hstack([X, df[["match_cor"]].values])
        xq = np.hstack([xq, [[1]]])

    # Perfil automático do query
    km_ano_q, perfil_q = km_ano_e_perfil(ano, km)
    df["match_perfil"] = (df["Perfil_Rodagem_auto"]==perfil_q).astype(int)
    X = np.hstack([X, df[["match_perfil"]].values])
    xq = np.hstack([xq, [[1]]])

    k = min(int(k), len(X))
    if k == 0:
        return None

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    dist, idx = nn.kneighbors(xq, n_neighbors=k)
    viz = df.iloc[idx[0]].copy()
    dias = viz["Dias_para_Venda"].astype(float)

    q10, q50, q90 = dias.quantile([0.10, 0.50, 0.90])

    return {
        "label": f"KNN (k={k}, perfil={perfil_q}, km/ano≈{km_ano_q:.0f})",
        "n": int(len(viz)),
        "min": round(q10,1), "med": round(q50,1), "max": round(q90,1),
        "mean_std": f"{dias.mean():.1f} ± {dias.std(ddof=1):.1f}"
    }

# ----------- MÉTODO C: MODELO DE QUANTIS -----------
def predict_quantis(row_dict, pre, models, feature_cols):
    km_ano, perfil = km_ano_e_perfil(row_dict["Ano"], row_dict["Quilometragem_Estimada"])
    row = {
        "Marca": row_dict["Marca"],
        "Modelo": row_dict["Modelo"],
        "Cor": row_dict["Cor"],
        "Perfil_Rodagem_auto": perfil,
        "Nota_Confianca_Marca": float(row_dict["Nota_Confianca_Marca"]),
        "Nota_Confianca_Modelo": float(row_dict["Nota_Confianca_Modelo"]),
        "Nota_Aparencia": float(row_dict["Nota_Aparencia"]),
        "Quilometragem_Estimada": int(row_dict["Quilometragem_Estimada"]),
        "Ano": int(row_dict["Ano"]),
        "Idade": max(0, ANO_ATUAL - int(row_dict["Ano"])),
        "KM_Ano_auto": km_ano
    }
    X = pd.DataFrame([row])[feature_cols]
    Xt = pre.transform(X)
    preds = {q: float(models[q].predict(Xt)[0]) for q in models}
    return {
        "label": f"Modelo (Quantis, perfil={perfil}, km/ano≈{km_ano:.0f})",
        "n": None,
        "min": round(preds.get(0.1, np.nan),1),
        "med": round(preds.get(0.5, np.nan),1),
        "max": round(preds.get(0.9, np.nan),1),
        "mean_std": "—"
    }

# -------------------- GUI (Tkinter) --------------------
root = tk.Tk()
root.title("Estimador de Tempo de Venda (3 métodos)")
root.geometry("1024x680")

# ---- MENUS ----
menubar = tk.Menu(root)
root.config(menu=menubar)

def menu_abrir_db():
    global DF_BASE, CURRENT_DB_PATH, LIST_MARCAS, LIST_CORES
    path = filedialog.askopenfilename(
        title="Abrir banco SQLite",
        filetypes=[("SQLite DB", "*.db"), ("Todos", "*.*")]
    )
    if not path:
        return
    try:
        df = load_table(path, TABLE)
        df = prepare_df_base(df)
        DF_BASE = df
        CURRENT_DB_PATH = os.path.abspath(path)
        refresh_lists_from_df()
        # atualiza combos
        cb_marca["values"] = LIST_MARCAS
        cb_cor["values"]   = LIST_CORES if LIST_CORES else ["Branco"]
        if LIST_MARCAS:
            cb_marca.set(LIST_MARCAS[0])
            on_select_marca()
        if LIST_CORES:
            cb_cor.set(LIST_CORES[0])
        lbl_db.config(text=f"Banco: {CURRENT_DB_PATH} | Tabela: {TABLE}")
        messagebox.showinfo("Banco", f"Banco carregado:\n{CURRENT_DB_PATH}")
    except Exception as e:
        messagebox.showerror("Erro ao abrir banco", str(e))

def menu_sobre():
    messagebox.showinfo(
        "Sobre",
        "Estimador de Tempo de Venda\nMétodos: Empírico, KNN e Quantis (GBR)\nFeito para Paulo 😉"
    )

men_arquivo = tk.Menu(menubar, tearoff=0)
men_arquivo.add_command(label="Abrir banco...", command=menu_abrir_db)
men_arquivo.add_separator()
men_arquivo.add_command(label="Sair", command=root.destroy)
menubar.add_cascade(label="Arquivo", menu=men_arquivo)

men_ajuda = tk.Menu(menubar, tearoff=0)
men_ajuda.add_command(label="Sobre", command=menu_sobre)
menubar.add_cascade(label="Ajuda", menu=men_ajuda)

# ---- FRAMES ----
main = ttk.Frame(root, padding=12)
main.pack(fill="both", expand=True)

# Linha 1: seleção de marca/modelo
row1 = ttk.Frame(main); row1.pack(fill="x", pady=4)
ttk.Label(row1, text="Marca:").pack(side="left")
cb_marca = ttk.Combobox(row1, values=[], width=22, state="readonly")
cb_marca.pack(side="left", padx=6)

ttk.Label(row1, text="Modelo:").pack(side="left")
cb_modelo = ttk.Combobox(row1, values=[], width=30, state="readonly")
cb_modelo.pack(side="left", padx=6)

def on_select_marca(event=None):
    marca = cb_marca.get()
    if not marca or DF_BASE is None:
        cb_modelo["values"] = []
        return
    lista = sorted(DF_BASE.loc[DF_BASE["Marca"]==marca, "Modelo"].dropna().unique().tolist())
    cb_modelo["values"] = lista
cb_marca.bind("<<ComboboxSelected>>", on_select_marca)

# Linha 2: ano/km/cor
row2 = ttk.Frame(main); row2.pack(fill="x", pady=4)
ttk.Label(row2, text="Ano:").pack(side="left")
ent_ano = ttk.Entry(row2, width=8); ent_ano.insert(0, "2021"); ent_ano.pack(side="left", padx=6)

ttk.Label(row2, text="KM:").pack(side="left")
ent_km = ttk.Entry(row2, width=12); ent_km.insert(0, "45000"); ent_km.pack(side="left", padx=6)

ttk.Label(row2, text="Cor:").pack(side="left")
cb_cor = ttk.Combobox(row2, values=[], width=16, state="readonly")
cb_cor.pack(side="left", padx=6)

# Linha 3: estado/nota e K
row3 = ttk.Frame(main); row3.pack(fill="x", pady=4)
ttk.Label(row3, text="Estado (ruim/medio/bom/otimo/excelente):").pack(side="left")
cb_estado = ttk.Combobox(row3, values=["ruim","medio","bom","otimo","excelente"], width=18, state="readonly")
cb_estado.set("bom"); cb_estado.pack(side="left", padx=6)

ttk.Label(row3, text="ou Nota Aparência (3.0–5.0):").pack(side="left")
ent_nota = ttk.Entry(row3, width=6); ent_nota.insert(0, ""); ent_nota.pack(side="left", padx=6)

ttk.Label(row3, text="K (KNN):").pack(side="left")
ent_k = ttk.Entry(row3, width=6); ent_k.insert(0, str(K_DEFAULT)); ent_k.pack(side="left", padx=6)

# Botões
row_btn = ttk.Frame(main); row_btn.pack(fill="x", pady=8)
btn_treinar = ttk.Button(row_btn, text="Treinar/Carregar Modelo", width=24)
btn_calcular = ttk.Button(row_btn, text="Calcular (3 métodos)", width=24)
btn_export   = ttk.Button(row_btn, text="Exportar tabela (CSV)", width=24)
btn_treinar.pack(side="left", padx=6)
btn_calcular.pack(side="left", padx=6)
btn_export.pack(side="left", padx=6)

# Tabela de resultados
cols = ("metodo","n","min","med","max","mean_std")
tree = ttk.Treeview(main, columns=cols, show="headings", height=12)
for c, txt, w in [
    ("metodo","Método",320),
    ("n","N",60),
    ("min","Mín",80),
    ("med","Med",80),
    ("max","Máx",80),
    ("mean_std","Média ± DP",180),
]:
    tree.heading(c, text=txt)
    tree.column(c, width=w, anchor="center")
tree.pack(fill="both", expand=True, pady=6)

# Resumo/consenso
lbl_consenso = ttk.Label(main, text="Consenso: —", font=("TkDefaultFont", 11, "bold"))
lbl_consenso.pack(anchor="w", pady=6)

# Rodapé com caminho do banco
lbl_db = ttk.Label(main, text=f"Banco: {CURRENT_DB_PATH} | Tabela: {TABLE}", foreground="#555")
lbl_db.pack(anchor="w", pady=4)

# -------------- FUNÇÕES DE AÇÃO --------------
def init_load_default_db():
    global DF_BASE, CURRENT_DB_PATH
    try:
        DF_BASE = load_table(DEFAULT_DB, TABLE)
        DF_BASE = prepare_df_base(DF_BASE)
        CURRENT_DB_PATH = os.path.abspath(DEFAULT_DB)
    except Exception as e:
        messagebox.showwarning("Banco padrão", f"Não consegui abrir {DEFAULT_DB}.\nUse Arquivo -> Abrir banco...\n\n{e}")
        return
    refresh_lists_from_df()
    cb_marca["values"] = LIST_MARCAS
    cb_cor["values"]   = LIST_CORES if LIST_CORES else ["Branco"]
    if LIST_MARCAS:
        cb_marca.set(LIST_MARCAS[0]); on_select_marca()
    if LIST_CORES:
        cb_cor.set(LIST_CORES[0])
    lbl_db.config(text=f"Banco: {CURRENT_DB_PATH} | Tabela: {TABLE}")

def action_treinar():
    global MODEL_CACHE, FEATURE_COLS_CACHE
    if DF_BASE is None:
        messagebox.showerror("Erro", "Carregue um banco primeiro (Arquivo -> Abrir banco...).")
        return
    try:
        mae = train_models(DF_BASE, MODELS_DIR)
        MODEL_CACHE = load_artifacts(MODELS_DIR)
        FEATURE_COLS_CACHE = MODEL_CACHE[2]["feature_cols"]
        messagebox.showinfo("Modelo", f"Treino concluído.\nMAE (mediana, teste): {mae:.2f} dias")
    except Exception as e:
        messagebox.showerror("Erro ao treinar", str(e))

def parse_inputs():
    if DF_BASE is None:
        raise ValueError("Carregue um banco primeiro (Arquivo -> Abrir banco...).")
    marca = cb_marca.get().strip()
    modelo = cb_modelo.get().strip()
    if not marca or not modelo:
        raise ValueError("Selecione Marca e Modelo.")
    try:
        ano = int(ent_ano.get().strip())
        km  = int(ent_km.get().strip())
    except:
        raise ValueError("Ano e KM precisam ser números inteiros.")
    cor = cb_cor.get().strip() or None

    nota_txt = ent_nota.get().strip()
    if nota_txt:
        try:
            nota = float(nota_txt)
        except:
            raise ValueError("Nota de aparência inválida (use número, ex.: 4.2).")
    else:
        nota = map_estado_para_nota(cb_estado.get())

    # notas de marca/modelo a partir do dataset (média) como fallback
    nota_marca  = DF_BASE.loc[DF_BASE["Marca"]==marca, "Nota_Confianca_Marca"].dropna().mean()
    nota_modelo = DF_BASE.loc[(DF_BASE["Marca"]==marca)&(DF_BASE["Modelo"]==modelo), "Nota_Confianca_Modelo"].dropna().mean()
    if math.isnan(nota_marca):  nota_marca  = 4.0
    if math.isnan(nota_modelo): nota_modelo = nota_marca

    return {
        "Marca": marca, "Modelo": modelo, "Ano": ano,
        "Quilometragem_Estimada": km, "Cor": cor,
        "Nota_Confianca_Marca": float(nota_marca),
        "Nota_Confianca_Modelo": float(nota_modelo),
        "Nota_Aparencia": float(nota)
    }

def action_calcular():
    global LAST_RESULTS
    for i in tree.get_children():
        tree.delete(i)
    try:
        entrada = parse_inputs()
    except Exception as e:
        messagebox.showerror("Entrada inválida", str(e))
        return

    # A) Empírico por filtros
    r_emp = estimativa_empirica(DF_BASE, entrada["Marca"], entrada["Modelo"], entrada["Ano"], entrada["Quilometragem_Estimada"])
    # B) KNN
    try:
        k = int(ent_k.get().strip())
    except:
        k = K_DEFAULT
    r_knn = estimativa_knn_auto(
        DF_BASE, entrada["Marca"], entrada["Modelo"], entrada["Ano"], entrada["Quilometragem_Estimada"],
        cor=entrada["Cor"], k=k
    )
    # C) Modelo Quantis
    try:
        if MODEL_CACHE is None or FEATURE_COLS_CACHE is None:
            ensure_model_trained()
        pre, models, meta = MODEL_CACHE
        r_ml = predict_quantis(entrada, pre, models, FEATURE_COLS_CACHE)
    except Exception as e:
        r_ml = None
        messagebox.showwarning("Modelo", f"Não foi possível usar o modelo: {e}")

    resultados = [r for r in [r_emp, r_knn, r_ml] if r is not None]
    if not resultados:
        messagebox.showinfo("Sem resultados", "Nenhum método retornou estimativa para esses filtros.")
        lbl_consenso.config(text="Consenso: —")
        LAST_RESULTS = None
        return

    for r in resultados:
        tree.insert("", "end", values=(r["label"], r["n"], r["min"], r["med"], r["max"], r["mean_std"]))

    # consenso simples
    mins = [r["min"] for r in resultados]
    meds = [r["med"] for r in resultados]
    maxs = [r["max"] for r in resultados]
    cons_min = round(float(np.median(mins)), 1)
    cons_med = round(float(np.median(meds)), 1)
    cons_max = round(float(np.median(maxs)), 1)

    km_ano_q, perfil_q = km_ano_e_perfil(entrada["Ano"], entrada["Quilometragem_Estimada"])
    lbl_consenso.config(
        text=f"Perfil auto: {perfil_q} (km/ano≈{km_ano_q:.0f}) | Consenso: mínimo≈{cons_min} • média≈{cons_med} • máximo≈{cons_max} dias"
    )

    LAST_RESULTS = {
        "entrada": entrada,
        "resultados": resultados,
        "consenso": {"min": cons_min, "med": cons_med, "max": cons_max},
        "db": CURRENT_DB_PATH
    }

def action_export():
    if not LAST_RESULTS:
        messagebox.showinfo("Exportar", "Calcule primeiro para exportar a tabela.")
        return
    path = filedialog.asksaveasfilename(
        title="Salvar resultados como CSV",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv")]
    )
    if not path:
        return
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["Banco", LAST_RESULTS["db"]])
            ent = LAST_RESULTS["entrada"]
            w.writerow(["Entrada"])
            for k, v in ent.items():
                w.writerow([k, v])
            w.writerow([])
            w.writerow(["Método","N","Mín","Med","Máx","Média ± DP"])
            for r in LAST_RESULTS["resultados"]:
                w.writerow([r["label"], r["n"], r["min"], r["med"], r["max"], r["mean_std"]])
            w.writerow([])
            cons = LAST_RESULTS["consenso"]
            w.writerow(["Consenso", "", cons["min"], cons["med"], cons["max"], ""])
        messagebox.showinfo("Exportar", f"Arquivo salvo:\n{path}")
    except Exception as e:
        messagebox.showerror("Exportar", str(e))

btn_treinar.config(command=action_treinar)
btn_calcular.config(command=action_calcular)
btn_export.config(command=action_export)

# Carrega DB padrão na inicialização
init_load_default_db()

root.mainloop()
