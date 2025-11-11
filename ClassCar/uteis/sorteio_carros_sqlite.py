# -*- coding: utf-8 -*-
"""
Gerador de dataset de carros + salvamento em SQLite/CSV.

Como usar (terminal):
    python sorteio_carros_sqlite.py --rows 1000 --out sorteio_carros.db --csv sorteio_carros.csv --seed 42
    python sorteio_carros_sqlite.py --rows 100000 --out sorteio_100k.db

Se quiser, troque as distribuições/dicionários marcados com "TODO" pelos seus reais.
"""

import random
import math
import sqlite3
import csv
import os
from typing import List, Dict

# ---------------------------------------------------------------------
# --- DADOS DE CONFIABILIDADE (SEUS, MANTIDOS) ------------------------
# ---------------------------------------------------------------------

classificacao_modelos_nota = [
    {"Marca": "Toyota", "Modelo": "Corolla", "Nota_Confianca": 5.0}, {"Marca": "Toyota", "Modelo": "Hilux", "Nota_Confianca": 4.9},
    {"Marca": "Honda", "Modelo": "Civic", "Nota_Confianca": 4.8}, {"Marca": "Toyota", "Modelo": "Corolla Cross", "Nota_Confianca": 4.7},
    {"Marca": "Hyundai", "Modelo": "HB20", "Nota_Confianca": 4.5}, {"Marca": "Volkswagen", "Modelo": "Gol", "Nota_Confianca": 4.4},
    {"Marca": "Fiat", "Modelo": "Strada", "Nota_Confianca": 4.4}, {"Marca": "Fiat", "Modelo": "Uno", "Nota_Confianca": 4.3},
    {"Marca": "Chevrolet", "Modelo": "Onix", "Nota_Confianca": 4.2}, {"Marca": "Chevrolet", "Modelo": "Celta", "Nota_Confianca": 4.2},
    {"Marca": "Volkswagen", "Modelo": "Fox", "Nota_Confianca": 4.1}, {"Marca": "Volkswagen", "Modelo": "Saveiro", "Nota_Confianca": 4.0},
    {"Marca": "Honda", "Modelo": "HR-V", "Nota_Confianca": 3.9}, {"Marca": "Ford", "Modelo": "Ka", "Nota_Confianca": 3.8},
    {"Marca": "Nissan", "Modelo": "Kicks", "Nota_Confianca": 3.7}, {"Marca": "Hyundai", "Modelo": "Creta", "Nota_Confianca": 3.6},
    {"Marca": "Fiat", "Modelo": "Mobi", "Nota_Confianca": 3.5}, {"Marca": "Fiat", "Modelo": "Argo", "Nota_Confianca": 3.4},
    {"Marca": "Fiat", "Modelo": "Siena/Grand Siena", "Nota_Confianca": 3.3}, {"Marca": "Renault", "Modelo": "Duster", "Nota_Confianca": 3.2},
    {"Marca": "Chevrolet", "Modelo": "Prisma/Joy Plus", "Nota_Confianca": 3.2}, {"Marca": "Volkswagen", "Modelo": "T-Cross", "Nota_Confianca": 3.1},
    {"Marca": "Volkswagen", "Modelo": "Nivus", "Nota_Confianca": 3.0}, {"Marca": "Chevrolet", "Modelo": "Tracker", "Nota_Confianca": 3.0},
    {"Marca": "Fiat", "Modelo": "Toro", "Nota_Confianca": 2.9}, {"Marca": "Volkswagen", "Modelo": "Polo", "Nota_Confianca": 2.8},
    {"Marca": "Renault", "Modelo": "Kwid", "Nota_Confianca": 2.7}, {"Marca": "Renault", "Modelo": "Sandero", "Nota_Confianca": 2.6},
    {"Marca": "Ford", "Modelo": "EcoSport", "Nota_Confianca": 2.5}, {"Marca": "Jeep", "Modelo": "Renegade", "Nota_Confianca": 2.4},
    {"Marca": "Jeep", "Modelo": "Compass", "Nota_Confianca": 2.3}, {"Marca": "Ford", "Modelo": "Fiesta", "Nota_Confianca": 2.0},
]

classificacao_marcas_nota = [
    {"Marca": "Toyota", "Nota_Confianca": 5.0}, {"Marca": "Honda", "Nota_Confianca": 4.8},
    {"Marca": "Hyundai", "Nota_Confianca": 4.5}, {"Marca": "Chevrolet", "Nota_Confianca": 4.3},
    {"Marca": "Volkswagen", "Nota_Confianca": 4.0}, {"Marca": "Nissan", "Nota_Confianca": 3.8},
    {"Marca": "Fiat", "Nota_Confianca": 3.6}, {"Marca": "Renault", "Nota_Confianca": 3.2},
    {"Marca": "Ford", "Nota_Confianca": 2.8}, {"Marca": "Jeep", "Nota_Confianca": 2.5}
]

# ---------------------------------------------------------------------
# --- OUTRAS DISTRIBUIÇÕES (PODE TROCAR PELAS SUAS) -------------------
# ---------------------------------------------------------------------

# TODO: substitua pelos seus perfis/valores
perfis_uso = [
    {"perfil": "Pouco Rodado", "km_ano": 8000, "probabilidade": 0.25},
    {"perfil": "Normal", "km_ano": 12000, "probabilidade": 0.45},
    {"perfil": "Alto", "km_ano": 18000, "probabilidade": 0.20},
    {"perfil": "Super Alto", "km_ano": 25000, "probabilidade": 0.10},
]

# TODO: substitua pela sua distribuição real
distribuicao_cores = {
    "Branco": 0.28, "Preto": 0.22, "Prata": 0.18, "Cinza": 0.16,
    "Vermelho": 0.06, "Azul": 0.05, "Verde": 0.03, "Marrom": 0.02
}

# TODO: substitua pela sua base completa
carros_acumulado_de_emplacamentos = [
    {"Marca": "Toyota", "Modelo": "Corolla", "Periodo_Analise": (2012, 2025), "Emplacamentos_Acumulados_Aprox": 850000},
    {"Marca": "Toyota", "Modelo": "Hilux", "Periodo_Analise": (2012, 2025), "Emplacamentos_Acumulados_Aprox": 400000},
    {"Marca": "Honda", "Modelo": "Civic", "Periodo_Analise": (2012, 2025), "Emplacamentos_Acumulados_Aprox": 500000},
    {"Marca": "Toyota", "Modelo": "Corolla Cross", "Periodo_Analise": (2021, 2025), "Emplacamentos_Acumulados_Aprox": 280000},
    {"Marca": "Hyundai", "Modelo": "HB20", "Periodo_Analise": (2013, 2025), "Emplacamentos_Acumulados_Aprox": 1200000},
    {"Marca": "Chevrolet", "Modelo": "Onix", "Periodo_Analise": (2013, 2025), "Emplacamentos_Acumulados_Aprox": 1500000},
    {"Marca": "Volkswagen", "Modelo": "Gol", "Periodo_Analise": (2010, 2022), "Emplacamentos_Acumulados_Aprox": 1000000},
    {"Marca": "Fiat", "Modelo": "Strada", "Periodo_Analise": (2010, 2025), "Emplacamentos_Acumulados_Aprox": 950000},
    {"Marca": "Fiat", "Modelo": "Uno", "Periodo_Analise": (2010, 2021), "Emplacamentos_Acumulados_Aprox": 900000},
    {"Marca": "Volkswagen", "Modelo": "Polo", "Periodo_Analise": (2018, 2025), "Emplacamentos_Acumulados_Aprox": 450000},
    {"Marca": "Chevrolet", "Modelo": "Tracker", "Periodo_Analise": (2014, 2025), "Emplacamentos_Acumulados_Aprox": 400000},
    {"Marca": "Jeep", "Modelo": "Renegade", "Periodo_Analise": (2015, 2025), "Emplacamentos_Acumulados_Aprox": 600000},
    {"Marca": "Jeep", "Modelo": "Compass", "Periodo_Analise": (2016, 2025), "Emplacamentos_Acumulados_Aprox": 500000},
    {"Marca": "Renault", "Modelo": "Kwid", "Periodo_Analise": (2017, 2025), "Emplacamentos_Acumulados_Aprox": 350000},
    {"Marca": "Renault", "Modelo": "Sandero", "Periodo_Analise": (2010, 2022), "Emplacamentos_Acumulados_Aprox": 600000},
    {"Marca": "Ford", "Modelo": "Ka", "Periodo_Analise": (2010, 2021), "Emplacamentos_Acumulados_Aprox": 700000},
    {"Marca": "Hyundai", "Modelo": "Creta", "Periodo_Analise": (2017, 2025), "Emplacamentos_Acumulados_Aprox": 450000},
    {"Marca": "Volkswagen", "Modelo": "T-Cross", "Periodo_Analise": (2019, 2025), "Emplacamentos_Acumulados_Aprox": 380000},
    {"Marca": "Volkswagen", "Modelo": "Nivus", "Periodo_Analise": (2020, 2025), "Emplacamentos_Acumulados_Aprox": 200000},
    {"Marca": "Fiat", "Modelo": "Argo", "Periodo_Analise": (2017, 2025), "Emplacamentos_Acumulados_Aprox": 500000},
    {"Marca": "Fiat", "Modelo": "Mobi", "Periodo_Analise": (2016, 2025), "Emplacamentos_Acumulados_Aprox": 450000},
    {"Marca": "Fiat", "Modelo": "Toro", "Periodo_Analise": (2016, 2025), "Emplacamentos_Acumulados_Aprox": 480000},
    {"Marca": "Chevrolet", "Modelo": "Celta", "Periodo_Analise": (2010, 2015), "Emplacamentos_Acumulados_Aprox": 250000},
    {"Marca": "Chevrolet", "Modelo": "Prisma/Joy Plus", "Periodo_Analise": (2013, 2020), "Emplacamentos_Acumulados_Aprox": 400000},
    {"Marca": "Volkswagen", "Modelo": "Fox", "Periodo_Analise": (2010, 2021), "Emplacamentos_Acumulados_Aprox": 500000},
    {"Marca": "Volkswagen", "Modelo": "Saveiro", "Periodo_Analise": (2010, 2025), "Emplacamentos_Acumulados_Aprox": 600000},
    {"Marca": "Ford", "Modelo": "Fiesta", "Periodo_Analise": (2010, 2019), "Emplacamentos_Acumulados_Aprox": 450000},
    {"Marca": "Renault", "Modelo": "Duster", "Periodo_Analise": (2012, 2025), "Emplacamentos_Acumulados_Aprox": 300000},
    {"Marca": "Honda", "Modelo": "HR-V", "Periodo_Analise": (2015, 2025), "Emplacamentos_Acumulados_Aprox": 420000},
]

# TODO: opcional—pese por ano onde você tiver dados reais
distribuicao_real_conhecida = {
    "Toyota Corolla": {str(ano): 100000 + 5000*((ano-2015)%3) for ano in range(2015, 2026)},
    "Chevrolet Onix": {str(ano): 140000 + 7000*((ano-2013)%4) for ano in range(2013, 2026)},
    "Hyundai HB20": {str(ano): 120000 + 6000*((ano-2012)%5) for ano in range(2012, 2026)},
}

# ---------------------------------------------------------------------
# --- FUNÇÕES AUXILIARES (COM FALLBACK CORRIGIDO) ---------------------
# ---------------------------------------------------------------------

def buscar_nota(marca, modelo, tipo):
    # Busca por marca
    if tipo == 'Marca':
        for item in classificacao_marcas_nota:
            if item['Marca'] == marca:
                return item['Nota_Confianca']
        return 3.0  # fallback neutro p/ marca desconhecida

    # Busca por modelo; se não achar, volta p/ nota da marca
    if tipo == 'Modelo':
        for item in classificacao_modelos_nota:
            if item['Marca'] == marca and item['Modelo'] == modelo:
                return item['Nota_Confianca']
        return buscar_nota(marca, None, 'Marca')

    return 3.0

def sortear_cor(distribuicao):
    cores = list(distribuicao.keys())
    pesos = list(distribuicao.values())
    return random.choices(cores, weights=pesos, k=1)[0]

def calcular_fator_sorteio(idade_base):
    fator = 10 + 2 - idade_base
    n = max(2, min(10, fator))
    return n

def sortear_nota_aparencia(idade_base):
    n = calcular_fator_sorteio(idade_base)
    melhor_nota = 0.0
    for _ in range(n):
        nota_atual = random.uniform(3.0, 5.0)
        if nota_atual > melhor_nota:
            melhor_nota = nota_atual
    return round(melhor_nota, 2)

def calcular_dias_para_venda_v2(nota_marca, nota_modelo, nota_aparencia, perfil_rodagem, cor):
    """
    Calcula Dias para Venda (giro) conforme suas regras, com pesos aleatórios 1–4.
    """
    DIAS_BASE = 60
    dias_venda = DIAS_BASE

    # 1) Confiabilidade: subtrai
    dias_venda -= nota_marca * random.uniform(1, 4)
    dias_venda -= nota_modelo * random.uniform(1, 4)
    dias_venda -= nota_aparencia * random.uniform(1, 4)

    # 2) Quilometragem: ajusta
    ajustes_km = {
        "Pouco Rodado": -12,
        "Normal": -8,
        "Alto": 10,
        "Super Alto": 15
    }
    dias_venda += ajustes_km.get(perfil_rodagem, 0)

    # 3) Cor: mais líquidas tiram dias
    if cor.title() in ["Branco", "Preto", "Cinza", "Prata"]:
        dias_venda -= 5

    return max(5, math.floor(dias_venda))

# ---------------------------------------------------------------------
# --- FUNÇÃO PRINCIPAL DE SORTEIO ------------------------------------
# ---------------------------------------------------------------------

def sortear_carros_final(dados_acumulados, distribuicao_anual_real, num_sorteios):
    """
    Monta opções (Marca, Modelo, Ano) ponderadas por distribuição anual real; se não houver,
    usa peso uniforme no período de análise. Sorteia `num_sorteios` unidades e computa os campos.
    """
    opcoes_com_peso = []
    for carro in dados_acumulados:
        marca = carro['Marca']
        modelo = carro['Modelo']
        nome_completo = f"{marca} {modelo}"
        periodo = range(carro['Periodo_Analise'][0], carro['Periodo_Analise'][1] + 1)

        if nome_completo in distribuicao_anual_real:
            vendas_anuais = distribuicao_anual_real[nome_completo]
            for ano in periodo:
                peso_carro_ano = vendas_anuais.get(str(ano), 0)
                if peso_carro_ano > 0:
                    opcoes_com_peso.append({"Marca": marca, "Modelo": modelo, "Ano": ano, "Peso": peso_carro_ano})
        else:
            anos = carro['Periodo_Analise'][1] - carro['Periodo_Analise'][0] + 1
            peso_uniforme = carro['Emplacamentos_Acumulados_Aprox'] / anos
            for ano in periodo:
                opcoes_com_peso.append({"Marca": marca, "Modelo": modelo, "Ano": ano, "Peso": peso_uniforme})

    lista_de_itens = [f"{it['Marca']} {it['Modelo']}|{it['Ano']}" for it in opcoes_com_peso]
    lista_de_pesos = [it['Peso'] for it in opcoes_com_peso]
    resultados_raw = random.choices(lista_de_itens, weights=lista_de_pesos, k=num_sorteios)

    carros_sorteados_final = []
    ANO_ATUAL = 2025

    perfis = [p['perfil'] for p in perfis_uso]
    pesos_perfis = [p['probabilidade'] for p in perfis_uso]
    km_por_perfil = {p['perfil']: p['km_ano'] for p in perfis_uso}

    for i, resultado_raw in enumerate(resultados_raw):
        partes = resultado_raw.split('|')
        marca_modelo = partes[0].split(' ', 1)
        marca = marca_modelo[0]
        modelo = marca_modelo[1]
        ano_fabricacao = int(partes[1])

        # Cor e notas
        cor_sorteada = sortear_cor(distribuicao_cores)
        nota_marca = buscar_nota(marca, None, 'Marca')
        nota_modelo = buscar_nota(marca, modelo, 'Modelo')

        # Aparência
        idade_base = ANO_ATUAL - ano_fabricacao
        nota_aparencia = sortear_nota_aparencia(idade_base)

        # Quilometragem
        fator_aleatorio = round(random.uniform(0, 1), 5)
        idade_calculo = idade_base + fator_aleatorio
        perfil_sorteado = random.choices(perfis, weights=pesos_perfis, k=1)[0]
        km_anual_base = km_por_perfil[perfil_sorteado]
        quilometragem_estimada = math.floor(idade_calculo * km_anual_base)

        # Giro de estoque
        dias_venda = calcular_dias_para_venda_v2(nota_marca, nota_modelo, nota_aparencia, perfil_sorteado, cor_sorteada)

        carros_sorteados_final.append({
            "Ordem_Sorteio": i + 1,
            "Marca": marca,
            "Modelo": modelo,
            "Ano": ano_fabricacao,
            "Cor": cor_sorteada,
            "Nota_Confianca_Marca": round(float(nota_marca), 1),
            "Nota_Confianca_Modelo": round(float(nota_modelo), 1),
            "Nota_Aparencia": float(nota_aparencia),
            "Perfil_Rodagem": perfil_sorteado,
            "Quilometragem_Estimada": int(quilometragem_estimada),
            "Dias_para_Venda": int(dias_venda)
        })

    return carros_sorteados_final

# ---------------------------------------------------------------------
# --- SALVAR EM SQLITE / CSV -----------------------------------------
# ---------------------------------------------------------------------

COLS = [
    "Ordem_Sorteio","Marca","Modelo","Ano","Cor",
    "Nota_Confianca_Marca","Nota_Confianca_Modelo","Nota_Aparencia",
    "Perfil_Rodagem","Quilometragem_Estimada","Dias_para_Venda"
]

def salvar_em_sqlite(registros: List[Dict], caminho_db="sorteio_carros100k.db", tabela="sorteio"):
    con = sqlite3.connect(caminho_db)
    cur = con.cursor()
    # Tweaks de performance: bons para até milhões de linhas em máquina local
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")

    cur.execute(f"DROP TABLE IF EXISTS {tabela};")
    cur.execute(f"""
    CREATE TABLE {tabela} (
        Ordem_Sorteio INTEGER PRIMARY KEY,
        Marca TEXT,
        Modelo TEXT,
        Ano INTEGER,
        Cor TEXT,
        Nota_Confianca_Marca REAL,
        Nota_Confianca_Modelo REAL,
        Nota_Aparencia REAL,
        Perfil_Rodagem TEXT,
        Quilometragem_Estimada INTEGER,
        Dias_para_Venda INTEGER
    );
    """)

    placeholders = ",".join("?" for _ in COLS)
    dados = [tuple(r.get(c) for c in COLS) for r in registros]
    cur.executemany(
        f"INSERT INTO {tabela} ({','.join(COLS)}) VALUES ({placeholders})",
        dados
    )

    # Índices úteis
    cur.execute(f"CREATE INDEX idx_{tabela}_marca_modelo ON {tabela}(Marca, Modelo);")
    cur.execute(f"CREATE INDEX idx_{tabela}_ano ON {tabela}(Ano);")
    cur.execute(f"CREATE INDEX idx_{tabela}_dias ON {tabela}(Dias_para_Venda);")

    con.commit()
    con.close()

def salvar_csv(registros: List[Dict], caminho_csv="sorteio_carros100k.csv"):
    if not registros:
        return
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in registros:
            w.writerow({c: r.get(c) for c in COLS})

def gerar_e_salvar(num_sorteios=10000, caminho_db="sorteio_carros100k.db", tabela="sorteio", seed=None, caminho_csv=None):
    if seed is not None:
        random.seed(seed)

    registros = sortear_carros_final(
        carros_acumulado_de_emplacamentos,
        distribuicao_real_conhecida,
        num_sorteios
    )
    salvar_em_sqlite(registros, caminho_db=caminho_db, tabela=tabela)
    if caminho_csv:
        salvar_csv(registros, caminho_csv)
    print(f"\n✅ Banco gerado: {os.path.abspath(caminho_db)} (tabela: {tabela})")
    if caminho_csv:
        print(f"✅ CSV gerado:   {os.path.abspath(caminho_csv)}")

# ---------------------------------------------------------------------
# --- CLI -------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# --- CLI + GERAÇÃO AUTOMÁTICA DE CSV -------------------------------
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gerar banco SQLite e CSV a partir do sorteio de carros")
    parser.add_argument("--rows", type=int, default=10000, help="Quantidade de carros a sortear (ex.: 1000)")
    parser.add_argument("--out", type=str, default="sorteio_carros100k.db", help="Arquivo .db de saída")
    parser.add_argument("--table", type=str, default="sorteio", help="Nome da tabela no SQLite")
    parser.add_argument("--seed", type=int, default=None, help="Seed aleatória (opcional)")
    args, _unknown = parser.parse_known_args()

    # Nome automático pro CSV (mesmo nome do .db, só mudando a extensão)
    csv_path = os.path.splitext(args.out)[0] + ".csv"

    gerar_e_salvar(
        num_sorteios=args.rows,
        caminho_db=args.out,
        tabela=args.table,
        seed=args.seed,
        caminho_csv=csv_path   # <= agora sempre gera CSV junto
    )
