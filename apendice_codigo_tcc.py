# -*- coding: utf-8 -*-
"""
Apêndice - Código-fonte da análise

Padrões sociodemográficos e acesso a serviços financeiros formais:
regras de associação nos microdados da POF 2017-2018 (IBGE).

Ambiente: Google Colab, pandas 2.2.2, NumPy 2.0.2, mlxtend 0.23.4
e matplotlib.

Etapas:
  1. Leitura dos microdados e definição da variável de acesso
  2. Pré-processamento das variáveis sociodemográficas
  3. Caracterização descritiva da base (Seção 4.1)
  4. Taxa de acesso por categoria (Seção 4.2)
  5. Mineração de regras de associação com Apriori (Seção 4.3)
"""

import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
from google.colab import drive
from mlxtend.frequent_patterns import apriori, association_rules


# ============================================================
# 0. CONFIGURAÇÃO
# ============================================================
# Oculta aviso interno do ambiente Colab (não afeta a análise)
warnings.filterwarnings('ignore', message=r'.*utcnow\(\) is deprecated.*')

drive.mount('/content/drive')
CAMINHO = '/content/drive/MyDrive/POF_TCC'
SAIDA = f'{CAMINHO}/resultados'
os.makedirs(SAIDA, exist_ok=True)

SALARIO_MINIMO = 954      # salário mínimo vigente em 2018 (R$)

# Parâmetros do Apriori
SUPORTE_MIN = 0.05        # itemset presente em pelo menos 5% das UCs
CONFIANCA_MIN = 0.50      # consequente presente em pelo menos 50% dos casos
TAMANHO_MAX = 4           # até 4 itens por itemset

# Códigos de produto da POF (Quadros 26 e 48) que indicam uso
# efetivo de serviços financeiros formais
CODIGOS_FINANCEIROS = [
    '2600101',  # Juros de cheque especial
    '2600201',  # Juros de cartão de crédito
    '2600301',  # Anuidade de cartão de crédito
    '2600401',  # Seguro de cartão de crédito
    '2600501',  # Tarifa de conta bancária
    '2600502',  # Manutenção de conta bancária
    '2600503',  # Manutenção de cheque especial
    '2600504',  # Pacote de serviço de conta bancária
    '2600601',  # Taxa de saque eletrônico
    '2600801',  # Taxa de cartão especial
    '2600901',  # Transferência interbancária (DOC ou TED)
    '2600902',  # DOC (transferência interbancária)
    '2600903',  # TED (transferência interbancária)
    '2601001',  # Talão de cheques
    '2601102',  # Abertura de crédito
    '2601103',  # Renovação de cheque especial
    '4800101',  # Pagamento de empréstimo
    '4800201',  # Juros de empréstimo
    '4800301',  # Seguro de empréstimo
    '4800401',  # Imposto sobre operação financeira
    '4803501',  # Consórcio de dinheiro
]

# Posições (início, fim) dos campos nos arquivos de largura fixa,
# conforme o dicionário de variáveis da POF 2017-2018
CAMPOS_DESPESA = {
    'COD_UPA':     (7, 16),
    'NUM_DOM':     (16, 18),
    'NUM_UC':      (18, 19),
    'COD_PRODUTO': (25, 32),
}
CAMPOS_MORADOR = {
    'UF':                (0, 2),
    'TIPO_SITUACAO_REG': (6, 7),
    'COD_UPA':           (7, 16),
    'NUM_DOM':           (16, 18),
    'NUM_UC':            (18, 19),
    'COD_INFORMANTE':    (19, 21),
    'IDADE':             (32, 35),
    'SEXO':              (35, 36),
    'COR_RACA':          (36, 37),
    'TRABALHO':          (38, 39),
    'RENDA_TOTAL':       (105, 115),
    'NIVEL_INSTRUCAO':   (115, 116),
}

# Ordem das categorias, títulos e rótulos acentuados das figuras
ORDEM_CAT = {
    'ACESSO_FINANCEIRO': ['Com acesso', 'Sem acesso'],
    'FAIXA_RENDA':       ['Ate 1 SM', '1 a 2 SM', '2 a 4 SM',
                          'Acima de 4 SM'],
    'NIVEL_INSTRUCAO':   ['Sem instrucao', 'Fundamental Incompleto',
                          'Fundamental', 'Medio', 'Superior'],
    'COR_RACA':          ['Branca', 'Não-branca', 'Sem declaracao'],
    'FAIXA_ETARIA':      ['Jovem', 'Adulto', 'Idoso'],
    'SEXO':              ['Homem', 'Mulher'],
    'TIPO_SITUACAO_REG': ['Urbano', 'Rural'],
    'REGIAO':            ['Norte', 'Nordeste', 'Sudeste', 'Sul',
                          'Centro-Oeste'],
    'TRABALHO':          ['Trabalhou', 'Nao trabalhou', 'Nao aplicavel'],
}
TITULOS = {
    'ACESSO_FINANCEIRO': 'Acesso financeiro',
    'FAIXA_RENDA':       'Faixa de renda',
    'NIVEL_INSTRUCAO':   'Nível de instrução',
    'COR_RACA':          'Cor ou raça',
    'FAIXA_ETARIA':      'Faixa etária',
    'SEXO':              'Sexo',
    'TIPO_SITUACAO_REG': 'Situação do domicílio',
    'REGIAO':            'Região',
    'TRABALHO':          'Trabalho',
}
ROTULOS = {
    'Ate 1 SM':               'Até 1 SM',
    'Sem instrucao':          'Sem instrução',
    'Fundamental Incompleto': 'Fundamental incompleto',
    'Medio':                  'Médio',
    'Sem declaracao':         'Sem declaração',
    'Nao trabalhou':          'Não trabalhou',
    'Nao aplicavel':          'Não aplicável',
}
FONTE = ('Fonte: elaboração própria a partir dos microdados '
         'da POF 2017-2018 (IBGE).')

plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'figure.facecolor': 'white',
                     'axes.facecolor': 'white',
                     'axes.titlesize': 12.5})


def fmt_pct(p):
    """Formata percentual com vírgula decimal (ex.: 44,4%)."""
    return f'{p:.1f}'.replace('.', ',') + '%'


def fmt_n(n):
    """Formata inteiro com ponto de milhar (ex.: 58.039)."""
    return f'{int(n):,}'.replace(',', '.')


def ler_largura_fixa(arquivo, campos):
    """Lê um arquivo de largura fixa da POF e devolve um DataFrame."""
    registros = []
    with open(arquivo, 'r', encoding='latin1') as f:
        for linha in f:
            registros.append({nome: linha[ini:fim].strip()
                              for nome, (ini, fim) in campos.items()})
    return pd.DataFrame(registros)


def fs(tamanho):
    """Escala o tamanho da fonte pelo valor atual de ESCALA."""
    return tamanho * ESCALA


def id_uc(tabela):
    """Identificador da unidade de consumo: UPA + domicílio + UC."""
    return tabela['COD_UPA'] + tabela['NUM_DOM'] + tabela['NUM_UC']


# ============================================================
# 1. LEITURA DOS MICRODADOS E VARIÁVEL DE ACESSO
# ============================================================
df_despesa = ler_largura_fixa(f'{CAMINHO}/DESPESA_INDIVIDUAL.txt',
                              CAMPOS_DESPESA)
df_despesa['ID_UC'] = id_uc(df_despesa)

# UC com acesso: registrou ao menos uma despesa com serviço financeiro
ucs_com_acesso = set(
    df_despesa.loc[df_despesa['COD_PRODUTO'].isin(CODIGOS_FINANCEIROS),
                   'ID_UC'])

df_morador = ler_largura_fixa(f'{CAMINHO}/MORADOR.txt', CAMPOS_MORADOR)
df_morador['ID_UC'] = id_uc(df_morador)

# Unidade de análise: pessoa de referência da UC (COD_INFORMANTE == 1)
df = df_morador[df_morador['COD_INFORMANTE'] == '1'].copy()
df['ACESSO_FINANCEIRO'] = (df['ID_UC'].isin(ucs_com_acesso)
                           .map({True: 'Com acesso', False: 'Sem acesso'}))

print(f'UCs com acesso financeiro: {fmt_n(len(ucs_com_acesso))}')
print(f'Pessoas de referência:     {fmt_n(len(df))}')


# ============================================================
# 2. PRÉ-PROCESSAMENTO
# ============================================================
df['SEXO'] = df['SEXO'].map({'1': 'Homem', '2': 'Mulher'})

# Cor ou raça dicotomizada: branca x demais categorias
df['COR_RACA'] = df['COR_RACA'].map({
    '1': 'Branca', '2': 'Não-branca', '3': 'Não-branca',
    '4': 'Não-branca', '5': 'Não-branca',
})

df['TIPO_SITUACAO_REG'] = df['TIPO_SITUACAO_REG'].map(
    {'1': 'Urbano', '2': 'Rural'})

df['TRABALHO'] = df['TRABALHO'].map(
    {'1': 'Trabalhou', '2': 'Nao trabalhou'})

# Nível de instrução agrupado pelo maior nível concluído
# (médio incompleto -> Fundamental; superior incompleto -> Medio)
df['NIVEL_INSTRUCAO'] = df['NIVEL_INSTRUCAO'].map({
    '1': 'Sem instrucao',
    '2': 'Fundamental Incompleto',
    '3': 'Fundamental', '4': 'Fundamental',
    '5': 'Medio', '6': 'Medio',
    '7': 'Superior',
})

# Região: o primeiro dígito do código da UF identifica a região
df['REGIAO'] = df['UF'].str[0].map({
    '1': 'Norte', '2': 'Nordeste', '3': 'Sudeste',
    '4': 'Sul', '5': 'Centro-Oeste',
})

df['IDADE'] = pd.to_numeric(df['IDADE'], errors='coerce')
df['FAIXA_ETARIA'] = pd.cut(df['IDADE'], bins=[0, 29, 59, 200],
                            labels=['Jovem', 'Adulto', 'Idoso'])

# Faixas de renda em salários mínimos (SM)
sm = SALARIO_MINIMO
df['RENDA_TOTAL'] = pd.to_numeric(df['RENDA_TOTAL'], errors='coerce')
df['FAIXA_RENDA'] = pd.cut(
    df['RENDA_TOTAL'],
    bins=[0, sm, 2 * sm, 4 * sm, float('inf')],
    labels=['Ate 1 SM', '1 a 2 SM', '2 a 4 SM', 'Acima de 4 SM'])

# Tratamento de valores ausentes:
# - cor ou raça fora do mapeamento -> 'Sem declaracao'
# - trabalho sem resposta          -> 'Nao aplicavel'
# - instrução sem resposta         -> 'Sem instrucao'
# - renda zero (fora do intervalo do pd.cut) ou nula -> 'Ate 1 SM'
df['COR_RACA'] = df['COR_RACA'].fillna('Sem declaracao')
df['TRABALHO'] = df['TRABALHO'].fillna('Nao aplicavel')
df['NIVEL_INSTRUCAO'] = df['NIVEL_INSTRUCAO'].fillna('Sem instrucao')
df['FAIXA_RENDA'] = df['FAIXA_RENDA'].fillna('Ate 1 SM')

COLUNAS_MODELO = ['REGIAO', 'TIPO_SITUACAO_REG', 'FAIXA_ETARIA', 'SEXO',
                  'COR_RACA', 'TRABALHO', 'NIVEL_INSTRUCAO', 'FAIXA_RENDA',
                  'ACESSO_FINANCEIRO']

print('\nValores ausentes nas variáveis do modelo:')
print(df[COLUNAS_MODELO].isnull().sum())

N = len(df)
TAXA_GERAL = (df['ACESSO_FINANCEIRO'] == 'Com acesso').mean() * 100


# ============================================================
# 3. CARACTERIZAÇÃO DESCRITIVA DA BASE (Seção 4.1)
# ============================================================
ESCALA = 1.7                      # maior valor = letras maiores

PAINEIS = ['ACESSO_FINANCEIRO', 'FAIXA_RENDA', 'NIVEL_INSTRUCAO',
           'COR_RACA', 'SEXO', 'FAIXA_ETARIA',
           'REGIAO', 'TIPO_SITUACAO_REG', 'TRABALHO']
TITULOS_PAINEL = {
    'ACESSO_FINANCEIRO': 'Acesso financeiro',
    'FAIXA_RENDA':       'Faixa de renda',
    'NIVEL_INSTRUCAO':   'Nivel de instrucao',
    'COR_RACA':          'Cor ou raca',
    'SEXO':              'Sexo',
    'FAIXA_ETARIA':      'Faixa etaria',
    'REGIAO':            'Regiao',
    'TIPO_SITUACAO_REG': 'Situacao do domicilio',
    'TRABALHO':          'Trabalho',
}

print('\n' + '=' * 55)
print(f'TOTAL DE UNIDADES DE CONSUMO: {N}')
vc_acesso = df['ACESSO_FINANCEIRO'].value_counts()
for cat in ['Com acesso', 'Sem acesso']:
    n = int(vc_acesso.get(cat, 0))
    print(f'  {cat}: {n} ({100 * n / N:.1f}%)')

# Renda zero e renda nula: ambas foram alocadas em 'Ate 1 SM'
zeros = int((df['RENDA_TOTAL'] == 0).sum())
nulos = int(df['RENDA_TOTAL'].isna().sum())
print('-' * 55)
print("RENDA ALOCADA EM 'Ate 1 SM' SEM VALOR POSITIVO:")
print(f'  renda == 0 (sem renda própria): {zeros} ({100 * zeros / N:.1f}%)')
print(f'  renda nula (falha de leitura):  {nulos} ({100 * nulos / N:.1f}%)')
print('=' * 55)


def categorias_ordenadas(var):
    """Categorias na ordem definida, seguidas de eventuais extras."""
    vc = df[var].value_counts()
    ordem = ORDEM_CAT[var]
    cats = ([c for c in ordem if c in vc.index]
            + [c for c in vc.index if c not in ordem])
    return cats, [int(vc[c]) for c in cats]


# Tabela-resumo: frequência de cada categoria
linhas = []
for var in PAINEIS:
    cats, vals = categorias_ordenadas(var)
    for cat, n in zip(cats, vals):
        linhas.append({'Variavel': TITULOS_PAINEL[var], 'Categoria': cat,
                       'n': n, '%': round(100 * n / N, 1)})
resumo = pd.DataFrame(linhas)
resumo.to_csv(f'{SAIDA}/panorama_variaveis.csv',
              index=False, encoding='utf-8-sig')
print(resumo.to_string(index=False))

# Figura: grade 3x3 com a distribuição de cada variável
fig, axes = plt.subplots(3, 3, figsize=(16, 15))
for ax, var in zip(axes.flat, PAINEIS):
    cats, vals = categorias_ordenadas(var)
    barras = ax.bar(range(len(cats)), vals, color='#4C72B0',
                    edgecolor='white')
    ax.set_title(TITULOS_PAINEL[var], fontsize=fs(12), fontweight='bold')
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=30, ha='right', fontsize=fs(9))
    ax.tick_params(axis='y', labelsize=fs(9))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt_n(x)))
    ax.set_ylim(0, max(vals) * 1.15)     # folga para os rótulos no topo
    ax.spines[['top', 'right']].set_visible(False)
    for b, v in zip(barras, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt_pct(100 * v / N),
                ha='center', va='bottom', fontsize=fs(8))

fig.suptitle(f'Panorama das variaveis — {N} unidades de consumo '
             '(POF 2017-2018)', fontsize=fs(14), fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(f'{SAIDA}/panorama_variaveis.png', dpi=300,
            bbox_inches='tight')
plt.show()


# ============================================================
# 4. TAXA DE ACESSO POR CATEGORIA (Seção 4.2)
# ============================================================
ESCALA = 2.2                      # maior valor = letras maiores

VARS_UNIVARIADAS = ['FAIXA_RENDA', 'NIVEL_INSTRUCAO', 'COR_RACA',
                    'FAIXA_ETARIA', 'SEXO', 'REGIAO',
                    'TIPO_SITUACAO_REG', 'TRABALHO']
COR_COM = '#3D5A80'
COR_SEM = '#C9603F'

# Tabela: percentual com e sem acesso em cada categoria
linhas = []
for var in VARS_UNIVARIADAS:
    ct = pd.crosstab(df[var], df['ACESSO_FINANCEIRO'])
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    for c in [c for c in ORDEM_CAT[var] if c in ct.index]:
        linhas.append({
            'Variavel': TITULOS[var], 'Categoria': c,
            'n': int(ct.loc[c].sum()),
            'pct_com_acesso': round(float(pct.loc[c].get('Com acesso', 0)),
                                    1),
            'pct_sem_acesso': round(float(pct.loc[c].get('Sem acesso', 0)),
                                    1),
        })
taxa = pd.DataFrame(linhas)
taxa.to_csv(f'{SAIDA}/taxa_acesso_por_categoria.csv',
            index=False, encoding='utf-8-sig')

print(f'\nTAXA GERAL: {fmt_pct(TAXA_GERAL)} das {fmt_n(N)} UCs têm acesso\n')
print(taxa.to_string(index=False))


def plota_atributo(ax, var):
    """Barras horizontais empilhadas (com x sem acesso) de uma variável."""
    sub = taxa[taxa['Variavel'] == TITULOS[var]]
    cats = [ROTULOS.get(c, c) for c in sub['Categoria']]
    p_com = sub['pct_com_acesso'].tolist()
    p_sem = sub['pct_sem_acesso'].tolist()
    y = np.arange(len(cats))[::-1]

    ax.barh(y, p_com, color=COR_COM, edgecolor='white', height=0.66,
            zorder=3)
    ax.barh(y, p_sem, left=p_com, color=COR_SEM, edgecolor='white',
            height=0.66, zorder=3)
    for yi, pc, ps in zip(y, p_com, p_sem):
        if pc >= 5:   # rótulo "com acesso" à esquerda, dentro da barra
            ax.text(1.5, yi, fmt_pct(pc), va='center', ha='left',
                    fontsize=fs(9), color='white', fontweight='bold',
                    clip_on=False)
        if ps >= 5:   # rótulo "sem acesso" à direita, dentro da barra
            ax.text(98.5, yi, fmt_pct(ps), va='center', ha='right',
                    fontsize=fs(9), color='white', fontweight='bold',
                    clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=fs(10.5))
    ax.set_xlim(0, 100)
    for lado in ['top', 'right', 'bottom']:
        ax.spines[lado].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.tick_params(length=0)
    ax.set_xticks([])
    ax.set_title(TITULOS[var], fontweight='bold', loc='left', pad=8,
                 color='#1a1a1a', fontsize=fs(12))


GRUPOS_FIGURA = [
    ('fig1_renda_instrucao', ['FAIXA_RENDA', 'NIVEL_INSTRUCAO']),
    ('fig2_raca_sexo_idade', ['COR_RACA', 'SEXO', 'FAIXA_ETARIA']),
    ('fig3_regiao_dom_trab', ['REGIAO', 'TIPO_SITUACAO_REG', 'TRABALHO']),
]
legenda = [mpatches.Patch(color=COR_COM, label='Com acesso'),
           mpatches.Patch(color=COR_SEM, label='Sem acesso')]

for nome, grupo in GRUPOS_FIGURA:
    ncols = len(grupo)
    max_cats = max(len([c for c in ORDEM_CAT[v] if c in df[v].unique()])
                   for v in grupo)
    fig, axes = plt.subplots(1, ncols,
                             figsize=(8.0 * ncols, 1.1 * max_cats + 3.4))
    for ax, var in zip(np.atleast_1d(axes), grupo):
        plota_atributo(ax, var)
    fig.legend(handles=legenda, loc='upper center', ncol=2, frameon=False,
               fontsize=fs(10.5), bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, FONTE, fontsize=fs(8), color='#888',
             style='italic')
    fig.tight_layout(rect=[0, 0.06, 1, 0.90], w_pad=5.0)
    fig.savefig(f'{SAIDA}/{nome}.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================
# 5. REGRAS DE ASSOCIAÇÃO - APRIORI (Seção 4.3)
# ============================================================
# Matriz binária: uma coluna por par variável_categoria
df_binario = pd.get_dummies(df[COLUNAS_MODELO],
                            columns=COLUNAS_MODELO).astype(bool)
print(f'\nDimensões da matriz binária: {df_binario.shape}')

itemsets = apriori(df_binario, min_support=SUPORTE_MIN,
                   use_colnames=True, max_len=TAMANHO_MAX)
print(f'Suporte mínimo equivale a {fmt_n(SUPORTE_MIN * N)} UCs')
print(f'Itemsets frequentes: {len(itemsets)}')

regras = association_rules(itemsets, metric='confidence',
                           min_threshold=CONFIANCA_MIN)
print(f'Regras geradas: {len(regras)}')

# Mantém apenas regras cujo consequente é somente o acesso financeiro
CONSEQ_COM = frozenset({'ACESSO_FINANCEIRO_Com acesso'})
CONSEQ_SEM = frozenset({'ACESSO_FINANCEIRO_Sem acesso'})

regras_acesso = (regras[regras['consequents'].isin([CONSEQ_COM,
                                                    CONSEQ_SEM])]
                 .sort_values('lift', ascending=False)
                 .copy())
print(f'Regras com acesso financeiro como único consequente: '
      f'{len(regras_acesso)}')
print(regras_acesso['consequents'].value_counts())

COLUNAS_REGRA = ['antecedents', 'support', 'confidence', 'lift']
com_acesso = regras_acesso[regras_acesso['consequents'] == CONSEQ_COM]
sem_acesso = regras_acesso[regras_acesso['consequents'] == CONSEQ_SEM]

print('\n=== TOP 15 PERFIS COM ACESSO FINANCEIRO ===')
print(com_acesso[COLUNAS_REGRA].head(15).to_string())
print('\n=== TOP 15 PERFIS SEM ACESSO FINANCEIRO ===')
print(sem_acesso[COLUNAS_REGRA].head(15).to_string())

# Tabela completa de regras (apêndice), com itens em texto legível
tabela_regras = regras_acesso[['antecedents', 'consequents', 'support',
                               'confidence', 'lift']].copy()
for col in ['antecedents', 'consequents']:
    tabela_regras[col] = tabela_regras[col].apply(
        lambda itens: ', '.join(sorted(itens)))
tabela_regras.to_csv(f'{SAIDA}/regras_acesso_completas.csv',
                     index=False, encoding='utf-8-sig')
