import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

# 1. Importando o arquivo CSV com os dados médicos
df = pd.read_csv('medical_examination.csv')

# 2. Calculando a coluna 'overweight' com base no IMC (Índice de Massa Corporal)
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)

# 3. Convertendo as colunas 'cholesterol' e 'gluc' em valores binários
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Definindo a função para plotar o gráfico de categorias
def draw_cat_plot():
    # 5. Derretendo o DataFrame para facilitar a plotagem do gráfico
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Agrupando os dados por categoria, valor e contagem
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Plotando o gráfico de categorias
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig

    # 8. Salvando o gráfico em um arquivo
    fig.savefig('catplot.png')
    return fig

# 10. Definindo a função para plotar o mapa de calor
def draw_heat_map():
    # 11. Filtrando os dados com base em algumas condições
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculando a matriz de correlação
    corr = df_heat.corr()

    # 13. Criando uma máscara triangular superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Criando uma figura e um eixo para o mapa de calor
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Plotando o mapa de calor
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', ax=ax, cmap='coolwarm', center=0, square=True, linewidths=.5)

    # 16. Salvando o mapa de calor em um arquivo
    fig.savefig('heatmap.png')
    return fig
