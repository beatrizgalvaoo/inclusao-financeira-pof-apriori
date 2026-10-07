# Como executar o código do TCC

## Visão geral

O código foi desenvolvido e executado no **Google Colab**, um ambiente gratuito do Google que roda Python direto no navegador, sem instalação local. Ele lê os microdados da POF 2017-2018 (IBGE), gera as tabelas e figuras do Capítulo 4 e aplica o algoritmo Apriori.

O script está dividido em seis etapas, na mesma ordem dos resultados do trabalho:

| Etapa | Conteúdo | Seção do TCC |
| --- | --- | --- |
| 0. Configuração | Bibliotecas, caminho dos dados, parâmetros e funções auxiliares | — |
| 1. Leitura dos microdados | Leitura dos arquivos e definição da variável de acesso financeiro | Capítulo 3 |
| 2. Pré-processamento | Categorização das variáveis sociodemográficas e tratamento de ausentes | Capítulo 3 |
| 3. Caracterização descritiva | Panorama das variáveis da base | 4.1 |
| 4. Taxa de acesso por categoria | Percentual com e sem acesso em cada categoria | 4.2 |
| 5. Regras de associação | Matriz binária, Apriori e seleção das regras | 4.3 |

## Pré-requisitos

- **Conta Google**, para acessar o Google Colab e o Google Drive.
- **Arquivos deste repositório**: o script do código-fonte (`apendice_codigo_tcc.py`) e os dois arquivos de microdados da POF 2017-2018 usados na análise, `DESPESA_INDIVIDUAL.txt` e `MORADOR.txt`.
- **Bibliotecas Python**, nas versões usadas no trabalho: pandas 2.2.2, NumPy 2.0.2, mlxtend 0.23.4 e matplotlib.

Os microdados são arquivos de texto de largura fixa, com codificação latin1. Eles devem ser usados como foram baixados, sem abrir e salvar em outro programa, porque o código lê cada variável pela posição dos caracteres em cada linha.

## Preparar os arquivos no Google Drive

Crie uma pasta no seu Google Drive e coloque nela os dois arquivos de microdados disponibilizados neste repositório. No trabalho original, a pasta se chama `POF_TCC` e fica na raiz do "Meu Drive":

```
Meu Drive/
└── POF_TCC/
    ├── DESPESA_INDIVIDUAL.txt
    ├── MORADOR.txt
    └── resultados/        (criada automaticamente pelo código)
```

Os nomes dos arquivos devem ficar exatamente como acima, com letras maiúsculas e extensão `.txt`. A pasta `resultados` não precisa ser criada: o código cria essa pasta na primeira execução e salva nela as tabelas e figuras.

## Indicar o caminho na etapa de configuração

O caminho da pasta com os dados é a única linha que precisa ser alterada no código. Ela fica na **etapa 0 (Configuração)**, na variável `CAMINHO`:

```python
drive.mount('/content/drive')
CAMINHO = '/content/drive/MyDrive/POF_TCC'
SAIDA = f'{CAMINHO}/resultados'
```

No Colab, o "Meu Drive" sempre aparece como `/content/drive/MyDrive`. Por isso, o caminho é formado por esse início seguido das pastas até os arquivos:

| Onde estão os arquivos no Drive | Valor de `CAMINHO` |
| --- | --- |
| Meu Drive > POF_TCC | `'/content/drive/MyDrive/POF_TCC'` |
| Meu Drive > TCC > dados | `'/content/drive/MyDrive/TCC/dados'` |
| Meu Drive > Pesquisa POF | `'/content/drive/MyDrive/Pesquisa POF'` |

Para copiar o caminho exato, abra o painel **Arquivos** (ícone de pasta na barra lateral do Colab) depois de montar o Drive. Clique com o botão direito na pasta e escolha **Copiar caminho**.

## Executar o código no Colab

1. Acesse [colab.research.google.com](https://colab.research.google.com) e crie um novo notebook em **Arquivo > Novo notebook**.
2. Na primeira célula, instale a versão do mlxtend usada no trabalho e execute a célula:

   ```python
   !pip install mlxtend==0.23.4
   ```

3. Copie o conteúdo de `apendice_codigo_tcc.py` para as células seguintes. Uma célula por etapa (0 a 5) facilita localizar eventuais erros.
4. Altere a variável `CAMINHO` na etapa 0, conforme a seção anterior.
5. Execute tudo em **Ambiente de execução > Executar tudo**. As etapas precisam rodar na ordem, porque cada uma usa variáveis criadas nas anteriores.
6. Na etapa 0, o Colab pede permissão para acessar o Google Drive. Escolha a conta onde estão os arquivos e autorize. A mensagem `Mounted at /content/drive` confirma que deu certo.

A leitura dos microdados (etapa 1) é a parte mais demorada, porque os arquivos são grandes e lidos linha a linha. Aguarde a conclusão antes de interromper a execução.

## Arquivos gerados

Ao final da execução, a pasta `resultados` contém:

| Arquivo | Conteúdo | Seção |
| --- | --- | --- |
| `panorama_variaveis.csv` | Frequência de cada categoria das variáveis | 4.1 |
| `panorama_variaveis.png` | Figura com a distribuição das variáveis | 4.1 |
| `taxa_acesso_por_categoria.csv` | Percentual com e sem acesso por categoria | 4.2 |
| `fig1_renda_instrucao.png` | Acesso por renda e instrução | 4.2 |
| `fig2_raca_sexo_idade.png` | Acesso por cor ou raça, sexo e faixa etária | 4.2 |
| `fig3_regiao_dom_trab.png` | Acesso por região, situação do domicílio e trabalho | 4.2 |
| `regras_acesso_completas.csv` | Todas as regras com acesso financeiro como consequente | 4.3 |
