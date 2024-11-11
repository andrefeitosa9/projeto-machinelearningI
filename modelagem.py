import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, recall_score
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV



# Ignorar warnings de convergência
import warnings

warnings.filterwarnings("ignore")

def clf_metrics_com_return(modelo, X, y_true, label, plot_conf_mat=True, print_cr=True):
    
    if print_cr:
        print(f"\nMétricas de avaliação de {label}:\n")
    
    y_pred = modelo.predict(X)

    if plot_conf_mat:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax[0]) 
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize="all", ax=ax[1])
        plt.show()

    if print_cr:
        print(classification_report(y_true, y_pred))
    
    return classification_report(y_true, y_pred, output_dict=True)

df = pd.read_csv("dados\dados_tratados_thiago.csv")


colunas_categoricas = df.drop(columns=['status_emprestimo']).select_dtypes(include=['category', 'object']).columns.tolist()
colunas_numericas = df.drop(columns=['status_emprestimo']).select_dtypes(include=['number']).columns.tolist()
# Tirei negativado pq ele também é binário (0, 1)



# Configurando o OneHotEncoder no ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), colunas_categoricas),
        ('num', StandardScaler(), colunas_numericas)
    ],
    remainder='passthrough'  # Mantém as outras colunas sem mudanças
)

X = df.drop(columns=['status_emprestimo'], axis=1)
y = df['status_emprestimo']

X_teste, X_treino, y_teste, y_treino = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y, shuffle=True)


# Hiperparâmetros
params_lgbm = {'subsample': 0.7, 'scale_pos_weight': 10, 'num_leaves': 20, 'n_estimators': 50, 'max_depth': -1, 'learning_rate': 0.1, 'colsample_bytree': 0.8}
params_xgboost = {'scale_pos_weight': 10, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 1.0}
params_catboost = {'learning_rate': 0.05, 'l2_leaf_reg': 3, 'iterations': 200, 'depth': 4, 'border_count': 128}

# Pipelines configurados com hiperparâmetros
pipe_catboost_balanceado_tunado = Pipeline([
    ('preprocessor', preprocessor),
    ("catboost", CatBoostClassifier(random_state=42, verbose=0, **params_catboost))
])

pipe_xgboost_tunado = Pipeline([
    ('preprocessor', preprocessor),
    ("xgboost", XGBClassifier(random_state=42, **params_xgboost))
])

pipe_lgbm_tunado = Pipeline([
    ('preprocessor', preprocessor),
    ("lgbm", LGBMClassifier(random_state=42, **params_lgbm))
])

# Pipelines configurados com hiperparâmetros e SMOTE:

pipe_catboost_balanceado_tunado_smote = Pipeline([
    ("catboost", CatBoostClassifier(random_state=42, verbose=0, **params_catboost))
])

pipe_xgboost_tunado_smote = Pipeline([
    ("xgboost", XGBClassifier(random_state=42, **params_xgboost))
])

pipe_lgbm_tunado_smote = Pipeline([
    ("lgbm", LGBMClassifier(random_state=42, **params_lgbm))
])

# Dicionário de pipelines

dict_pipes_tunado= {
    "CatBoost balanceado tunado": pipe_catboost_balanceado_tunado,
    "XGBoost balanceado tunado": pipe_xgboost_tunado,
    "LGBM balanceado tunado": pipe_lgbm_tunado
}


dict_pipes_tunado_smote = {
    "CatBoost balanceado tunado SMOTE": pipe_catboost_balanceado_tunado_smote,
    "XGBoost balanceado tunado SMOTE": pipe_xgboost_tunado_smote,
    "LGBM balanceado tunado SMOTE": pipe_lgbm_tunado
}

# Mesma coisa usando CV e tunagem

# Configuração do StratifiedKFold
kf = StratifiedKFold(n_splits=5)

# Inicialização do dicionário de resultados
resultado_experimentos = {"estimador": [], "recall_treino": [], "recall_teste": []}

# Loop pelos pipelines
for nome_modelo, pipeline in dict_pipes_tunado.items():
    recall_treino_lista = []
    recall_teste_lista = []
    soma = 0
    
    # Realiza a validação cruzada estratificada
    for indice_treino, indice_valida in kf.split(X_treino, y_treino):
        X_treino_split, X_valida_split = X_treino.iloc[indice_treino], X_treino.iloc[indice_valida]
        y_treino_split, y_valida_split = y_treino.iloc[indice_treino], y_treino.iloc[indice_valida]
        
        # Treina o modelo
        pipeline.fit(X_treino_split, y_treino_split)
        
        # Avalia o modelo no treino
        y_pred_treino = pipeline.predict(X_treino_split)
        recall_treino = recall_score(y_treino_split, y_pred_treino, average='binary')
        recall_treino_lista.append(recall_treino)
        
        # Avalia o modelo no teste (validação)
        y_pred_valida = pipeline.predict(X_valida_split)
        recall_teste = recall_score(y_valida_split, y_pred_valida, average='binary')
        recall_teste_lista.append(recall_teste)
        soma += 1

        print(f"\nClassification Report do modelo {nome_modelo} no CV número {soma}\n")

        print(classification_report(y_valida_split, y_pred_valida))
    
    # Calcula a média dos recalls
    recall_treino_medio = np.mean(recall_treino_lista)
    recall_teste_medio = np.mean(recall_teste_lista)
    
    # Armazena os resultados
    resultado_experimentos["estimador"].append(nome_modelo)
    resultado_experimentos["recall_treino"].append(recall_treino_medio)
    resultado_experimentos["recall_teste"].append(recall_teste_medio)
    
    print(f'Treinamento do modelo {nome_modelo} finalizado')

# Resultado final em DataFrame
df_resultados = pd.DataFrame(resultado_experimentos)

# Em casos de underfit, calcula a diferença entre o recall de treino e teste
df_resultados["gap"] = (df_resultados["recall_treino"] - df_resultados["recall_teste"]).apply(lambda x: x if x > 0 else np.inf)

# Ordena os resultados pelo recall no teste
df_resultados = df_resultados.sort_values("recall_teste", ascending=False)

print(df_resultados)

