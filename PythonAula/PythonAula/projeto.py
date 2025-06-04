import pandas as pd
import numpy as np
import random

def calcular_imc(peso, altura):
    return peso / (altura ** 2)

def recomendar_treino(meta, imc):
    if meta == 'emagrecimento' and imc < 30:
        return 'hipertrofia'
    elif meta == 'emagrecimento':
        return 'emagrecimento'
    elif meta in ['ganhar massa muscular', 'hipertrofia']:
        return 'hipertrofia'
    elif meta == 'força':
        return 'força'
    elif meta == 'resistência':
        return 'resistência'
    elif meta in ['mobilidade', 'condicionamento', 'melhorar postura', 'saúde geral']:
        return random.choice(['resistência', 'força', 'hipertrofia'])
    else:
        return 'hipertrofia'

metas_possiveis = [
    'ganhar massa muscular', 'emagrecimento', 'hipertrofia', 'secar gordura', 'força',
    'condicionamento', 'melhorar postura', 'mobilidade', 'resistência', 'saúde geral'
]
sexos = ['H', 'M']

random.seed(42)
np.random.seed(42)

dados = []
for _ in range(400):
    sexo = random.choice(sexos)
    idade = random.randint(18, 75)
    altura = round(np.random.uniform(1.50, 2.00), 2)
    peso = round(np.random.uniform(45, 120), 1)
    meta = random.choice(metas_possiveis)
    imc = calcular_imc(peso, altura)
    treino = recomendar_treino(meta, imc)

    dados.append({
        'sexo': sexo,
        'idade': idade,
        'peso': peso,
        'altura': altura,
        'treino': treino,
        'meta': meta
    })

df = pd.DataFrame(dados)
df.to_csv("base_treinos.csv", index=False)
print("Arquivo CSV gerado com sucesso: base_treinos.csv")



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Carregar base de dados
df = pd.read_csv("base_treinos.csv")

# Codificar colunas categóricas
sexo_encoder = LabelEncoder()
meta_encoder = LabelEncoder()
treino_encoder = LabelEncoder()

df['sexo'] = sexo_encoder.fit_transform(df['sexo'])
df['meta'] = meta_encoder.fit_transform(df['meta'])
df['treino'] = treino_encoder.fit_transform(df['treino'])

# Adicionar coluna de IMC
df['imc'] = df['peso'] / (df['altura'] ** 2)


# Separar variáveis independentes e alvo
X = df[['sexo', 'idade', 'peso', 'altura', 'imc', 'meta']]
y = df['treino']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Avaliar o modelo
y_pred = model.predict(X_test)
print("\n🔍 Avaliação do modelo:")
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(
    y_test,
    y_pred,
    labels=list(range(len(treino_encoder.classes_))),
    target_names=treino_encoder.classes_
))

# Salvar modelo e encoders
joblib.dump(model, "modelo_treino.joblib")
joblib.dump(sexo_encoder, "sexo_encoder.joblib")
joblib.dump(meta_encoder, "meta_encoder.joblib")
joblib.dump(treino_encoder, "treino_encoder.joblib")
print("✅ Modelo e encoders salvos!")

import pandas as pd


exercicio_musculo = {
    "Agachamento": "perna", "Leg Press": "perna", "Cadeira Extensora": "perna", "Cadeira Flexora": "perna",
    "Cadeira Abdutora": "gluteo", "Cadeira Adutora": "gluteo", "Stiff": "gluteo", "Avanço": "perna",
    "Panturrilha em pé": "perna", "Panturrilha sentado": "perna", "Afundo com halteres": "gluteo",
    "Afundo com barra": "gluteo", "Stiff com halteres": "gluteo", "Stiff com barra": "gluteo",
    "Sumô com halteres": "perna", "Sumô com barra": "perna", "Agachamento smith": "perna",
    "Agachamento frontal": "perna", "Agachamento sumô": "perna", "Agachamento com halteres": "perna",
    "Glúteo na máquina": "gluteo", "Glúteo com caneleira": "gluteo", "Leg 45": "perna",
    
    "Supino reto": "peito", "Supino inclinado": "peito", "Supino declinado": "peito",
    "Crucifixo reto": "peito", "Crucifixo inclinado": "peito", "Voador": "peito",
    
    "Remada curvada": "costa", "Remada unilateral": "costa", "Remada baixa": "costa",
    "Puxada frente": "costa", "Puxada atrás": "costa", "Barra fixa": "costa",
    
    "Desenvolvimento com barra": "ombro", "Desenvolvimento com halteres": "ombro",
    "Desenvolvimento arnold": "ombro", "Elevação lateral": "ombro", "Elevação frontal": "ombro",
    
    "Rosca direta": "biceps", "Rosca alternada": "biceps", "Rosca scott": "biceps",
    "Rosca martelo": "biceps", "Rosca concentrada": "biceps",
    
    "Tríceps testa": "triceps", "Tríceps corda": "triceps", "Tríceps francês": "triceps",
    
    "Prancha abdominal": "abdomen", "Abdominal supra": "abdomen", "Abdominal infra": "abdomen",
    "Abdominal oblíquo": "abdomen", "Abdominal com carga": "abdomen",
    
    "Jumping jack": "cardio", "Corrida parada": "cardio", "Polichinelo": "cardio",
    "Ciclismo indoor": "cardio", "Step": "cardio", "Esteira": "cardio",
    
    "Glúteo kickback": "gluteo", "Ponte de glúteo": "gluteo", "Hip thrust": "gluteo",
    "Agachamento bulgaro": "gluteo", "Elevação de quadril": "gluteo",
    
    "Arnold press": "ombro", "Face pull": "ombro", "Press militar": "ombro",
    "Flexão de ombro": "ombro", "Hollow body hold": "abdomen",
    
    "Sprint": "cardio", "Escada ergométrica": "cardio", "Treino aeróbico": "cardio",
    "Burpee avançado": "cardio", "Battle ropes": "cardio",

    "Subida no banco com halteres":"gluteo",
    "Good morning": "gluteo",
    "Passada lateral": "perna",
    "Step-up com barra": "gluteo",
    "Agachamento com salto": "cardio",
    "Leg horizontal": "perna",
    "Panturrilha leg press": "perna",
    "Mesa flexora": "perna",
    "Mesa extensora": "perna",
    "Glúteo 4 apoios": "gluteo",
    
    "Cross over": "peito",
    "Flexão de braço": "peito",
    "Peck deck": "peito",
    "Flexão declinada": "peito",
    "Flexão inclinada": "peito",
    
    "Puxada aberta": "costa",
    "Remada cavalinho": "costa",
    "Pulldown com triangulo": "costa",
    "Puxada unilateral": "costa",
    "Remada na máquina": "costa",
    
    "Desenvolvimento máquina": "ombro",
    "Elevação lateral no cross": "ombro",
    "Elevação lateral inclinada": "ombro",
    "Remada alta": "ombro",
    "Crucifixo invertido": "ombro",
    
    "Rosca 21": "biceps",
    "Rosca na polia": "biceps",
    "Rosca spider": "biceps",
    "Rosca barra W": "biceps",
    "Rosca inversa": "biceps",
    
    "Tríceps banco": "triceps",
    "Tríceps pulley": "triceps",
    "Tríceps kickback": "triceps",
    "Tríceps mergulho": "triceps",
    "Tríceps coice": "triceps",
    
    "Abdominal prancha lateral": "abdomen",
    "Abdominal canivete": "abdomen",
    "Abdominal remador": "abdomen",
    "Abdominal bicicleta": "abdomen",
    "Abdominal na bola": "abdomen",
    
    "Mountain climber": "cardio",
    "Salto com joelhos ao peito": "cardio",
    "Pular corda": "cardio",
    "High knees": "cardio",
    "Shadow boxing": "cardio",
    
    "Corrida": "cardio",
    "Escalador cruzado": "cardio",
    "Corrida intervalada": "cardio",
    "Burpee simples": "cardio",
    "Air bike": "cardio"
}

df = pd.DataFrame(list(exercicio_musculo.items()), columns=["exercicio", "musculo"])
df.to_csv("listaExercicios.csv", index=False)

print("✅ listaExercicios.csv atualizado com sucesso!")