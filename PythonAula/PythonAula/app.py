from flask import Flask, render_template, request
import joblib
import random
import pandas as pd

# Instancia o app Flask
app = Flask(__name__)

# Carregar modelo de machine learning e os encoders utilizados para transformar os dados categóricos em numéricos
model = joblib.load("modelo_treino.joblib")
sexo_encoder = joblib.load("sexo_encoder.joblib")
meta_encoder = joblib.load("meta_encoder.joblib")
treino_encoder = joblib.load("treino_encoder.joblib")

# Ler arquivo CSV contendo os exercícios e os respectivos grupos musculares
df_exercicios = pd.read_csv("listaExercicios.csv")

# Criar um dicionário no formato {exercicio: grupo_muscular}
exercicio_musculo = dict(zip(df_exercicios['exercicio'], df_exercicios['musculo']))

# Rota principal que renderiza a página inicial
@app.route("/")
def home():
    return render_template("home.html")

# Rota para a funcionalidade de sugestão de treino com IA
@app.route("/Treino_Com_IA", methods=["GET", "POST"])
def Treino_Com_IA():
    # Variáveis iniciais
    resultado = None
    imc = None
    treino = None
    tabela_exercicios = None

    # Se o formulário for enviado (método POST)
    if request.method == "POST":
        try:
            # Captura e trata os dados do formulário
            nome = request.form["nome"].strip().capitalize()
            sexo = request.form["sexo"].strip().upper()
            idade = int(request.form["idade"])
            peso = float(request.form["peso"])
            altura = float(request.form["altura"])
            meta = request.form["meta"].strip().lower()

            # Validação do campo sexo
            if sexo not in ['H', 'M']:
                raise ValueError("Sexo deve ser H (HOMEM) ou M (MULHER).")

            # Cálculo do IMC
            imc = peso / (altura ** 2)

            # Regras fixas de decisão baseadas no IMC
            if imc > 30:
                treino = "emagrecimento"
                resultado = f"{nome}, seu treino sugerido é: emagrecimento (IMC: {imc:.2f})"
            elif imc < 18:
                treino = "hipertrofia"
                resultado = f"{nome}, seu treino sugerido é: hipertrofia (IMC: {imc:.2f})"
            else:
                # Codifica os dados de entrada para o modelo preditivo
                sexo_cod = sexo_encoder.transform([sexo])[0]
                meta_cod = meta_encoder.transform([meta])[0]

                # Monta a entrada do modelo com os dados tratados
                entrada = [[sexo_cod, idade, peso, altura, imc, meta_cod]]

                # Realiza a predição com o modelo treinado
                pred = model.predict(entrada)

                # Decodifica a predição para um valor compreensível
                treino = treino_encoder.inverse_transform(pred)[0]
                resultado = f"{nome}, seu treino sugerido é: {treino} (IMC: {imc:.2f})"

            # Se o usuário também escolheu quantidade de exercícios e grupos musculares
            if "quantidade" in request.form and "opcoes" in request.form:
                quantidade = int(request.form["quantidade"])
                grupos_musculares = request.form.getlist("opcoes")

                # Garante que cardio será incluído se o treino for emagrecimento
                if treino == "emagrecimento" and "cardio" not in grupos_musculares:
                    grupos_musculares.append("cardio")

                # Dicionário para armazenar os exercícios sorteados por grupo muscular
                exercicios_por_grupo = {}
                for grupo in grupos_musculares:
                    # Filtra os exercícios do grupo atual
                    opcoes = [ex for ex, musculo in exercicio_musculo.items() if musculo == grupo]
                    # Sorteia os exercícios, respeitando a quantidade máxima disponível
                    escolhidos = random.sample(opcoes, k=min(quantidade, len(opcoes)))
                    exercicios_por_grupo[grupo] = escolhidos

                # Converte o dicionário para DataFrame e depois para HTML (tabela)
                df_final = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in exercicios_por_grupo.items()]))
                tabela_exercicios = df_final.to_html(classes="table table-bordered", index=False)

        # Tratamento de erros para qualquer exceção que ocorra
        except Exception as e:
            resultado = f"Erro: {e}"

    # Renderiza a página com o resultado, treino e a tabela de exercícios (se houver)
    return render_template("index.html", resultado=resultado, treino=treino, tabela_exercicios=tabela_exercicios)

# Executa a aplicação Flask em modo debug
if __name__ == "__main__":
    app.run(debug=True)
