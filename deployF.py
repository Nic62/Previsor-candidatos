import joblib
import streamlit as st
import pandas as pd
import plotly.express as px

# Carregar o modelo
modelo = joblib.load("Modelo.pkl")

# Título da aplicação
st.title('Previsor de contratação')

# Entradas do usuário
st.header('Digite as informações abaixo:')

idade_usuario = st.number_input("Idade:", 0, 100, step=1)
genero_usuario = st.radio("Gênero de nascença:", ["Homem", "Mulher"], index=None)
education_level = st.selectbox("Nível de educação:", ["Bacharelado(EAD)", "Bacharelado(Presencial)", "Mestrado", "PHD"])
anos_experiencia = st.number_input("Anos de experiência na área:", 0, 50, step=1)
empresas_recorrentes = st.number_input("Anos de experiência em empresas recorrentes:", 0, 100, step=1)
distancia_empresa = st.number_input("Distância até a empresa (km):", 0.0, 100.0, step=0.1)
pontuacao_entrevista = st.number_input("Pontuação da entrevista:(de 0 a 100)", 0, 100, step=1)
pontuacao_skills = st.number_input("Pontuação de skills:(de 0 a 100)", 0, 100, step=1)
score_personalidade = st.number_input("Pontuação de personalidade:(de 0 a 100)", 0, 100, step=1)
estrategia_recrutamento = st.number_input("Estratégia de recrutamento:(de 1 a 3)", 0, 3, step=1)

# Botão para realizar a previsão
botao = st.button("Previsão")

if botao:
    try:
        # Mapeamento de educação e gênero
        education_level_map = {
            'Bacharelado(EAD)': 1,
            'Bacharelado(Presencial)': 2,
            'Mestrado': 3,
            'PHD': 4
        }
        education_level_num = education_level_map[education_level]
        genero_usuario_num = 1 if genero_usuario == 'Homem' else 0

        # Armazenar as variáveis de entrada em um DataFrame
        df_previsao = pd.DataFrame({
            'Idade': [idade_usuario],
            'Gênero': [genero_usuario_num],
            'EducationLevel': [education_level_num],
            'Anos de experiência': [anos_experiencia],
            'Empresas recorrentes': [empresas_recorrentes],
            'Distância empresa': [distancia_empresa],
            'pontuação entrevista': [pontuacao_entrevista],
            'pontuação skills': [pontuacao_skills],
            'Score personalidade': [score_personalidade],
            'Estratégia de Recrutamento': [estrategia_recrutamento]
        })

        # Fazer a previsão
        predicao = modelo.predict(df_previsao)
        probabilidade = modelo.predict_proba(df_previsao)

        # Mostrar a previsão e a probabilidade
        resultado = 'Contratado' if predicao >= 0.5  else 'Não Contratado'
        porcentagem_contratacao = probabilidade[0][1] * 100
        st.write(f'Resultado da Previsão: {resultado}')
        st.write(f'Probabilidade de Contratação: {porcentagem_contratacao:.2f}%')

        # Criação de gráficos do Plotly
        with st.container():
            st.header("Gráficos das entradas")

            # Gráfico de barra para as variáveis numéricas
            df_melt = df_previsao.melt(var_name='Variável', value_name='Valor')
            fig = px.bar(df_melt, x='Variável', y='Valor', title='Valores das Variáveis de Entrada')
            st.plotly_chart(fig)

            # Gráfico de rosca para gênero
            genero_df = pd.DataFrame({
                'Gênero': ['Homem', 'Mulher'],
                'Contagem': [genero_usuario_num, 1 - genero_usuario_num]
            })
            fig_genero = px.pie(genero_df, values='Contagem', names='Gênero', hole=0.5, title='Distribuição de Gênero')
            st.plotly_chart(fig_genero)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
