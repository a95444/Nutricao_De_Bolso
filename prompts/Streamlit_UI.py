import streamlit as st
from gemma_nutri_RAG import NutritionAssistant

# Inicialização do assistente em session_state
if "assistant" not in st.session_state:
    st.session_state.assistant = NutritionAssistant("gemma_nutri_v4:latest")
    st.session_state.history = []
    st.session_state.profile_complete = False

assistant = st.session_state.assistant

# Configuração da página
st.set_page_config(page_title="Gemma Nutri RAG", layout="wide")
st.title("💚 Gemma Nutri RAG")

# Container de chat com altura fixa e scroll
chat_style = """
<style>
.chat-container {
  height: 70vh;
  overflow-y: auto;
  padding: 10px;
  border: 1px solid #ddd;
  background-color: #f9f9f9;
}
</style>
"""
st.markdown(chat_style, unsafe_allow_html=True)
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Renderização das mensagens
for role, msg in st.session_state.history:
    if role == "Tu":
        st.markdown(f"**Tu:** {msg}")
    else:
        st.markdown(f"**Assistente:** {msg}")

st.markdown("</div>", unsafe_allow_html=True)

# Lógica de controle do questionário
if not st.session_state.profile_complete:
    current_field = assistant.next_field()
    prompt = f"Por favor, indica o teu(a) **{current_field}**:" if current_field else "Escreve a tua pergunta:"
else:
    prompt = "Escreve a tua pergunta:"

user_msg = st.chat_input(prompt, key="chat_input")

if user_msg:
    st.session_state.history.append(("Tu", user_msg))

    if not st.session_state.profile_complete:
        with st.spinner("A guardar informação..."):
            # Obter o campo atual antes de atualizar
            current_field = assistant.next_field()

            # Chamar método corretamente com ambos argumentos
            assistant.update_profile_field(current_field, user_msg)

            # Verificar conclusão usando next_field()
            st.session_state.profile_complete = (assistant.next_field() is None)

            confirmation = f"✅ {current_field} guardado(a)!"
            st.session_state.history.append(("Assistente", confirmation))

            st.rerun()
    else:
        with st.spinner("A pensar…"):
            resp = assistant.ask(user_msg)
            print(f"resposta: {resp}")
        st.session_state.history.append(("Assistente", resp))
        st.rerun()  # Força atualização imediata do chat

# Como correr:
# pip install streamlit pandas scikit-learn faiss-cpu ollama
# Executar: streamlit run streamlit_app.py