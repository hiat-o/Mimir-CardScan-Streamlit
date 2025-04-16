import streamlit as st
from PIL import Image
import numpy as np
import time
import cv2
from streamlit_webrtc import webrtc_streamer
from av import VideoFrame

# Configuração da página
st.set_page_config(
    page_title="Mimir CardScan - Scanning Cards",
    page_icon="🔍",
    layout="wide"
)

# Função de callback para processar frames de vídeo (inicialmente apenas retorna o frame)
def video_frame_callback(frame: VideoFrame) -> VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # Lógica de pausa (exemplo: não processar se pausado)
    if st.session_state.get("scanning_paused", True):
        # Pode adicionar um overlay "Paused" ou apenas retornar
        pass # Por enquanto, retorna o frame normal mesmo pausado
    else:
        # Aqui viria o processamento real da imagem (detecção, etc.)
        # Ex: Adicionar um retângulo simples para visualização
        cv2.rectangle(img, (100, 100), (540, 380), (0, 255, 0), 2)

    return VideoFrame.from_ndarray(img, format="bgr24")

# Inicialização do estado da sessão (movido para antes do uso)
if "scanning_paused" not in st.session_state:
    st.session_state["scanning_paused"] = True

if "scanning_finished" not in st.session_state:
    st.session_state["scanning_finished"] = False
    
# Título principal
st.title("Scanning Cards")

# Layout de duas colunas
left_col, right_col = st.columns(2)

# ===== COLUNA ESQUERDA: CONTROLES E RESULTADOS =====
with left_col:
    # Informação do scan
    st.header("magic Scan")
    
    # Botões Pause/Resume em linha
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state["scanning_paused"] = True
            st.rerun()
    with col2:
        if st.button("▶️ Resume", use_container_width=True):
            st.session_state["scanning_paused"] = False
            st.rerun()
        
    # Botão de Finalizar
    if st.button("⊗ Finish", use_container_width=True, type="primary"):
        st.session_state["scanning_finished"] = True
        # Aqui você redirecionaria para a página de resultados em um aplicativo completo
        st.success("Scanning finished. Results are being processed...")
    
    # Seção de cartas detectadas
    st.markdown("---")  # Separador
    st.header("Detected Cards (0)")
    
    # Área para mostrar as cartas detectadas
    # Esta seria preenchida dinamicamente à medida que as cartas são detectadas
    # Exemplo estático para demonstração:
    if "detected_cards" not in st.session_state:
        st.session_state["detected_cards"] = []
    
    # Mostra cartas detectadas (nenhuma por enquanto)
    if not st.session_state["detected_cards"]:
        st.write("No cards detected yet. Position a card in front of the camera.")

# ===== COLUNA DIREITA: VISUALIZAÇÃO DA CÂMERA =====
with right_col:
    # Cabeçalho da câmera com status
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Camera")
    with col2:
        st.write("")  # Espaço para alinhar
        status = "Paused" if st.session_state.get("scanning_paused", True) else "Active"
        st.markdown(f"<div style='text-align: right'><h3>{status}</h3></div>", unsafe_allow_html=True)
        
    # Visualização da câmera usando webrtc_streamer
    webrtc_ctx = webrtc_streamer(
        key="scan-camera-stream",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={ 
            "video": {"width": {"ideal": 1920}, "height": {"ideal": 1080}},
            "audio": False 
        },
        async_processing=True,
    )