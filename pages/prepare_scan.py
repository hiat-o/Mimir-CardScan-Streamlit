import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer
from av import VideoFrame
import time

# Função de callback para processar frames de vídeo (inicialmente apenas retorna o frame)
def video_frame_callback(frame: VideoFrame) -> VideoFrame:
    # No futuro, aqui você pode adicionar lógica para detectar cartas, etc.
    img = frame.to_ndarray(format="bgr24")
    
    # Por enquanto, apenas retorna o frame como está
    return VideoFrame.from_ndarray(img, format="bgr24")

# Configuração da página
st.set_page_config(
    page_title="Mimir CardScan - Prepare Scan",
    page_icon="🔍",
    layout="wide"
)

# Título principal
st.title("Prepare Scan")

# Layout de duas colunas
col1, col2 = st.columns(2)

# Coluna 1: Scanner Setup
with col1:
    st.header("Scanner Setup")
    st.write("Select your card game and prepare the scanner.")
    
    # Seleção do jogo de cartas
    card_game = st.selectbox(
        "Card Game",
        ["Magic: The Gathering"]
    )
    
    # Indicadores de progresso
    st.write("")  # Espaço
    
    # Indicador de carregamento de modelos
    models_loaded = True  # Simular carregamento completo
    if models_loaded:
        st.markdown("✅ Models loaded")
        models_progress = st.progress(100)
    else:
        st.markdown("⏳ Loading models...")
        models_progress = st.progress(0)
    
    # Indicador de câmera pronta
    camera_ready = True  # Simular câmera pronta
    if camera_ready:
        st.markdown("✅ Camera ready")
        camera_progress = st.progress(100)
    else:
        st.markdown("⏳ Preparing camera...")
        camera_progress = st.progress(0)
    
    # Nome da digitalização
    scan_name = st.text_input("Scan Name", f"{card_game} Collection")
    
    # Dica para o usuário
    st.info("**Tip:** Make sure you have good lighting and position cards flat in front of the camera.")
    
    # Botão para iniciar digitalização
    start_button = st.button("Start Scanning", use_container_width=True)
    if start_button:
        st.switch_page("pages/scan.py")

# Coluna 2: Visualização da câmera
with col2:
    st.header("Camera Preview")
    
    # Seleção de câmera (simulado) - Pode ser removido ou adaptado se webrtc gerenciar a seleção
    # camera_options = ["Câmera FaceTime HD (1C1C:B782)", "Camera 2", "Camera 3"]
    # selected_camera = st.selectbox("Selecione a câmera:", camera_options)
    
    # Visualização da câmera com streamlit_webrtc
    webrtc_ctx = webrtc_streamer(
        key="camera-stream",
        video_frame_callback=video_frame_callback,
        # Solicitando resolução HD
        media_stream_constraints={ 
            "video": {"width": {"ideal": 1920}, "height": {"ideal": 1080}},
            "audio": False 
        },
        async_processing=True,
    )

# Se estiver digitalizando (após o clique no botão)
if 'scanning' in st.session_state and st.session_state['scanning']:
    # Aqui você adicionaria o código para processar a digitalização
    st.success("Scanning in progress... This will be implemented in the next phase.")
    # Reset para evitar loop infinito
    st.session_state['scanning'] = False