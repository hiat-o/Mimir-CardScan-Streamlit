import streamlit as st
from PIL import Image
import numpy as np
import time
import cv2
import os
from streamlit_webrtc import webrtc_streamer
from av import VideoFrame
from utils.Detect_and_crop_cards import HybridCardDetector
import threading
import random

# --- Configurações do Detector ---
# !!! AJUSTE ESTES VALORES PARA SEU AMBIENTE !!!
MODEL_PATH = 'data/models/detect_cards_model.pt' # <<< CAMINHO ATUALIZADO
OUTPUT_DIR = "cards_detected_streamlit" # <<< PASTA DE SAÍDA RELATIVA
CONF_THRESHOLD = 0.45
CROP_W = 480
CROP_H = int(CROP_W * 1.4) # ~672
GRID_SIZE_BLOCKS = 3
ILLUM_THRESH = 0.07
NEWCARD_THRESH = 0.14
VAR_THRESH = 0.010
REQUIRED_STABILITY = 5 # Frames de estabilidade
SAVE_COOLDOWN = 1.5 # Segundos entre salvamentos
# !!! FIM DOS AJUSTES !!!

# Cria diretório de saída se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuração da página
st.set_page_config(
    page_title="Mimir CardScan - Scanning Cards",
    page_icon="🔍",
    layout="wide"
)

# Função cacheada para carregar o detector (executa apenas uma vez)
@st.cache_resource
def load_detector():
    print("Carregando o modelo HybridCardDetector...")
    try:
        detector = HybridCardDetector(
            model_path=MODEL_PATH,
            output_dir=OUTPUT_DIR,
            conf_threshold=CONF_THRESHOLD,
            crop_width=CROP_W,
            crop_height=CROP_H,
            grid_size=GRID_SIZE_BLOCKS,
            illumination_threshold=ILLUM_THRESH,
            new_card_threshold=NEWCARD_THRESH,
            variance_threshold=VAR_THRESH,
            required_stability_frames=REQUIRED_STABILITY,
            save_cooldown=SAVE_COOLDOWN
        )
        print("Detector carregado com sucesso.")
        return detector
    except Exception as e:
        st.error(f"Erro ao carregar o modelo/detector: {e}")
        print(f"Erro fatal ao carregar detector: {e}")
        # Em um app real, talvez queira parar a execução ou mostrar uma mensagem mais clara
        return None

# Carrega o detector usando a função cacheada
detector = load_detector()
print(f"Estado inicial do detector: {detector is not None}")

# Variáveis globais para contornar o problema de acesso ao st.session_state entre threads
global_detector_state = None
global_prev_frame = None  # Inicializa como None
global_detected_cards = [] # Lista de caminhos das cartas detectadas

# Inicializa o estado do detector (variável global)
if detector is not None:
    try:
        print("Inicializando global_detector_state")
        global_detector_state = detector.initialize_state()
        print(f"global_detector_state inicializado: {global_detector_state is not None}")
    except Exception as e:
        print(f"ERRO ao inicializar global_detector_state: {e}")

# Mutex para acesso seguro às variáveis compartilhadas
# Isso garante thread-safety ao acessar essas variáveis do callback e da UI principal
state_lock = threading.Lock()

# Variáveis para sincronização de estado
scanning_active = False  # Valor padrão
last_pause_state = True  # Rastreia o último estado de pausa conhecido

# Função para pausar o escaneamento
def pause_scanning():
    global scanning_active, last_pause_state
    with state_lock:
        st.session_state["scanning_paused"] = True
        scanning_active = False
        last_pause_state = True
    print("📟 EVENTO: Escaneamento pausado! Estado pausado:", st.session_state["scanning_paused"])

# Função para retomar o escaneamento
def resume_scanning():
    global scanning_active, last_pause_state
    with state_lock:
        # Estado anterior para log
        old_state = st.session_state.get("scanning_paused", True)
        
        # Atualiza os estados
        st.session_state["scanning_paused"] = False
        scanning_active = True
        last_pause_state = False
        
        # Log detalhado
        print(f"📟 EVENTO: Escaneamento ATIVADO! Mudança de estado: {old_state} -> {st.session_state['scanning_paused']}")
        print(f"📟 Variáveis de controle: scanning_active={scanning_active}, last_pause_state={last_pause_state}")

# Função de callback baseada no exemplo simples da documentação
def video_frame_callback(frame):
    global global_detector_state, global_prev_frame, global_detected_cards, scanning_active, last_pause_state
    img = frame.to_ndarray(format="bgr24")
    
    # Acesso seguro ao estado de pausa
    with state_lock:
        # Verifica o session_state diretamente para o estado de pausa
        is_paused = st.session_state.get("scanning_paused", True)
        # Detecta mudanças no estado de pausa
        if is_paused != last_pause_state:
            scanning_active = not is_paused
            last_pause_state = is_paused
            print(f"📟 Callback detectou mudança de estado: is_paused={is_paused}, scanning_active={scanning_active}")
    
    # Debug: imprime o estado atual
    if random.random() < 0.01:  # Apenas 1% dos frames para não sobrecarregar os logs
        print(f"Frame processado: is_paused={is_paused}, scanning_active={scanning_active}")
    
    # Guarda uma cópia do frame original antes de qualquer modificação
    # Esta cópia será usada para o processamento interno, sem marcações visuais
    original_img = img.copy()
    
    # Tenta detectar cartas com o detector YOLO independente do estado de pausa
    if detector is not None:
        box, corners, score = detector.detect_card_yolo(img)
        
        # Adiciona texto com informações sobre a detecção - apenas para visualização
        cv2.putText(img, f"Score: {score:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Exibe o estado atual se disponível
        if global_detector_state is not None:
            current_state = global_detector_state.get("state", "DESCONHECIDO")
            cv2.putText(img, f"Estado: {current_state}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        # Se detectou com confiança suficiente, desenha os contornos - apenas para visualização
        if corners is not None and score >= 0.3:  # Threshold mais baixo para teste
            try:
                # Desenha um contorno verde ao redor da carta na imagem de visualização
                cv2.polylines(img, [corners.astype(int)], isClosed=True, 
                             color=(0, 255, 0), thickness=2)
            except Exception as e:
                error_msg = str(e)[:20]
                print(f"Erro ao desenhar contorno: {error_msg}")
                cv2.putText(img, f"Erro desenho: {error_msg}", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Processa o frame independente do estado - isso permite contar frames estáveis 
        # mesmo que estejamos pausados, só para visualização
        try:
            # Chama o método process_frame do detector para atualizar a máquina de estados
            # Passa o frame original (sem marcações) para processamento interno
            processed_frame, updated_state, saved_filepath = detector.process_frame(
                frame=original_img,  # Importante: usa o frame original sem marcações visuais
                prev_frame=global_prev_frame,
                current_state=global_detector_state
            )
            
            # Atualiza o estado global e o frame anterior (também sem marcações)
            global_detector_state = updated_state
            global_prev_frame = original_img.copy()
            
            # Se não está pausado E uma nova carta foi salva, adiciona à lista global
            if not is_paused and saved_filepath:
                if saved_filepath not in global_detected_cards:
                    global_detected_cards.append(saved_filepath)
                    # Atualiza a lista no st.session_state com proteção de thread
                    with state_lock:
                        st.session_state['detected_cards'] = global_detected_cards.copy()
                    
                    # Adiciona mensagens mais visíveis no frame quando uma carta é salva
                    cv2.putText(img, "CARTA SALVA!", (20, img.shape[0] - 60), 
                               cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
                    cv2.putText(img, os.path.basename(saved_filepath), (20, img.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    print(f"💾 Carta salva com sucesso: {saved_filepath}")
            
            # Se houver um frame processado (com visualizações), usa-o em vez do frame marcado manualmente
            if processed_frame is not None:
                img = processed_frame  # Use o frame processado para visualização
                
        except Exception as e:
            error_msg = str(e)[:30]
            print(f"❌ Erro ao processar frame: {error_msg}")
            cv2.putText(img, f"Erro proc: {error_msg}", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Retorna o frame modificado - isso é crucial para ver as mudanças
    return VideoFrame.from_ndarray(img, format="bgr24")

# Inicialização do estado da sessão
if "scanning_paused" not in st.session_state:
    st.session_state["scanning_paused"] = True
    print("📟 Inicializando scanning_paused como True")
if "scanning_finished" not in st.session_state:
    st.session_state["scanning_finished"] = False
if "detected_cards" not in st.session_state:
    st.session_state["detected_cards"] = global_detected_cards  # Inicializa com a lista global

# Sincroniza a variável global com o estado da sessão
with state_lock:
    scanning_active = not st.session_state.get("scanning_paused", True)
    last_pause_state = st.session_state.get("scanning_paused", True)

# DEBUG: Verificar estado final
print(f"Estado final: detector={detector is not None}, global_detector_state={global_detector_state is not None}, scanning_active={scanning_active}")

# Título principal
st.title("Scanning Cards")

# Layout de duas colunas
left_col, right_col = st.columns(2)

# ===== COLUNA ESQUERDA: CONTROLES E RESULTADOS =====
with left_col:
    # Informação do scan
    st.header("Magic Card Scan")
    
    # Status e instruções
    st.markdown("**📋 Instruções:**")
    st.markdown("1. Conecte a câmera")
    st.markdown("2. Clique em **Iniciar Escaneamento** para começar")
    st.markdown("3. Posicione uma carta Magic em frente à câmera")
    st.markdown("4. As cartas detectadas aparecerão abaixo")
    
    # Botão de Finalizar
    if st.button("⊗ Finalizar Escaneamento", use_container_width=True, type="primary"):
        st.session_state["scanning_finished"] = True
        # Aqui você redirecionaria para a página de resultados em um aplicativo completo
        st.success("Escaneamento finalizado. Os resultados estão sendo processados...")
    
    # Seção de cartas detectadas
    st.markdown("---")  # Separador
    # Atualiza contador no header
    st.header(f"Cartas Detectadas ({len(st.session_state.get('detected_cards', []))})")
    
    # Área para mostrar as cartas detectadas
    # Exemplo estático para demonstração:
    if "detected_cards" not in st.session_state:
        st.session_state["detected_cards"] = []

    # Mostra cartas detectadas (lógica será adicionada depois)
    if not st.session_state.get("detected_cards", []):
        st.write("Nenhuma carta detectada ainda. Posicione uma carta em frente à câmera.")
    else:
        # Itera sobre as cartas detectadas e as exibe
        for i, card_path in enumerate(reversed(st.session_state.get("detected_cards", []))):
            try:
                image = Image.open(card_path)
                st.image(image, caption=os.path.basename(card_path), width=150) # Mostra thumbnail

                # Adiciona botão de download
                with open(card_path, "rb") as file:
                    btn = st.download_button(
                        label=f"Baixar {os.path.basename(card_path)}",
                        data=file,
                        file_name=os.path.basename(card_path),
                        mime="image/jpeg",
                        key=f"download_{i}_{card_path}" # Chave única para cada botão
                    )
                st.markdown(" ---") # Separador entre cartas

            except FileNotFoundError:
                st.warning(f"Arquivo não encontrado: {os.path.basename(card_path)}. Pode ter sido removido.")
                # Opcional: remover o path da lista se não encontrado
                # st.session_state['detected_cards'].remove(card_path)
                # st.rerun() # Cuidado com rerun aqui também
            except Exception as e:
                st.error(f"Erro ao carregar/mostrar {os.path.basename(card_path)}: {e}")

# ===== COLUNA DIREITA: VISUALIZAÇÃO DA CÂMERA =====
with right_col:
    # Cabeçalho da câmera com status
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Camera")
    with col2:
        st.write("")  # Espaço para alinhar
        status = "Pausado" if st.session_state.get("scanning_paused", True) else "Ativo"
        st.markdown(f"<div style='text-align: right'><h3>{status}</h3></div>", unsafe_allow_html=True)
        
    # Visualização da câmera usando webrtc_streamer
    webrtc_ctx = webrtc_streamer(
        key="scan-camera-stream",
        video_frame_callback=video_frame_callback,  # Usando o callback simplificado
        media_stream_constraints={ 
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, # Reduzido para melhor performance web
            "audio": False 
        },
        async_processing=True,
        rtc_configuration={  # Configuração STUN necessária para conexões remotas
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
    
    # Atualiza o estado baseado no estado do WebRTC
    if webrtc_ctx.state.playing:
        # Adiciona botões diretamente sob o player de vídeo para melhor UX
        st.text("Controles de escaneamento:")
        pause_col, resume_col = st.columns(2)
        with pause_col:
            if st.button("⏸️ Pausar", use_container_width=True, key="pause_btn_camera"):
                pause_scanning()
                st.rerun()
        with resume_col:
            # Destaque visual para o botão de iniciar
            if st.button("▶️ INICIAR", use_container_width=True, type="primary", key="resume_btn_camera"):
                resume_scanning()
                st.rerun()
                
        # Mostra o estado atual claramente
        current_state = "🔴 PAUSADO" if st.session_state.get("scanning_paused", True) else "🟢 ATIVO"
        st.markdown(f"<h3 style='text-align: center'>{current_state}</h3>", unsafe_allow_html=True)
        
        # Mensagens informativas baseadas no estado
        if st.session_state.get("scanning_paused", True):
            st.info("Câmera conectada, mas escaneamento pausado. Clique em **INICIAR** para começar a detectar e salvar cartas.")
        else:
            st.success("✅ Escaneamento ATIVO! Posicione cartas em frente à câmera para detecção e salvamento automático.")
    else:
        # WebRTC não está rodando, força o estado para pausado
        if not st.session_state.get("scanning_paused", True):
            st.session_state["scanning_paused"] = True
            scanning_active = False
            st.rerun()  # Versão atualizada de st.experimental_rerun()
        st.warning("Câmera não conectada. Permita acesso à câmera no navegador.")