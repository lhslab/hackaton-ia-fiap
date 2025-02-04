import io
import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import tempfile
import time
from dotenv import load_dotenv

# Configurações do Custom Vision
load_dotenv()

ENDPOINT = os.environ.get('ENDPOINT')
PREDICTION_KEY = os.environ.get('PREDICTION_KEY')
PROJECT_ID_DETECTION = os.environ.get('PROJECT_ID_DETECTION')
ITERATION_NAME = os.environ.get('ITERATION_NAME')
PREDICTION_URL = f"{ENDPOINT}customvision/v3.0/Prediction/{PROJECT_ID_DETECTION}/detect/iterations/{ITERATION_NAME}/image"
VERSAO_DO_APP = "v1.15 (Azure Custom Vision)"

# Função para enviar notificação WirePusher
def send_wirepusher_notification(device_id, title, message, category):
    url = "https://wirepusher.com/send"
    payload = {
        "id": device_id,
        "title": title,
        "message": message,
        "type": category
    }
    response = requests.get(url, params=payload)
    if response.status_code == 200:
        print("Notificação enviada com sucesso!")
    else:
        print(f"Erro ao enviar: {response.status_code} - {response.text}")

def get_prediction(image_data):
    headers = {"Prediction-Key": PREDICTION_KEY, "Content-Type": "application/octet-stream"}
    response = requests.post(PREDICTION_URL, headers=headers, data=image_data)
    if response.status_code != 200:
        st.error(f"Erro na API: {response.status_code} - {response.text}")
        return {"predictions": []}
    return response.json()

def draw_boxes_on_frame(frame, predictions, threshold=60):
    # Cria uma cópia da imagem para desenhar as caixas
    draw_image = frame.copy()
    draw = ImageDraw.Draw(draw_image)
    font_size = max(10, int(min(frame.width, frame.height) * 0.08))
    font = ImageFont.truetype("arial.ttf", size=font_size)

    for prediction in predictions:
        tag = prediction['tagName']
        probability = prediction['probability'] * 100
        if probability >= threshold:
            bbox = prediction['boundingBox']
            left, top = bbox['left'] * frame.width, bbox['top'] * frame.height
            right, bottom = (bbox['left'] + bbox['width']) * frame.width, (bbox['top'] + bbox['height']) * frame.height
            draw.rectangle([left, top, right, bottom], outline="red", width=2)
            text = f"{tag}: {probability:.2f}%"
            text_width, text_height = font.getbbox(text)[2], font.getbbox(text)[3] - font.getbbox(text)[1]
            draw.rectangle([left, top - text_height, left + text_width, top], fill="red")
            draw.text((left, top - text_height), text, fill="white", font=font)
    
    # Retorna a imagem com as caixas desenhadas como um array NumPy
    return np.array(draw_image)

def process_frame(frame_rgb, threshold):
    pil_image = Image.fromarray(frame_rgb)
    with io.BytesIO() as output:
        pil_image.save(output, format="JPEG")
        image_data = output.getvalue()
    
    prediction_result = get_prediction(image_data)
    predictions = prediction_result.get("predictions", [])

    detected = False  # Inicializa a variável antes do loop

    # Verifique se alguma previsão está acima do limiar
    for prediction in predictions:
        probability = prediction['probability'] * 100
        if probability >= threshold:
            detected = True
            break  # Sai do loop na primeira detecção válida

    # Se nenhuma detecção válida, exibe a imagem original com uma mensagem
    if not detected:  
        return frame_rgb, False  # Retorna a imagem original e indica que não houve detecção

    # Se há detecção válida, desenha as caixas e retorna a imagem processada
    return draw_boxes_on_frame(pil_image, predictions, threshold=threshold), True

st.set_page_config(
    page_title="FIAP VisionGuard",
    page_icon="👁️"
)

st.markdown("""
    <h1 style="text-align: center; margin-bottom: 0;">FIAP VisionGuard</h1>
    <h4 style="text-align: center; font-weight: normal; margin-top: 0;">Detecção de objetos cortantes</h4>
""", unsafe_allow_html=True)


st.markdown(
    f"""
    <div style="text-align: right; font-size: 12px; color: gray;">
        {VERSAO_DO_APP}
    </div>
    """,
    unsafe_allow_html=True
)

confidence_threshold = st.slider("Nível de confiança:", min_value=0, max_value=100, value=60, step=1)

# Checkbox para receber alerta no celular
receive_alerts = st.checkbox("Receber alerta no celular")

# Exibir o campo Device ID apenas se o checkbox estiver marcado
device_id = None
if receive_alerts:
    #device_id = st.text_input("Device ID do WirePusher (para receber notificações):", placeholder="Insira o Device ID", help="Este campo é obrigatório se deseja receber alertas.")
    col1, col2 = st.columns([5, 1])  # O input ocupa mais espaço que o botão

    with col1:
        device_id = st.text_input(
            "Device ID do WirePusher:",
            placeholder="Insira o Device ID",
            help="Este campo é obrigatório se deseja receber alertas."
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Ajuste fino para alinhamento vertical
        inserir_id = st.button("Inserir", use_container_width=True)  # Faz o botão ocupar toda a largura da coluna

    # Se o botão for pressionado e o Device ID for válido
    if inserir_id and device_id:
        st.session_state["device_id"] = device_id
        st.success(f"Device ID: **{device_id}**")

    if not device_id:
        st.stop()  # Interrompe a execução se o Device ID não for fornecido

uploaded_file = st.file_uploader("Escolha uma imagem (JPG, PNG) ou vídeo (MP4)", type=["jpg", "png", "mp4"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if file_extension in ["jpg", "png"]:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        processed_image, detected = process_frame(image_np, confidence_threshold)
        st.image(processed_image, caption="Imagem Processada", use_container_width=False)
        if detected and device_id:
            send_wirepusher_notification(
                device_id=device_id,
                title="Alerta Crítico",
                message="Objeto cortante detectado na imagem!",
                category="alerta"
            )
            st.success(f"Notificação enviada para o Device: **{device_id}**")
    
    elif file_extension == "mp4":
        if "cancel_processing" not in st.session_state:
            st.session_state.cancel_processing = False

        cancel_button = st.button("Cancelar Processamento")
        if cancel_button:
            st.session_state.cancel_processing = True

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        cap = cv2.VideoCapture(video_path)
        processed_frames = []
        detected_in_video = False  # Flag para indicar se houve detecção no vídeo
        frame_num = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total de frames do vídeo
        progress_bar = st.progress(0)  # Barra de progresso
        start_time = time.time()  # Tempo de início
        estimated_total_time = None  # Inicializa a variável para o tempo estimado total
        status_placeholder = st.empty()  # Placeholder para mostrar status atualizado

        while True:
            if st.session_state.cancel_processing:
                st.warning("Processamento cancelado pelo usuário.")
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, detected = process_frame(frame_rgb, confidence_threshold)
            processed_frames.append(processed_frame)

            if detected:
                detected_in_video = True  # Atualiza a variável caso haja detecção
            
            frame_num += 1
            progress = frame_num / total_frames
            progress_bar.progress(progress)
            
            elapsed_time = time.time() - start_time
            if frame_num > 0 and estimated_total_time is None:
                estimated_total_time = (elapsed_time / frame_num) * total_frames

            estimated_time_remaining = estimated_total_time - elapsed_time if estimated_total_time else 0
            elapsed_display = f"{int(elapsed_time // 60):02}:{int(elapsed_time % 60):02}"
            estimated_total_display = f"{int(estimated_total_time // 60):02}:{int(estimated_total_time % 60):02}" if estimated_total_time else "--:--"
            estimated_remaining_display = f"{int(estimated_time_remaining // 60):02}:{int(estimated_time_remaining % 60):02}" if estimated_time_remaining else "--:--"
            
            status_placeholder.markdown(
                f"**Quadros processados**: {frame_num}/{total_frames} ({progress * 100:.2f}%)\n\n"
                f"**Tempo decorrido**: {elapsed_display}\n\n"
                f"**Tempo estimado total**: {estimated_total_display}\n\n"
                f"**Tempo estimado restante**: {estimated_remaining_display}"
            )

        cap.release()
        os.makedirs("processed_videos", exist_ok=True)
        output_video_path = f"processed_videos/processed_video_{int(time.time())}.mp4"
        
        frame_height, frame_width, _ = processed_frames[0].shape
        #fourcc = cv2.VideoWriter_fourcc(*"H264")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))
        
        for frame in processed_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        
        with open(output_video_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)

        # Envia notificação ao final do processamento, se houve detecção e o Device ID foi fornecido
        if detected_in_video and device_id:
            send_wirepusher_notification(
                device_id=device_id,
                title="Alerta Crítico",
                message="Objeto cortante detectado no vídeo!",
                category="alerta"
            )
            st.success(f"Notificação enviada para o Device: **{device_id}**")
