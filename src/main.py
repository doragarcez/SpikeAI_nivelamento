import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pose import draw_landmarks_on_image
from video import apply_black_background
from landmark2csv import append_landmarks, write_csv_file

model_path = 'model/pose_landmarker_heavy.task'

# Configuração do MediaPipe
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configuração do MediaPipe PoseLandmarker
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    output_segmentation_masks=True,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Input/Output 
input_video_path = 'input/ataque.mp4'
output_video_path = 'output/output_video.mp4'
output_csv_path = 'output/landmarks.csv'

# Abre o video de entrada
cap = cv2.VideoCapture(input_video_path)

# Coleta as informações do vídeo
widht = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define o codec e cria o objeto VideoWriter para salvar o vídeo de saída
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (widht, height))

# Inicializando variaveis para armazenar os dados dos landmarks
frame_number = 0
landmark_csv = []

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        # Converte o frame do BGR para o RGB (Explicação para eu não esquecer: OpenCV usa BGR por padrão, mas o MediaPipe espera imagens em RGB)    
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Conversão do frame para o formato de imagem do MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Processamento do frame
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Se houver landmarks detectados, adiciona os dados ao CSV
        if pose_landmarker_result.pose_landmarks:
            append_landmarks(pose_landmarker_result.pose_landmarks[0], frame_number, landmark_csv)

        # Aplica o fundo preto e desenha os landmarks no frame
        rgb_with_black_background = apply_black_background(rgb, pose_landmarker_result)
        annotated_rgb = draw_landmarks_on_image(rgb_with_black_background, pose_landmarker_result)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Salva o frame processado no vídeo de saída
        out.write(annotated_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

cap.release()
out.release()
cv2.destroyAllWindows()

write_csv_file(landmark_csv, output_csv_path)

print("Processamento concluído. O vídeo de saída foi salvo em:", output_video_path)
print("Os dados dos landmarks foram salvos em:", output_csv_path)
    
