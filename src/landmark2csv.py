import csv

# Define a lista de nomes dos landmarks do corpo humano
POSE_LANDMARK_NAMES = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
    'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
    'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
    'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
    'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
    'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

def append_landmarks(landmarks, frame_number, landmark_csv):
    # Se os landmarks não forem detectados no frame, vai ser adicionado uma linha com None para cada Landmark
    if not landmarks:
        for idx, landmark_name in enumerate(POSE_LANDMARK_NAMES):
            landmark_csv.append([frame_number, idx, landmark_name, None, None, None, None, None, False])
        return

    # Se os landmarks forem detectados, adiciona cada landmark no CSV com suas coordenadas e informações de visibilidade e presença
    for idx, landmark in enumerate(landmarks):
        landmark_name = POSE_LANDMARK_NAMES[idx]
        landmark_csv.append([frame_number, idx, landmark_name, landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence, True])

# Função para escrever os dados dos landmarks em um arquivo CSV
def write_csv_file(landmark_csv, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(['frame', 'landmark_id', 'landmark_name', 'x', 'y', 'z', 'visibility', 'presence', 'detected'])
        writer.writerows(landmark_csv)