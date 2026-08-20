import csv

def append_landmarks(landmarks, frame_number, landmark_csv):
    for idx, landmark in enumerate(landmarks):
        landmark_csv.append([frame_number, idx, landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence])

def write_csv_file(landmark_csv, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(['frame', 'landmark_id', 'x', 'y', 'z', 'visibility', 'presence'])
        writer.writerows(landmark_csv)