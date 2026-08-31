# SpikeAI - Nivelamento

Projeto de nivelamento para treino da diretoria de Visão Computacional da Cortechx, servindo como preparação para o projeto oficial. Consiste no processamento de vídeo com estimativa de pose humana usando [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) e OpenCV.

O script processa um vídeo de entrada, detecta os landmarks (pontos-chave) do corpo em cada frame, e gera:

- Um **vídeo de saída** com fundo preto e os landmarks desenhados sobre a pessoa detectada.
- Um **arquivo CSV** com as coordenadas (x, y, z), visibilidade e presença de cada landmark, frame a frame.
- Um **resumo do processamento** impresso no console (resolução, FPS, taxa de detecção, tempo de processamento, etc).

## Estrutura do projeto

```
├── input/              # Vídeos de entrada
│   └── ataque.mp4
├── model/              # Modelo do MediaPipe Pose Landmarker
│   └── pose_landmarker_heavy.task
├── output/             # Resultados gerados
│   ├── output_video.mp4
│   └── landmarks.csv
├── src/
│   ├── main.py          # Ponto de entrada: lê o vídeo, roda a detecção e gera as saídas
│   ├── pose.py           # Desenha os landmarks e conexões sobre a imagem
│   ├── video.py          # Aplica o fundo preto usando a máscara de segmentação
│   └── landmark2csv.py   # Monta e escreve os dados dos landmarks em CSV
└── requirements.txt
```

## Requisitos

- Python 3.10+
- Dependências listadas em [requirements.txt](requirements.txt) (principais: `mediapipe`, `opencv-python`, `numpy`)

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

1. Coloque o vídeo de entrada em `input/` (por padrão o script espera `input/ataque.mp4`).
2. Garanta que o modelo `pose_landmarker_heavy.task` esteja em `model/`.
3. Execute:

```bash
python src/main.py
```

Os resultados serão salvos em:

- `output/output_video.mp4` — vídeo com fundo preto e landmarks desenhados.
- `output/landmarks.csv` — dados de cada landmark por frame (`frame`, `landmark_id`, `landmark_name`, `x`, `y`, `z`, `visibility`, `presence`, `detected`).

> Os caminhos de entrada/saída e as opções do modelo (confiança mínima de detecção/tracking, número de poses) estão definidos no início de [src/main.py](src/main.py) e podem ser ajustados conforme necessário.
