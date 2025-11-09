# SubVtt 🎧 → 📜

Genera subtítulos `.vtt` en español a partir de archivos `.wav` en inglés, usando Whisper y Deep Translator.

## 🚀 Instalación

Clona el repositorio y crea un entorno virtual:

```bash
git clone git@github.com:EmmaSanchg/SubVtt.git
cd SubVtt
sudo apt install python3.12-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo apt install ffmpeg -y
