import os
import whisper
from deep_translator import GoogleTranslator

def transcribe_audio_to_vtt(audio_path, output_vtt):
    print("Usando dispositivo:", "cuda" if whisper.torch.cuda.is_available() else "cpu")

    # Carga el modelo Whisper (puedes cambiar a "base" o "large" según la GPU)
    model = whisper.load_model("small")

    print("Transcribiendo audio...")
    result = model.transcribe(audio_path, task="transcribe", language="en")

    segments = result["segments"]

    print("Traduciendo al español...")
    translator = GoogleTranslator(source="en", target="es")

    with open(output_vtt, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = seg["start"]
            end = seg["end"]
            text_en = seg["text"].strip()

            # Traducir texto al español
            try:
                text_es = translator.translate(text_en)
            except Exception as e:
                print("Error traduciendo:", e)
                text_es = text_en  # fallback

            # Formato de tiempo
            start_time = format_timestamp(start)
            end_time = format_timestamp(end)

            f.write(f"{start_time} --> {end_time}\n{text_es}\n\n")

    print(f"Archivo .vtt generado correctamente: {output_vtt}")


def format_timestamp(seconds):
    """Convierte segundos a formato hh:mm:ss.mmm"""
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


if __name__ == "__main__":
    audio_input = input("Ruta del archivo .wav: ").strip()
    output_vtt = os.path.splitext(audio_input)[0] + "_es.vtt"
    transcribe_audio_to_vtt(audio_input, output_vtt)
