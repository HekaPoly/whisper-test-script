import time
import sounddevice as sd
import numpy as np
import whisper
import io

class WhisperStream:
    audio_buffer = []

    @staticmethod
    def audio_callback(indata: np.ndarray, frames: int, time, status):
        """
        Callback function to handle incoming audio data.

        Parameters:
        - indata (np.ndarray): Input audio data.
        - frames (int): Number of frames.
        - time: Time.
        - status: Status of the audio stream.
        """
        if status:
            print(status, flush=True)
        audio_data = indata[:, 0]
        WhisperStream.audio_buffer.append(audio_data.copy())

    @staticmethod
    def process_audio(seconds: float = float('inf'), model_type: str = "base") -> io.StringIO:
        """
        Process audio data using the Whisper model.

        Parameters:
        - seconds (float): Duration in seconds to process audio (default: infinite).
        - model_type (str): Type of Whisper model to use (default: "base").

        Returns:
        - io.StringIO: Text stream containing transcribed text.
        """
        text_stream = io.StringIO()
        start_time = time.time()
        model = whisper.load_model(model_type)

        try:
            with sd.InputStream(
                callback=WhisperStream.audio_callback, 
                channels=1, 
                samplerate=16000, 
                dtype='float32'
            ):
                while time.time() - start_time < seconds:
                    if len(WhisperStream.audio_buffer) > 0:
                        audio_data = np.concatenate(WhisperStream.audio_buffer, axis=0)
                        WhisperStream.audio_buffer.clear()
                        result = model.transcribe(audio_data)

                        print(result['text'], flush=True)
                        text_stream.write(result['text'])
                    sd.sleep(1000)
        except KeyboardInterrupt as e:
            print("KeyboardInterrupt was triggered")
        finally:
            text_stream.seek(0)
            print(text_stream.read())

        return text_stream
    
if __name__ == "__main__":
    ws = WhisperStream()

    text = ws.process_audio(seconds=15)  # Change the duration as needed
    print(text.read)