import whisper
import datetime

date = str(datetime.date.today())
model = whisper.load_model("base")
out = model.transcribe("HUxUbcoTB_4.m4a", language="english")

print(out['text'])

print("Start writing into text.txt")
with open("text.txt", "a")  as f : 
        f.write(date)
        f.write(out['text'])
        f.write('\n')
        f.close()

print("End of writing into text.txt")
