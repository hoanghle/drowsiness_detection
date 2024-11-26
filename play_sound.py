from pygame import mixer
import time

def play_alert(file_path, duration):
    mixer.init()
    mixer.music.load(file_path)
    mixer.music.play()

    time.sleep(duration)
    mixer.music.stop()
    
    