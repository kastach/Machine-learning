import pandas as pd
import matplotlib.pyplot as plt
import librosa
import os
import pygame

#reading csv file
df = pd.read_csv('Data/features_3_sec.csv')
#print(df.head(-1))

#about the dataset
print(df.shape)
#df = df.drop(labels='filename', axis=1)
print(df.dtypes)

#understanding the audio files based on example of chosen file
audio_recording = 'Data/genres_original/hiphop/hiphop.00004.wav'
data, sr = librosa.load(audio_recording)
librosa.load(audio_recording, sr=45600)
#print(type(data), type(sr))

pygame.init()
pygame.mixer.music.load('Data/genres_original/hiphop/hiphop.00004.wav')
pygame.mixer.music.play()

# while pygame.mixer.music.get_busy():
#     pygame.event.wait()
# # if pygame.mixer.music.stop():
# #     pygame.event.clear()
# pygame.mixer.music.set_endevent()

# VISUALIZING AUDIO FILES
# plot raw wave files
plt.figure(figsize=(9,5))
librosa.display.waveshow(data, color = "seagreen", axis="s")
plt.title("Raw wave of file: " + audio_recording)
plt.ylabel("Amplitude")
plt.waitforbuttonpress()
plt.close()

# zero crossing rate
start = 1000
end = 1200
plt.figure(figsize=(9,5))
plt.plot(data[start:end], color="mediumorchid")
plt.title("Zero crossing rate of file: " + audio_recording)
plt.xlabel("Time[s]")
plt.grid()
plt.show()


zero_cross_rate = librosa.zero_crossings(data[start:end], pad=False)
print("The number od zero-crossing is:", sum(zero_cross_rate))
