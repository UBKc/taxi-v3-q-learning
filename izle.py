import time
import numpy as np
import gymnasium as gym

# egitilen ajani konsolda adim adim izlemek icin
q = np.load("q_tablosu.npy")
env = gym.make("Taxi-v3", render_mode="ansi")

durum, _ = env.reset()
bitti = False
adim = 0
while not bitti:
    print(env.render())
    aksiyon = int(np.argmax(q[durum]))
    durum, odul, sonlandi, kesildi, _ = env.step(aksiyon)
    bitti = sonlandi or kesildi
    adim += 1
    time.sleep(0.4)
print(env.render())
print(f"{adim} adimda bitti")
