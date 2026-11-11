import numpy as np
import gymnasium as gym
import logging

logging.basicConfig(level=logging.INFO)

# egitilen q tablosunu rastgelelik olmadan (sadece en iyi aksiyon) 1000 yeni bolumde deniyorum
q = np.load("q_tablosu.npy")
env = gym.make("Taxi-v3")

test_sayisi = 1000
oduller = []
adimlar = []
basarili = 0

for i in range(test_sayisi):
    # egitimde 0-9999 arasi seed kullanildi, testte farkli baslangiclar olsun
    durum, _ = env.reset(seed=100000 + i)
    toplam_odul = 0
    adim = 0
    bitti = False
    while not bitti:
        aksiyon = int(np.argmax(q[durum]))
        durum, odul, sonlandi, kesildi, _ = env.step(aksiyon)
        bitti = sonlandi or kesildi
        toplam_odul += odul
        adim += 1
    if sonlandi:
        basarili += 1
    oduller.append(toplam_odul)
    adimlar.append(adim)

logging.info(f"basari orani: %{100 * basarili / test_sayisi:.1f}")
logging.info(f"ortalama odul: {np.mean(oduller):.2f}")
logging.info(f"ortalama adim: {np.mean(adimlar):.2f}")

with open("sonuclar/test_sonucu.txt", "w") as f:
    f.write(f"test bolumu: {test_sayisi}\n")
    f.write(f"basari orani: %{100 * basarili / test_sayisi:.1f}\n")
    f.write(f"ortalama odul: {np.mean(oduller):.2f}\n")
    f.write(f"ortalama adim: {np.mean(adimlar):.2f}\n")
