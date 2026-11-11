import numpy as np
import gymnasium as gym
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

# taxi-v3: 500 durum (taksi konumu x yolcu konumu x hedef), 6 aksiyon
# (guney, kuzey, dogu, bati, yolcu al, yolcu birak)
env = gym.make("Taxi-v3")

durum_sayisi = env.observation_space.n
aksiyon_sayisi = env.action_space.n
q = np.zeros((durum_sayisi, aksiyon_sayisi))

alfa = 0.1           # ogrenme orani
gama = 0.99          # gelecekteki odulun onemi
epsilon = 1.0        # basta tamamen rastgele geziyor
epsilon_min = 0.01
epsilon_azalma = 0.999
bolum_sayisi = 10000

np.random.seed(42)
env.action_space.seed(42)   # rastgele aksiyonlar da her calistirmada ayni olsun
oduller = []
adimlar = []

for bolum in range(bolum_sayisi):
    durum, _ = env.reset(seed=bolum)
    toplam_odul = 0
    adim = 0
    bitti = False

    while not bitti:
        # epsilon-greedy: bazen rastgele dene, cogunlukla en iyi bildigini yap
        if np.random.rand() < epsilon:
            aksiyon = env.action_space.sample()
        else:
            aksiyon = int(np.argmax(q[durum]))

        yeni_durum, odul, sonlandi, kesildi, _ = env.step(aksiyon)
        bitti = sonlandi or kesildi

        # q(s,a) <- q(s,a) + alfa * (odul + gama * max q(s',a') - q(s,a))
        # bolum sonlandiysa gelecek odul yok
        hedef = odul if sonlandi else odul + gama * np.max(q[yeni_durum])
        q[durum, aksiyon] += alfa * (hedef - q[durum, aksiyon])

        durum = yeni_durum
        toplam_odul += odul
        adim += 1

    epsilon = max(epsilon_min, epsilon * epsilon_azalma)
    oduller.append(toplam_odul)
    adimlar.append(adim)

    if (bolum + 1) % 1000 == 0:
        logging.info(f"bolum {bolum + 1}: son 1000 bolum ortalama odul {np.mean(oduller[-1000:]):.1f}, epsilon {epsilon:.3f}")

np.save("q_tablosu.npy", q)

# 100 bolumluk hareketli ortalama ile ogrenme egrisi
pencere = 100
ortalama = np.convolve(oduller, np.ones(pencere) / pencere, mode="valid")
plt.figure(figsize=(9, 4))
plt.plot(ortalama)
plt.xlabel("bolum")
plt.ylabel(f"toplam odul ({pencere} bolum ortalamasi)")
plt.title("Taxi-v3 Q-Learning ogrenme egrisi")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("sonuclar/ogrenme_egrisi.png", dpi=120)
logging.info("q_tablosu.npy ve sonuclar/ogrenme_egrisi.png kaydedildi")
