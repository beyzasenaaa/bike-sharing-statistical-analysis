# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 17:52:12 2024

@author: aktas
"""

import numpy as np
import pandas as pd

# pandas kutuphanesi ile bilgisayardaki csv dosyasindan veri seti okuma
veriSeti = pd.read_csv("C:/Users/aktas/Desktop/İSTATİSTİKSEL YAZILIMLAR FİNAL ÖDEVİ/bike+sharing+dataset/day.csv")

veriSeti.info()
veriSeti.dtypes
print(veriSeti.shape)

row, col = veriSeti.shape
print(row)
print(col)
print(veriSeti.size) 
list(veriSeti.columns) 


for col in veriSeti.columns:
    print(col)
    veriSeti.rename(
    columns={"instant": "kayıt_dizini", 
        "dteday": "tarih", 
        "season": "mevsim", 
        "yr": "yıl", 
        "mnth": "ay", 
        "holiday": "tatil", 
        "weekday": "haftanın_günü", 
        "workingday": "çalışma_günü", 
        "weathersit": "hava_durumu", 
        "temp": "sıcaklık", 
        "atemp": "hissedilen_sıcaklık", 
        "hum": "nem", 
        "windspeed": "rüzgar_hızı", 
        "casual": "gündelik_kullanıcı", 
        "registered": "kayıtlı_kullanıcı", 
        "cnt": "toplam_kullanıcı"},
    inplace=True,
)


    # Kategorik değişkenlerin veri tipini "category" yapma ve kategorik değerleri açıklamalı hale getirme
veriSeti["mevsim"] = veriSeti["mevsim"].astype("category")
veriSeti["mevsim"] = veriSeti["mevsim"].replace({
    1: "İlkbahar",
    2: "Yaz",
    3: "Sonbahar",
    4: "Kış"
})

veriSeti["yıl"] = veriSeti["yıl"].astype("category")
veriSeti["yıl"] = veriSeti["yıl"].replace({
    0: "2011",
    1: "2012"
})

veriSeti["ay"] = veriSeti["ay"].astype("category")
veriSeti["ay"] = veriSeti["ay"].replace({
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
    7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
})

veriSeti["tatil"] = veriSeti["tatil"].astype("category")
veriSeti["tatil"] = veriSeti["tatil"].replace({
    0: "Tatil Değil",
    1: "Tatil"
})

veriSeti["haftanın_günü"] = veriSeti["haftanın_günü"].astype("category")
veriSeti["haftanın_günü"] = veriSeti["haftanın_günü"].replace({
    0: "Pazar", 1: "Pazartesi", 2: "Salı", 3: "Çarşamba",
    4: "Perşembe", 5: "Cuma", 6: "Cumartesi"
})

veriSeti["çalışma_günü"] = veriSeti["çalışma_günü"].astype("category")
veriSeti["çalışma_günü"] = veriSeti["çalışma_günü"].replace({
    0: "Çalışma Günü Değil",
    1: "Çalışma Günü"
})

veriSeti["hava_durumu"] = veriSeti["hava_durumu"].astype("category")
veriSeti["hava_durumu"] = veriSeti["hava_durumu"].replace({
    1: "Açık/Az bulutlu/Parçalı bulutlu",
    2: "Sisli + Bulutlu",
    3: "Hafif Kar/Yagmur + Fırtına",
    4: "Şiddetli Yağmur + Kar + Sis"
})

# Değişikliklerin kontrolü
print(veriSeti.dtypes)
print(veriSeti.head())

veriSeti.describe()  # sadece sayisal degiskenlerin istaitstikleri
describe_output = veriSeti.describe()

pd.set_option('display.max_columns', None)
describe_output = veriSeti.describe(include="all") # tüm degiskenlerin istaitstikleri

#Eksik Değer
veriSeti.isnull().any() #Tüm değerler false dönmekte, bu sütunda hiç eksik değer olmadığını söyleyebiliriz.
veriSeti.isnull().sum() #Tüm sütunlar için eksik değer sayısı sıfır (0), bu da bize her bir sütunun eksiksiz olduğunu gösterir.


from scipy.stats import trim_mean, skew, kurtosis
from scipy.stats import mode
from statsmodels import robust

# Tanımlayıcı istatistik fonksiyonu
def summary_statistics(sütun):

    return {
        "Ortalama": sütun.mean(),
        "Ortanca": sütun.median(),
        "Kırpılmış Ortalama": trim_mean(sütun, 0.1),
        "Mod": mode(sütun).mode[0] if not sütun.mode().empty else np.nan,
        "Aralık": sütun.max() - sütun.min(),
        "Çeyreklik Aralık": sütun.quantile(0.75) - sütun.quantile(0.25),
        "Ortalama Mutlak Sapma": sütun.mad(),
        "Varyans": sütun.var(),
        "Standart Sapma": sütun.std(),
        "Medyan Mutlak Sapma": robust.mad(sütun),
        "Eğim (Skewness)": skew(sütun),
        "Basıklık (Kurtosis)": kurtosis(sütun)
    }

# Analiz edilecek sütunlar
sütunlar = ["sıcaklık", "hissedilen_sıcaklık", "nem", "rüzgar_hızı", "gündelik_kullanıcı", "kayıtlı_kullanıcı", "toplam_kullanıcı"]

# Her sütun için istatistikleri hesaplama
for sütun_adı in sütunlar:
    print(f"{sütun_adı} için Tanımlayıcı İstatistikler:")
    istatistikler = summary_statistics(veriSeti[sütun_adı])  # Verilen sütun adı ile fonksiyonu çağırıyoruz
    for key, value in istatistikler.items():
        print(f"{key}: {value}")
    print("\n" + "-"*50 + "\n")
     
## KORELASYON------------------------------------------------------------------
veriSeti.corr() 
veriSeti.corr().unstack().min()  
veriSeti.corr().unstack().idxmin()  

# Korelasyon grafigi
import seaborn as sns
veriSeti.corr()
sns.heatmap(
    veriSeti.corr(), 
    annot = True, # Korelasyon degerlerinin grafigin uzerine yazdirma
    square=True, # Kutularin kare bicimde gosterilmesi
    cmap="Oranges" # Renklendirme secenegi
)
sns.heatmap(veriSeti.corr(),annot=True,linewidth = 0.5, cmap='coolwarm')


##GRAFİKLER--------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Scatter Plot: Sıcaklık ve Nem
x = veriSeti['sıcaklık']  
y = veriSeti['nem']   
plt.scatter(x, y, color="hotpink")
plt.title("Sıcaklık ve Nem Değerleri")
plt.xlabel("Sıcaklık")
plt.ylabel("Nem")
plt.show()

# 2. Seaborn relplot: 'rüzgar_hızı' ve 'nem' ile 'çalışma_günü' büyüklüğüne göre
sns.set_theme(style="white")
sns.relplot(x="rüzgar_hızı", y="nem", hue="çalışma_günü", size="rüzgar_hızı",  
            sizes=(40, 400), alpha=.5, palette="muted", height=6, data=veriSeti)

# 3. Seaborn relplot: 'sıcaklık' ve 'rüzgar_hızı' ile 'ay' kategorisine göre
sns.relplot(x="sıcaklık", y="rüzgar_hızı", hue="ay",  
            sizes=(40, 400), alpha=.5, palette="muted", height=6, data=veriSeti)

# 4. Swarm plot: 'sıcaklık' ve 'haftanın_günü', 'ay' ile renkli
ax = sns.swarmplot(data=veriSeti, x="sıcaklık", y="haftanın_günü", hue="ay")  
ax.set(ylabel="")

# 5. Pairplot: Alt grup 'sıcaklık', 'rüzgar_hızı', 'ay', ve 'çalışma_günü' ile
alt_grup = veriSeti[["sıcaklık", "rüzgar_hızı", "hava_durumu", "çalışma_günü"]]  
sns.pairplot(alt_grup, hue="çalışma_günü")

# 6. Pairplot: Alt grup 'sıcaklık', 'rüzgar_hızı', 'hava_durumu', ve 'ay' ile
alt_grup_2 = veriSeti[["sıcaklık", "rüzgar_hızı", "hava_durumu", "ay"]]  
sns.pairplot(alt_grup_2, hue="ay")

plt.show()


## BASİT RASTGELE ÖRNEKLEME----------------------------------------------------

rasgele_orneklem = veriSeti.sample(n=100, random_state=42)
print(rasgele_orneklem)



## MERKEZİ LİMİT TEOREMİ-------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Örneklem büyüklükleri 
sample_sizes = [10, 50, 1000]  
n_experiments = 1000  # 100 örneklem alınacak


# Grafikleri oluşturmak için alt figürler
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Her bir örneklem büyüklüğü için örneklem ortalamalarını hesapla
for i, size in enumerate(sample_sizes):
    sample_means = [np.mean(np.random.choice(veriSeti["kayıtlı_kullanıcı"], size=size)) for _ in range(n_experiments)]
    
    # Histogram çizimi
    sns.histplot(sample_means, bins=30, kde=True, color='pink', ax=axes[i])
    axes[i].set_title(f'Sample size = {size}')
    axes[i].set_xlabel('Ortalama')
    axes[i].set_ylabel('Sıklık')

plt.tight_layout()
plt.show()

## PAREMETRELERİN TAHMİNİ------------------------------------------------------
import pandas as pd

# Popülasyon ortalamasını hesaplayın
populasyon_ortalama = veriSeti['kayıtlı_kullanıcı'].mean()

# Popülasyon standart sapmasını hesaplayın
populasyon_std = veriSeti['kayıtlı_kullanıcı'].std()


print(f"Popülasyon Ortalaması: {populasyon_ortalama}")
print(f"Popülasyon Standart Sapması: {populasyon_std}")


## GÜVEN ARALIĞI---------------------------------------------------------------
import scipy.stats as stats


populasyon_ortalama = veriSeti['kayıtlı_kullanıcı'].mean()  # Örneklem ortalaması
populasyon_std = veriSeti['kayıtlı_kullanıcı'].std()  # Popülasyon standart sapması
n = len(veriSeti)  # Örneklem büyüklüğü

# %95 güven için Z-değeri
z_degeri = stats.norm.ppf(0.975)

# Güven aralığı
margin_of_error = z_degeri * (populasyon_std / np.sqrt(n))
lower_limit = populasyon_ortalama - margin_of_error
upper_limit = populasyon_ortalama + margin_of_error

print(f"%95 Güven Aralığı: [{lower_limit}, {upper_limit}]")


from statsmodels.stats.weightstats import ztest
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, shapiro, mannwhitneyu
import numpy as np

# 14.1. Tek Örneklem Z-Testi
z_stat, z_p_value = ztest(veriSeti['sıcaklık'], value=0.25)
print(f"14.1. Tek Örneklem Z-Testi -> Z-İstatistiği: {z_stat}, P-Değeri: {z_p_value}")

# 14.2. Tek Örneklem T-Testi
t_stat, t_p_value = ttest_1samp(veriSeti['nem'], popmean=0.5)
print(f"14.2. Tek Örneklem T-Testi -> T-İstatistiği: {t_stat}, P-Değeri: {t_p_value}")

# 14.3. Bağımsız Örneklem T-Testi (Student Testi)
calisma = veriSeti[veriSeti['çalışma_günü'] == 'Çalışma Günü']['sıcaklık']
tatil = veriSeti[veriSeti['çalışma_günü'] == 'Çalışma Günü Değil']['sıcaklık']
t_stat, t_p_value = ttest_ind(calisma, tatil, equal_var=True)
print(f"14.3. Bağımsız Örneklem T-Testi (Student) -> T-İstatistiği: {t_stat}, P-Değeri: {t_p_value}")

# 14.4. Bağımsız Örneklem T-Testi (Welch Testi)
t_stat, t_p_value = ttest_ind(calisma, tatil, equal_var=False)
print(f"14.4. Bağımsız Örneklem T-Testi (Welch) -> T-İstatistiği: {t_stat}, P-Değeri: {t_p_value}")

# 14.5. Eşleştirilmiş Örneklem T-Testi
before = veriSeti[veriSeti['tarih'] < '2011-01-15']['sıcaklık']
after = veriSeti[veriSeti['tarih'] >= '2011-01-15']['sıcaklık'][:len(before)]
t_stat, t_p_value = ttest_rel(before, after)
print(f"14.5. Eşleştirilmiş Örneklem T-Testi -> T-İstatistiği: {t_stat}, P-Değeri: {t_p_value}")

# 14.6. Tek Yönlü Testler
t_stat, t_p_value = ttest_ind(calisma, tatil, alternative='greater')
print(f"14.6. Tek Yönlü Testler -> T-İstatistiği: {t_stat}, P-Değeri: {t_p_value}")

# 14.7. Etki Büyüklüğü (Effect Size)
mean_diff = calisma.mean() - tatil.mean()
pooled_std = np.sqrt((calisma.std()**2 + tatil.std()**2) / 2)
effect_size = mean_diff / pooled_std
print(f"14.7. Etki Büyüklüğü (Cohen's d): {effect_size}")

# 14.8. Bir Örneklemin Normalliğini Kontrol Etme
shapiro_stat, shapiro_p_value = shapiro(veriSeti['sıcaklık'])
print(f"14.8. Bir Örneklemin Normalliğini Kontrol Etme -> Shapiro-Wilk Testi: {shapiro_stat}, P-Değeri: {shapiro_p_value}")

# 14.9. Normale Uymayan Veriler için Wilcoxon Testleri
mann_stat, mann_p_value = mannwhitneyu(calisma, tatil)
print(f"14.9. Normale Uymayan Veriler için Wilcoxon Testleri -> Mann-Whitney U Testi: {mann_stat}, P-Değeri: {mann_p_value}") 





