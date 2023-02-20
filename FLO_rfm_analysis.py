"""
RFM ANALİZİ İLE MÜŞTERİ SEGMENTASYONU
CUSTOMER SEGMENTATION WITH RFM
-------------------------------------------------
İş Problemi:
Online ayakkabı mağazası olan FLO müşterilerini
segmentlere ayırıp bu segmentlere göre pazarlama
stratejileri belirlemek istiyor. Buna yönelik olarak
müşterilerin davranışları tanımlanacak ve bu
davranışlardaki öbeklenmelere göre gruplar oluşturulacak.

Veri Seti Hikayesi:
Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

Değişkenler:
master_id Eşsiz müşteri numarası
order_channel Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
last_order_channel En son alışverişin yapıldığı kanal
first_order_date Müşterinin yaptığı ilk alışveriş tarihi
last_order_date Müşterinin yaptığı son alışveriş tarihi
last_order_date_online Müşterinin online platformda yaptığı son alışveriş tarihi
last_order_date_offline Müşterinin offline platformda yaptığı son alışveriş tarihi
order_num_total_ever_online Müşterinin online platformda yaptığı toplam alışveriş sayısı
order_num_total_ever_offline Müşterinin offline'da yaptığı toplam alışveriş sayısı
customer_value_total_ever_offline Müşterinin offline alışverişlerinde ödediği toplam ücret
customer_value_total_ever_online Müşterinin online alışverişlerinde ödediği toplam ücret
interested_in_categories_12 Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
"""

# Görev 1: Veriyi Anlama ve Hazırlama
import pandas as pd
import datetime as dt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.5f" % x)


# Adım 1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
df.head()


# Adım 2: Veri setinde
# a. İlk 10 gözlem,
# b. Değişken isimleri,
# c. Betimsel istatistik,
# d. Boş değer,
# e. Değişken tipleri, incelemesi yapınız.
df.head(10)
df.columns
df.describe().T
df.isnull().sum()
df.dtypes


# Adım3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]


# Adım 4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()
date_cols = [col for col in df.columns if "date" in col] # II. date_cols = df.columns[df.columns.str.contains("date")]
for col in date_cols:
    df[col] = pd.to_datetime(df[col])
# II. df[date_cols] = df[date_cols].apply(pd.to_datetime)


# Adım 5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
df.groupby("order_channel").agg({
    "master_id": "count",
    "order_num_total": "sum",
    "customer_value_total": "sum"})


# Adım 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.sort_values("customer_value_total", ascending=False).head(10)


# Adım 7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.sort_values("order_num_total", ascending=False).head(10)


# Adım 8: Veri ön hazırlık sürecini fonksiyonlaştırınız.
def data_prep(df):
    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    date_cols = df.columns[df.columns.str.contains("date")]
    df[date_cols] = df[date_cols].apply(pd.to_datetime)
    return df


# Görev 2: RFM Metriklerinin Hesaplanması
# Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.
# recency: Analiz tarihi ile müşterinin son alışveriş tarihi arasındaki fark (gün)
# frequency: müşterinin toplam alışveriş sayısı (eşsiz fatura sayısı)
# monetary: müşterinin bıraktığı toplam parasal değer


# Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
# Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
# Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.
# recency değerini hesaplamak için analiz tarihini maksimum tarihten 2 gün sonrası seçebilirsiniz.
today_date = df["last_order_date"].max() + dt.timedelta(days=2)
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (today_date - df["last_order_date"]).astype('timedelta64[D]')
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]
rfm.head()


# Görev 3: RF Skorunun Hesaplanması
# Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
# Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
# Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4 ,5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4 ,5])
rfm["RF_SCORE"] = rfm["recency_score"].astype("str") + rfm["frequency_score"].astype("str")


# Görev 4: RF Skorunun Segment Olarak Tanımlanması
# Adım 1: Oluşturulan RF skorları için "seg_map" yardımı ile segment tanımlamaları yapınız.
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
rfm.head()


# Görev 5: Aksiyon Zamanı !
# Adım1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm.groupby("segment")[["recency", "frequency", "monetary"]].mean()

df.columns
df.info()
df["interested_in_categories_12"]
df.shape
# Adım2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.
#     a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
#     tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
#     iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
#     yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz.
#     b. Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
#     iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
#     gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.
royal_or_champ_cust_ids = rfm[rfm["segment"].isin(["champions", "loyal_customers"])]["customer_id"]
target_cust_ids = df[(df["master_id"].isin(royal_or_champ_cust_ids)) & (df["interested_in_categories_12"].str.contains("KADIN"))]
target_cust_ids.to_csv("target_cust.csv", index=False)
"""
II. YOL
merge = df.merge(rfm, how="inner", left_on="master_id", right_on="customer_id")
merge.head()
target_cust = merge[(merge["segment"].isin(["champions", "loyal_customers"])) & (merge["interested_in_categories_12"].str.contains("KADIN"))]
"""

target_segments_cust_ids = rfm[rfm["segment"].isin(["cant_loose", "at_Risk", "hibernating", "new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_cust_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK")) | (df["interested_in_categories_12"].str.contains("COCUK")))]
cust_ids.to_csv("indirim_hedef_musteri.csv",index=False)

