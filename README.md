# tddi-model-service


## TDDI-Model-Service Prediction Endpoint | Example Request Function

```python
import requests
import pandas as pd
import datetime

def fetch_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sends a request to the TDDI-Model-Service prediction endpoint with a given DataFrame and retrieves the predictions for each text in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing a 'text' column that includes the texts to be predicted.

    Returns
    -------
    pd.DataFrame
        A DataFrame that includes the original 'text' column, a new 'clean_text' column that includes the cleaned version of the texts, 
        and two new columns that include the predicted target class and whether the text is offensive or not.
        
    Examples
    --------
    >>> import pandas as pd
    >>> pd.DataFrame({'text': ['Bu bir örnek metindir.','Bu da bir örnek metin!']})
    >>> result = get_predictions(df)
    >>> print(result.head())
    
                         text   target      is_offensive
    0  bu bir örnek metindir    OTHER             0
    1  bu da bir örnek metin    OTHER             0
    """
    print('Bağlantı kuruluyor..')
    start_date = datetime.datetime.now()
    api_url = "http://127.0.0.1:5000/prediction?turkish_char=true"
    response = requests.post(api_url, json={"texts": list(df.text)})
    end_date = datetime.datetime.now()
    print(f'sonuc döndü bu dönüş: {end_date-start_date} zaman sürdü.')

    predictions = response.json()['result']['model']
    
    for i, prediction in enumerate(predictions):
        df.at[i, 'target'] = prediction['prediction']
        df.at[i, 'is_offensive'] = int(prediction['is_offensive'])
    
    df['is_offensive'] = df['is_offensive'].astype(int)

    return df


df = pd.read_csv('static/obs_clean_data_not_turkish_char_dumy.csv')

result = fetch_predictions(df)
result.to_csv('result.csv')

print(get_predictions(df).head())
```

Bu Python kodu, bir metin kümesini temsil eden bir pandas DataFrame'ini alan ve bu metinler üzerinde tahminler gerçekleştiren bir dış servise (TDDI-Model-Service) istek gönderen bir fonksiyon içerir. Fonksiyon, temel olarak, metinlerin içeriğine göre sınıflandırılması ve metinlerin hakaret içerip içermediğinin belirlenmesi için kullanılır.

Kodun ana bölümleri şunlardır:

fetch_predictions fonksiyonu: Bu fonksiyon, metinler içeren bir DataFrame alır ve bu metinlerin tahminlerini almak için TDDI-Model-Service sunucusuna bir istek gönderir. İstek, tahminlerin yapıldığı sunucuya bağlantı kurmak ve metinlerin temizlenmiş ve sınıflandırılmış hallerini almak için kullanılır.
Sunucudan gelen tahminlerin DataFrame'e işlenmesi: Fonksiyon, sunucudan dönen tahminlerin her birini işler ve orijinal DataFrame'e 'target' ve 'is_offensive' sütunlarını ekler. 'target' sütunu, metnin tahmin edilen sınıfını içerirken, 'is_offensive' sütunu, metnin hakaret içerip içermediğini gösterir (1: hakaret içerir, 0: hakaret içermez).
Kodun geri kalanı, veri kümesini okur, tahminleri alır ve sonuçları bir CSV dosyasına yazar. Bu, modelin tahminlerinin değerlendirilmesi ve analiz edilmesi için kullanılabilir.
Özetle, bu kod, belirli bir metin kümesi üzerinde sınıflandırma ve hakaret içerik analizi gerçekleştiren bir dış servise istek gönderir ve sonuçları bir pandas DataFrame olarak geri döndürür. Bu, metinlerin içeriğine göre analiz edilmesini ve değerlendirilmesini sağlar.




## Example Request Function


```
import requests

api_url = "http://127.0.0.1:5000/prediction?turkish_char=true"

data = {"texts": ["Bu bir örnek metindir.", "Bu da bir örnek metin!"]}
response = requests.post(api_url, json=data)

if response.status_code == 200:
    predictions = response.json()['result']['model']
    print(predictions)
else:
    print(f"An error occurred: {response.text}")
```

Bu Python kodu, TDDI-Model-Service adlı bir dış servise HTTP POST isteği gönderen basit bir örnek gösterir. Bu özel servis, metinlerin sınıflandırılması ve hakaret içerik analizi için kullanılır.

Kodun temel bileşenleri şunlardır:

api_url: Bu değişken, TDDI-Model-Service tahmin sunucusunun URL adresini içerir. Bu örnekte, sunucu "http://127.0.0.1:5000/prediction?turkish_char=true" adresinde çalışıyor. URL'deki "turkish_char=true" sorgu parametresi, Türkçe karakterleri düzgün bir şekilde işlemek için gereklidir.

data: Bu değişken, analiz edilecek metinleri içeren bir sözlük yapısında veriyi temsil eder. Bu örnekte, iki metin içeren bir liste kullanılır: "Bu bir örnek metindir." ve "Bu da bir örnek metin!".

response: Bu değişken, TDDI-Model-Service sunucusuna gönderilen isteğin yanıtını temsil eder. İstek, requests.post fonksiyonu kullanılarak gönderilir ve api_url ve json=data argümanlarıyla yapılandırılır. Bu işlem, metinlerin sunucuya gönderilmesini ve tahminlerin alınmasını sağlar.

Yanıtın durum kodunun kontrolü: Bu bölüm, yanıtın durum kodunu kontrol ederek sunucudan başarılı bir yanıt alınıp alınmadığını belirler. Eğer durum kodu 200 ise, başarılı bir yanıt alındığını ve tahminlerin çıktı olarak yazdırılması gerektiğini gösterir. Aksi takdirde, bir hata mesajı yazdırılır.

