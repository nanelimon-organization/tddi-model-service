<h1 align='center'>🤖 Model Service</h1>

## TDDI Model Service

Geliştirilmiş olan BERT Base, Türkçe doğal dil işleme ile hakaret tespiti modelinin çeşitli uygulamalardaki kullanımını kolaylaştırmak amacıyla geliştirilmiş bir mikroservistir. Deployment işlemleri AWS EC2 üzerinden sağlanmaktadır.

Swagger dökümanına erişmek için **[tıklayınız](http://44.210.240.127/docs)**


### Ortam Oluşturma

Lütfen Python sürümünüzü '3.10' olarak ayarlayın.

Python versiyonunuzdan emin olmak için:

```bash
python3 --version
```

#### Geliştirme Ortamını Ayarlamak
- Virtual environment oluşturunuz.
```bash
    $ python -m venv <venv-name>
```
- Virtual environmentınızı aktive ediniz.
```bash
    $ source <venv-name>/bin/activate
```
- Kütüphaneleri Yükleyiniz.
```bash
    $ pip install -r requirements.txt
```
### Çalıştırma

Uygulamanın çalışması için gerekli adımlar tamamlanmıştır.

```bash
    $ python3 main.py
```

App 5000 portunda çalışmaktadır.
> http://localhost:5000/ 


#### Örnek İstek Fonksiyonu

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
    api_url = "http://44.210.240.127/docs"
    response = requests.post(api_url, json={"texts": list(df.text)})
    end_date = datetime.datetime.now()
    print(f'sonuc döndü bu dönüş: {end_date-start_date} zaman sürdü.')

    predictions = response.json()['result']['model']
    
    for i, prediction in enumerate(predictions):
        df.at[i, 'target'] = prediction['prediction']
        df.at[i, 'is_offensive'] = int(prediction['is_offensive'])
    
    df['is_offensive'] = df['is_offensive'].astype(int)

    return df
```

