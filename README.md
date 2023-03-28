# tddi-model-service

Example code :

```bash
import requests
import pandas as pd
import datetime


def get_predictions(df: pd.DataFrame) -> pd.DataFrame:

    print('Bağlantı kuruluyor..')
    start_date = datetime.datetime.now()
    api_url = "http://127.0.0.1:5000/prediction?turkish_char=false"
    response = requests.post(api_url, json={"texts": list(df.text)})
    end_date = datetime.datetime.now()
    print(f'sonuc döndü bu dönüş: {end_date-start_date} zaman sürdü.')

    predictions = response.json()['result']['model']
    for i, prediction in enumerate(predictions):
        df.at[i, 'target'] = prediction['prediction']
        df.at[i, 'is_offensive'] = prediction['is_offensive']

    return df


df = pd.read_csv('static/obs_clean_data_not_turkish_char_dumy.csv')

result = get_predictions(df)
result.to_csv('result.csv')

print(get_predictions(df).head())

```