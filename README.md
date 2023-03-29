# tddi-model-service

## TDDI-Model-Service Prediction Endpoint | Example Request Function

```python
import requests
import pandas as pd
import datetime


def get_predictions(df: pd.DataFrame) -> pd.DataFrame:
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
    >>> df = pd.DataFrame({'text': ['This is a sample text.','Another sample text!']})
    >>> result = get_predictions(df)
    """
    
    print('Bağlantı kuruluyor..')
    start_date = datetime.datetime.now()
    api_url = "http://127.0.0.1:5000/prediction?turkish_char=true"
    response = requests.post(api_url, json={"texts": list(df.text)})
    end_date = datetime.datetime.now()
    print(f'sonuc döndü bu dönüş: {end_date-start_date} zaman sürdü.')

    predictions = response.json()['result']['model']
    texts = response.json()['result']['texts']
    df['clean_text'] = texts
    for i, prediction in enumerate(predictions):
        df.at[i, 'target'] = prediction['prediction']
        df.at[i, 'is_offensive'] = prediction['is_offensive']

    return df


df = pd.read_csv('static/obs_clean_data_not_turkish_char_dumy.csv')

result = get_predictions(df)
result.to_csv('result.csv')

print(get_predictions(df).head())

```
