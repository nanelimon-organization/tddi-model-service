import requests
from fastapi import APIRouter
from transformers import TextClassificationPipeline
import wsgi
from api.models.requests import Items

model_router = APIRouter()


@model_router.post("/prediction")
async def get_label_score(items: Items, turkish_char: bool) -> dict:
    """
    This endpoint takes a list of text inputs in the 'texts' field of the Items class and a boolean value
    for the 'turkish_char' parameter. It preprocesses the text inputs, encodes them with the BERT tokenizer,
    and feeds them to a pre-trained BERT model (dbmdz/bert-base-turkish-128k-uncased) to perform a text
    classification. The predicted labels and whether each text is offensive or not are returned in a list of
    dictionaries.

    If the prediction label is 1, the text is considered offensive, and the 'is_offensive' key in the output
    dictionary will be 0. Otherwise, the 'is_offensive' key will be 1.

    Parameters:
    -----------
    items : Items
        An instance of the Items class that contains a list of text inputs.
    turkish_char : bool
        A boolean value indicating whether to replace non-Turkish characters with their Turkish counterparts.

    Returns:
    --------
    return : dictionary
        A dictionary containing the predicted labels and indicators for whether each text is offensive or not.

    NOTE:
        This function uses a pre-trained BERT model for Turkish language available at
        'https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased'.
    """

    api_url = f'https://cryptic-oasis-68424.herokuapp.com/bulk_preprocess?turkish_char={turkish_char}'

    response = requests.post(api_url, json={"texts": items.texts})

    processed_text = response.json()["result"]

    pipeline = TextClassificationPipeline(model=wsgi.bert_model, tokenizer=wsgi.bert_tokenizer)
    results = pipeline(processed_text)

    return {"success": True,
            "payload": results}