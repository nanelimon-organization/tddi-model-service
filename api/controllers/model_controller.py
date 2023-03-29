from datetime import datetime
import requests
from fastapi import APIRouter
from transformers import TextClassificationPipeline
import wsgi
from api.models.requests import Items
from logger.utils_logger import logger
model_router = APIRouter()


@model_router.post("/prediction")
async def get_label_score(items: Items, turkish_char: bool) -> dict:
    """
    This endpoint that defines an endpoint for a FastAPI router.
    It takes in a list of text inputs (items) and a boolean value (turkish_char)
    indicating whether to replace non-Turkish characters with their Turkish counterparts.
    It then preprocesses the text inputs, encodes them with a BERT tokenizer,
    and feeds them to a pre-trained
    BERT model for Turkish language available at 'https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased'
    to perform a text classification.
    The predicted labels and whether each text is offensive or not are returned in a list of dictionaries.
    If the prediction label is "OTHER", the text is considered not offensive,
    and the "is_offensive" key in the output dictionary will be 1. Otherwise, the "is_offensive" key will be 0.
    The function returns a dictionary containing the predicted labels and scores.
    
    Parameters:
    -----------
    items : Items
        An instance of the Items class that contains a list of text inputs.
    turkish_char : bool
        A boolean value indicating whether to replace non-Turkish characters with their Turkish counterparts.

    Returns:
    --------
    return : dictionary
        A dictionary containing the predicted labels and scores.
    
    NOTE:
        This function uses a pre-trained BERT model for Turkish language available at
        'https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased'.
    """
    api_url = f'https://cryptic-oasis-68424.herokuapp.com/bulk_preprocess?turkish_char={turkish_char}'
    start_date = datetime.now()
    response = requests.post(api_url, json={"texts": items.texts})
    processed_text = response.json()["result"]
    end_date = datetime.now()
    logger.info(f' [✓] request[https://cryptic-oasis-68424.herokuapp.com/bulk_preprocess?turkish_char={turkish_char}] '
                f' returned successfully - time : {end_date-start_date}')

    start_date = datetime.now()
    pipeline = TextClassificationPipeline(model=wsgi.bert_model, tokenizer=wsgi.bert_tokenizer)
    results = pipeline(processed_text)
    end_date = datetime.now()
    logger.info(f' [✓] request[http://127.0.0.1:5000/prediction?turkish_char={turkish_char}]'
                f' finished successfully - time : {end_date-start_date}')

    pred_list = [{"prediction": result["label"], "is_offensive": 0 if result["label"] == "OTHER" else 1} for result in
                 results]
    response_body = {"result": {"model": pred_list}}
    return response_body
