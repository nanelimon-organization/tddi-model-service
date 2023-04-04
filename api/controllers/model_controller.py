from datetime import datetime
import requests
from fastapi import APIRouter
from transformers import TextClassificationPipeline
import wsgi
from api.models.requests import Items
from logger.utils_logger import logger

model_router = APIRouter()


@model_router.post("/prediction")
async def get_label_score(items: Items) -> dict:
    """
    This endpoint that defines an endpoint for a FastAPI router.
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

    Returns:
    --------
    return : dictionary
        A dictionary containing the predicted labels and scores.

    NOTE:
        This function uses a pre-trained BERT model for Turkish language available at
        'https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased'.
    """
    start_date = datetime.now()

    api_url = "https://cryptic-oasis-68424.herokuapp.com/preprocess?tr_chars=false&acc_marks=true&punct=true&lower=true&offensive=false&norm_numbers=true&remove_numbers=false&remove_spaces=true&remove_stopwords=false&min_len=4"
    preprocess_response = requests.post(api_url, json={"texts": items.texts})
    processed_text = preprocess_response.json()['result']
    end_date = datetime.now()
    logger.info(
        f" [✓] request[https://cryptic-oasis-68424.herokuapp.com/preprocess?tr_chars=false&acc_marks=true&punct=true&lower=true&offensive=false&norm_numbers=true&remove_numbers=false&remove_spaces=true&remove_stopwords=false&min_len=4] "
        f" returned successfully - time : {end_date - start_date}"
    )

    pipeline = TextClassificationPipeline(
        model=wsgi.bert_model, tokenizer=wsgi.bert_tokenizer
    )

    results = pipeline(processed_text)

    end_date = datetime.now()
    logger.info(
        f" [✓] request[http://127.0.0.1:5000/prediction]"
        f" finished successfully - time : {end_date - start_date}"
    )

    pred_list = [
        {
            "prediction": result["label"],
            "is_offensive": 0 if result["label"] == "OTHER" else 1,
        }
        for result in results
    ]

    response_body = {"result": {"model": pred_list}}
    return response_body
