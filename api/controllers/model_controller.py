from datetime import datetime
import requests
from fastapi import APIRouter
import wsgi
from api.models.requests import Items
from logger.utils_logger import logger
import torch
import numpy as np

model_router = APIRouter()

@model_router.post("/multilabel-prediction")
async def get_label_score_multilabel(items: Items) -> dict:
    """
    Performs multilabel prediction on input texts using a saved PyTorch model.

    Parameters
    ----------
    * items : Items
        - Request body containing a list of texts to be predicted.

    Returns
    -------
    * dict
        - Response body containing a list of predictions and corresponding is_offensive values.

    Examples
    --------
    >>> texts = {"texts": ["Naber Canım?", "Naber lan hıyarto?"]}
    >>> response = requests.post("http://44.210.240.127/docs", json=texts)
    >>> print(response.json())
    {"result": {"model": [{"prediction": "OTHER", "is_offensive": 0}, {"prediction": "INSULT", "is_offensive": 1}]}}
    """
    start_date = datetime.now()

    api_url = (
        "https://cryptic-oasis-68424.herokuapp.com/preprocess?"
        "tr_chars=false&acc_marks=true&punct=true&lower=true&offensive=false&"
        "norm_numbers=true&remove_numbers=false&remove_spaces=true&remove_stopwords=false&"
        "min_len=4"
    )
    preprocess_response = requests.post(api_url, json={"texts": items.texts})
    processed_text = preprocess_response.json()["result"]
    model = wsgi.saved_model
    end_date = datetime.now()
    logger.info(
        f" [✓] request[https://cryptic-oasis-68424.herokuapp.com/preprocess?"
        f"tr_chars=false&acc_marks=true&punct=true&lower=true&offensive=false&"
        f"norm_numbers=true&remove_numbers=false&remove_spaces=true&remove_stopwords=false&"
        f"min_len=4] "
        f" returned successfully - time : {end_date - start_date}"
    )

    target_list = ["INSULT", "OTHER", "PROFANITY", "RACIST", "SEXIST"]

    predicted = []
    for text in processed_text:
        encodings = wsgi.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=64,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            model.linear.weight.fill_(0.5)
            model.linear.bias.fill_(0.1)
            input_ids = encodings["input_ids"].to("cpu", dtype=torch.long)
            attention_mask = encodings["attention_mask"].to("cpu", dtype=torch.long)
            token_type_ids = encodings["token_type_ids"].to("cpu", dtype=torch.long)
            output = model(input_ids, attention_mask, token_type_ids)
            final_output = torch.sigmoid(output).cpu().detach().numpy().tolist()
            predicted.append(target_list[int(np.argmax(final_output, axis=1))])

    end_date = datetime.now()
    logger.info(
        f" [✓] request[http://127.0.0.1:5000/prediction]"
        f" finished successfully - time : {end_date - start_date}"
    )

    pred_list = [
        {
            "prediction": result,
            "is_offensive": 0 if result == "OTHER" else 1,
        }
        for result in predicted
    ]

    response_body = {"result": {"model": pred_list}}
    return response_body
