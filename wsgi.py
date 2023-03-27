from fastapi import FastAPI
from api import router
from fastapi.middleware.cors import CORSMiddleware
from transformers import (BertForSequenceClassification, BertTokenizer,
                          TextClassificationPipeline)


BERT_MODEL_PATH = "static/model/bigscience_t0_model"
BERT_TOKENIZER_PATH = "static/model/bigscience_t0_tokenizer"
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
bert_tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIZER_PATH, do_lower_case=True)
pipeline = TextClassificationPipeline(model=bert_model, tokenizer=bert_tokenizer)


def make_middleware(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )


def init_routers(app: FastAPI) -> None:
    """
    Initialize routers for the application.
    Parameters
    ----------
    app : FastAPI
        The FastAPI instance to attach the routers to.
    Returns
    -------
    None
    """
    app.include_router(router)


def create_app() -> FastAPI:
    """
    Create the FastAPI application.
    Returns
    -------
    app : FastAPI
        The FastAPI instance.
    """
    app = FastAPI(
        title="BERT Model Micro Service",
        version="0.1.0",
        description="This api is designed to analyze Turkish text using BERT, a natural language processing technology."
                    "It aims to help brands in the Telco and OTT industries to monitor and analyze Turkish text data for potential use cases, "
                    "such as identifying patterns in customer feedback or detecting inappropriate language. By providing insights into Turkish text data, "
                    "the api can assist brands in improving their customer experience and reputation management."
    )
    init_routers(app)
    make_middleware(app)
    return app


app = create_app()
