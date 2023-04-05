from fastapi import FastAPI
from api import router
from fastapi.middleware.cors import CORSMiddleware
import __main__
from api.models.bert import BERTClass
from transformers import BertTokenizer
import torch
from logger.log_config import LogConfig
from logging.config import dictConfig


setattr(__main__, "BERTClass", BERTClass)
saved_model = torch.load("model_path", map_location=torch.device("cpu"))
saved_model.eval()
tokenizer = BertTokenizer.from_pretrained(
    "dbmdz/bert-base-turkish-128k-uncased", do_lower_case=True
)


def init_logger():
    dictConfig(LogConfig().dict())


def make_middleware(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
        description="This API analyzes Turkish text using BERT, a natural language processing technology. "
        "It helps Telco and OTT brands to monitor and analyze Turkish text data to identify patterns "
        "in customer feedback or detect inappropriate language, and improve their customer experience "
        "and reputation management.",
    )
    init_routers(app)
    make_middleware(app)
    init_logger()
    return app


app = create_app()
