from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from api.controllers.model_controller import model_router
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TextClassificationPipeline

BERT_MODEL_PATH = "api/static/model/bigscience_t0_model"
BERT_TOKENIZER_PATH = "api/static/model/bigscience_t0_tokenizer"


class BERTModelMicroService:
    def __init__(self):
        """
        Initializes a new instance of the BERTModelMicroService class.
        """
        self.app = FastAPI(
            title="BERT Model Micro Service",
            version="0.1.0",
            description="This API analyzes Turkish text using BERT, a natural language processing technology. "
                        "It helps Telco and OTT brands to monitor and analyze Turkish text data to identify patterns in customer feedback "
                        "or detect inappropriate language, and improve their customer experience and reputation management."
        )

        self.make_middleware()
        self.bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
        self.bert_tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIZER_PATH, do_lower_case=True)
        self.pipeline = TextClassificationPipeline(model=self.bert_model, tokenizer=self.bert_tokenizer)

    def predict(self, processed_text):
        results = [f"{processed_text[index]} - {i['label']}" for index, i in enumerate(self.pipeline(processed_text))]
        return results

    def make_middleware(self):
        """
        Adds middleware to the application to enable cross-origin resource sharing.
        """
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

    def init_routes(self):
        """
        Initialize routes for the application.

        Parameters
        ----------
        app : FastAPI
            The FastAPI instance to attach the routes to.

        Returns
        -------
        None
        """
        self.app.include_router(model_router)
