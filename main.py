import uvicorn
from wsgi import BERTModelMicroService

def main():
     uvicorn.run(
        BERTModelMicroService().app,
        port=5000,
        log_level="info",
        workers=1,
    )

if __name__ == "__main__":
   main()
