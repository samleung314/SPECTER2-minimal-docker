FROM pytorch/pytorch:latest

WORKDIR /SPECTER2

COPY . .

RUN pip install -r requirements.txt \
    && pip install -U adapters
    
EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]