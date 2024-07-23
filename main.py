from transformers import AutoTokenizer
from adapters import AutoAdapterModel

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_aug2023refresh_base')

#load proximity base model
proximity_model = AutoAdapterModel.from_pretrained('allenai/specter2_aug2023refresh_base')
#load Adhoc Query base model
adhoc_query_model = AutoAdapterModel.from_pretrained('allenai/specter2_aug2023refresh_base')

#load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
proximity_model.load_adapter("allenai/specter2_aug2023refresh", source="hf", load_as="specter2_proximity", set_active=True)
# load Adhoc Query adapter
adhoc_query_model.load_adapter("allenai/specter2_aug2023refresh_adhoc_query", source="hf", load_as="specter2_adhoc_query", set_active=True)

app = FastAPI()

class Paper(BaseModel):
    title: str
    abstract: Optional[str] = None

@app.post("/proximity")
async def proximity_embedding(papers: List[Paper]):
    # concatenate title and abstract
    text_batch = [p.title + tokenizer.sep_token + (p.abstract or '') for p in papers]
    # preprocess the input
    inputs = tokenizer(text_batch, padding=True, truncation=True,
                                       return_tensors="pt", return_token_type_ids=False, max_length=512)
    output = proximity_model(**inputs)
    # take the first token in the batch as the embedding
    embeddings = output.last_hidden_state[:, 0, :]
    embeddings = embeddings.detach().numpy().tolist()
    
    return {"embeddings": embeddings}

@app.get("/adhoc")
async def adhoc_query_embedding(query: str):
    inputs = tokenizer(query, padding=True, truncation=True,
                                       return_tensors="pt", return_token_type_ids=False, max_length=512)
    output = adhoc_query_model(**inputs)
    # take the first token in the batch as the embedding
    embeddings = output.last_hidden_state[:, 0, :]
    embeddings = embeddings.detach().numpy().tolist()
    
    return {"embeddings": embeddings}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
