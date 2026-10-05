# Here we will have a function that will search for simmilar products and return an offer.
# It's simmilar to the pipeline

from google.adk import Context, Event

from .classes import EmailExtraction
from .product_lookup import search_by_description_openai
from .create_offer import build_grouped_offer

import numpy as np
import pandas as pd

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
MODEL_OFFER = "text-embedding-3-large" 

def alcohol_arg(state):
    """extraction state -> retrieval function's `alcohol` parameter"""
    return {"none": False,      # bezalkoholowe -> no alcohol
            "required": True,    # z alkoholem -> allow alcohol
            "specific": True,    # konkretny alkohol -> allow (+ filter by alcohol_detail)
            "any": None          # nie wspomniano -> any
            }.get(state, None)

def if_correct_alcohol(product, offer,df):
    """ we check if the offered products have the right alcohol"""
    correct_offer = []
    for code in offer:
        row = df[df["code"] == code]
        df_is_alcohol = row["alcohol"].iloc[0]
        if not df_is_alcohol:
            correct_offer.append(code)
        df_alcohol = row["alcohol_type"].iloc[0]                
        if product["alcohol_detail"] in df_alcohol:
            correct_offer.append(code)

    return correct_offer

# ---- Arguments that will be passes automaticly into the function ------

V = np.load("vectors_openai.npy") # previously embedded vectors 
df = pd.read_parquet("catalog.parquet") # downloaded catalog with information about the products 
perc = 0.8
delivery_price = 20
n_results = 5
model_offer = MODEL_OFFER

# ------ The actual function for building an offer --------


def create_proposals(node_input : EmailExtraction, ctx : Context):

    if isinstance(node_input, EmailExtraction):
        extraction = node_input
    else:
        extraction = EmailExtraction.model_validate(node_input)
 
    proposals = []
 
    for p in extraction.products:

        if p.url is None and p.description is not None:
            price_min, price_max = p.price_min, p.price_max
            if price_max is not None and price_min is None:
                price_min = price_max * perc
            if p.price_includes_delivery:
                price_max = price_max - delivery_price
                price_min = price_min - delivery_price
 
            res = search_by_description_openai(
                p.description, df, V, n_results,
                alcohol_arg(p.alcohol), price_min, price_max,
                p.price_basis, client, model_offer,
            )

            proposals.append({
                "label": p.description,
                "quantity": p.quantity,
                "product_description" : res
            })
 
        if p.url is None and p.name is not None:
            price_min, price_max = p.price_min, p.price_max
            if price_max is not None and price_min is None:
                price_min = price_max * perc
            if p.price_includes_delivery:
                price_max = price_max - delivery_price
                # original only subtracted from price_max here, not price_min
 
            res = search_by_description_openai(
                p.name, df, V, n_results,
                alcohol_arg(p.alcohol), price_min, price_max,
                p.price_basis, client, model_offer,
            )

            proposals.append({
                "label": p.name,
                "quantity": p.quantity,
                "product_description" : res
            })

    ctx.state['proposals'] = proposals
    return proposals
