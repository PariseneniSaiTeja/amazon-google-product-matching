import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

# load saved model data
with open('model_data.pkl', 'rb') as f:
    data = pickle.load(f)

amazon       = data['amazon']
google       = data['google']
tfidf        = data['tfidf']
amazon_vecs2 = data['amazon_vecs2']
google_vecs2 = data['google_vecs2']
amazon_row   = data['amazon_row']
google_row   = data['google_row']

app = FastAPI()

# define input structure
class Product(BaseModel):
    title        : str
    manufacturer : str  = ''
    price        : float = None

def assign_confidence(score):
    if score >= 0.65:  return 'HIGH'
    elif score >= 0.5: return 'MEDIUM'
    elif score >= 0.3: return 'LOW'
    else:              return 'NONE'

def find_matches(title, manufacturer='', price=None, top_n=3):
    title        = title.lower().strip()
    manufacturer = manufacturer.lower().strip()
    input_vec    = tfidf.transform([title])
    results = []
    for _, ggl in google.iterrows():
        g_i       = google_row.get(ggl['id'])
        title_sim = cosine_similarity(input_vec, google_vecs2[g_i])[0][0]
        g_mfr     = ggl['manufacturer']
        mfr_match = 1.0 if (manufacturer != '' and g_mfr != '' and manufacturer == g_mfr) else 0.0
        g_price   = ggl['price']
        if price is not None and pd.notna(g_price) and price > 0 and g_price > 0:
            price_sim = max(0, 1 - abs(price - g_price) / max(price, g_price))
        else:
            price_sim = 0.5
        final_score = (0.8 * title_sim) + (0.2 * price_sim)
        results.append({
            'google_product': ggl['name'],
            'title_sim'     : round(title_sim, 4),
            'price_sim'     : round(price_sim, 4),
            'final_score'   : round(final_score, 4),
            'confidence'    : assign_confidence(final_score)
        })
    results_df = (pd.DataFrame(results)
                .sort_values('final_score', ascending=False)
                .head(top_n)
                .reset_index(drop=True))
    return results_df.to_dict(orient='records')

# home route
@app.get('/')
def home():
    return {'message': 'Amazon Google Product Matching API is running'}

# match route
@app.post('/match')
def match(product: Product):
    matches = find_matches(
        title        = product.title,
        manufacturer = product.manufacturer,
        price        = product.price
    )
    return {
        'amazon_product': product.title,
        'matches'       : matches
    }