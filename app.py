# to do:
# fetch movie/show images
# let users choose between shows and movies

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import gradio as gr

embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

all_emb = np.load('embeddings.npy')

titles = pd.read_csv("data/titles.csv", encoding='utf-8', usecols=['title', 'description'])
descriptions = pd.read_csv("data/titles.csv", encoding='utf-8', usecols=['description'])
descriptions = descriptions.dropna()
titles = titles.dropna()
titles = titles.drop('description', axis=1)

def recommend(prompt):
    prompt_emb = embedder.encode(prompt, convert_to_tensor=True)
    res = util.semantic_search(prompt_emb, all_emb, top_k=1)
    res = pd.DataFrame(res[0], columns=['corpus_id', 'score'])
    match = titles.iloc[res['corpus_id']]
    des = descriptions.iloc[res['corpus_id']]
    #print(match.values[0][0])
    
    return (
        match.values[0][0],
        des.values[0][0]
    )
    
app = gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.red))

with app:
    gr.Markdown(
        """
        # NetflixGenie 🧞
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                Tell me what you're looking for and 
                I will recommend a show or movie on Netflix!
                """
            )
            prompt = gr.TextArea(
                label="What would you like to watch?",
                value="A documentary on polar bears",
                placeholder="I want to watch..."
            )
            submit = gr.Button(value="Recommend Me Something!")
        
        with gr.Column():
            result = gr.Textbox(
                label="Your recommendation... ",
            )
            description = gr.Textbox(
                label="Description"
            )
    submit.click(
        fn=recommend,
        inputs=[prompt],
        outputs=[result, description]
    )
            
            

app.launch()