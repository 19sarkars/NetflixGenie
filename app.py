from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import gradio as gr

embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

all_emb = np.load('embeddings.npy')

def recommend(prompt, option):
    raw_data = pd.read_csv("data/titles.csv", encoding='utf-8')
    titles_unfiltered = pd.read_csv("data/titles.csv", encoding='utf-8', usecols=['title', 'description', 'type'])
    titles_unfiltered = titles_unfiltered.dropna() #6114 titles
    titles_unfiltered = titles_unfiltered.reset_index(drop=True)
    
    if option == "Movie": 
        titles = titles_unfiltered.loc[titles_unfiltered['type'] == 'MOVIE']
        removed = titles_unfiltered.index.difference(titles.index).tolist()
        filtered_emb = np.delete(all_emb, removed, 0)
        
    elif option == "TV Show": 
        titles = titles_unfiltered.loc[titles_unfiltered['type'] == 'SHOW']
        removed = titles_unfiltered.index.difference(titles.index).tolist()
        filtered_emb = np.delete(all_emb, removed, 0)
        
    else:
        filtered_emb = all_emb
        titles = titles_unfiltered
        
    titles = titles.drop(['description', 'type'], axis=1)
    prompt_emb = embedder.encode(prompt, convert_to_tensor=True)
    res = util.semantic_search(prompt_emb, filtered_emb, top_k=1)
    res = pd.DataFrame(res[0], columns=['corpus_id', 'score'])
    match = titles.iloc[res['corpus_id']]
    pd.set_option('display.max_colwidth', None)
    des = raw_data.loc[raw_data['title'] == match.values[0][0], 'description']
    imdb = raw_data.loc[raw_data['title'] == match.values[0][0], 'imdb_score']
    
    return (
        match.values[0][0],
        des.to_string(index=False),
        imdb.to_string(index=False)
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
            choice = gr.Radio(
                ["Movie", "TV Show", "Anything"],
                label="Pick one:",
                value="Anything"
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
            rating = gr.Textbox(
                label="IMDb Rating"
            )
            description = gr.Textbox(
                label="Description"
            )
    submit.click(
        fn=recommend,
        inputs=[prompt, choice],
        outputs=[result, description, rating]
    )
            
            

app.launch()