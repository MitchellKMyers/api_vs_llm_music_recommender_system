from load_options_table import load_song_options_df
from helper_functions import create_feature_table_input, get_track_data_v2, get_playlist_data_v4, compress_playlist_data, generate_recommendations_based_on_song
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from mistralai import Mistral
import json

import pandas as pd

def run_recommender_spotify_direct(input_type, input_uri, num_recommendations):
    api_secret = '<spotipy api secret>'
    api_client_id = '<spotify api client id>'

    client_credentials_manager = SpotifyClientCredentials(client_id=api_client_id, client_secret=api_secret)
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

    base_playlist_df, ref_song_feature_df, fitted_tfidf, fitted_scaler, fitted_ohe, fitted_enc, genre_groups = load_song_options_df()

    try:
        if input_type.lower() == 'song':
            input_song_data = pd.DataFrame([get_track_data_v2(sp.track(input_uri), sp, playlist_song=False)])
        elif input_type.lower() == 'playlist':
            input_song_data = get_playlist_data_v4(input_uri, sp)
    except:
        input_song_data = base_playlist_df.sample(10)

    input_song_feature_df = create_feature_table_input(input_song_data, fitted_tfidf, fitted_scaler, fitted_ohe, fitted_enc, genre_groups)

    compressed_song_row = compress_playlist_data(input_song_feature_df)

    top_recos = generate_recommendations_based_on_song(ref_song_feature_df, compressed_song_row, num_recommendations, same_genres=False)

    cols_to_show = ['track_name',	'artist_names']
    print(input_song_data[cols_to_show])
    return [f'{nm} by {arts}' for nm, arts in base_playlist_df.loc[top_recos.index][cols_to_show].values]



def run_recommender_llm(num_recommendations, fav_songs, like_genres, dislike_genres):
    mistral_key = '<mistral api key>'
    model = "mistral-large-latest"
    
    client = Mistral(api_key=mistral_key)
    messages = [
        {
            "role": "user",
            "content": f'''Im going to provide a list of songs I am currently listenting to along with some general types of music I like and, some genres I dislike. 
            Make me a playlist of {num_recommendations} songs you would recommend based on my preferences and output the list of song name and song artist as a short json object, all recommendations cannot contain artists that I already listen to.
            Songs I listen to: {fav_songs}
            
            Genres I like: {like_genres}
            
            Genres I dislike: {dislike_genres}
            
            The output should be in a short json object where the keys are playlist[song_name, song_artist]''',
        }
    ]


    chat_response = client.chat.complete(
        model= model,
        messages = messages,
        response_format = {
            "type": "json_object",
        }
    )
    recommendations = json.loads(chat_response.choices[0].message.content)
    print(recommendations)
    return [f"{d['song_name']} by {d['song_artist']}" for d in recommendations['playlist']]
