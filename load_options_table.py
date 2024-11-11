import json
import pandas as pd
from helper_functions import create_feature_table_options
from update_load_genre_dict import update_load_genre_conversion_dict



def load_song_options_df():
    options_file = './data/multi_playlist_feature_table_vfull.pkl'
    base_playlist_df = pd.read_pickle(options_file).reset_index(drop=True)

    genre_groups = update_load_genre_conversion_dict(base_playlist_df)

    ref_song_feature_df, fitted_tfidf, fitted_scaler, fitted_ohe, fitted_enc = create_feature_table_options(base_playlist_df, genre_groups)

    # print(ref_song_feature_df.shape)
    return base_playlist_df, ref_song_feature_df, fitted_tfidf, fitted_scaler, fitted_ohe, fitted_enc, genre_groups