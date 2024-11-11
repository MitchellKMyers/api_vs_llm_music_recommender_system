from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd

### Helper functions
def get_track_data_v2(track_info, sp_connection, playlist_song=True, playlist_id=None, playlist_name=None):
    if playlist_song:
        track = track_info['track']
    else:
        track = track_info
    track_name = track['name']
    artists = ', '.join([artist['name'] for artist in track['artists']])
    artists_ids_lst = [artist['id'] for artist in track['artists']]
    album_name = track['album']['name']
    album_id = track['album']['id']
    track_id = track['id']
    artist_genres = []
    [artist_genres.extend(sp_connection.artist(art_id)['genres']) for art_id in artists_ids_lst]
    artist_genres = ', '.join([x for x in set(artist_genres)])
    if artist_genres == '':
        artist_genres = 'unknown'

    # Get audio features for the track
    try:
        audio_features = sp_connection.audio_features(track_id)[0] if track_id != 'Not available' else None
    except:
        audio_features = None

    try:
        release_date = track['album']['release_date']
    except:
        release_date = None

    try:
        popularity = track['popularity'] 
    except:
        popularity = None

    track_data = {
        'playlist_id': playlist_id,
        'playlist_name': playlist_name,
        'track_id': track_id,
        'track_name': track_name,
        'artist_names': artists,
        'artist_genres': artist_genres,
        'album_name': album_name,
        'album_id': album_id,
        'popularity': popularity,
        'release_date': release_date,
        'duration_ms': audio_features['duration_ms'] if audio_features else None,
        'explicit_flag': track['explicit'],
        'danceability': audio_features['danceability'] if audio_features else None,
        'energy': audio_features['energy'] if audio_features else None,
        'key': audio_features['key'] if audio_features else None,
        'loudness': audio_features['loudness'] if audio_features else None,
        'mode': audio_features['mode'] if audio_features else None,
        'speechiness': audio_features['speechiness'] if audio_features else None,
        'acousticness': audio_features['acousticness'] if audio_features else None,
        'instrumentalness': audio_features['instrumentalness'] if audio_features else None,
        'liveness': audio_features['liveness'] if audio_features else None,
        'valence': audio_features['valence'] if audio_features else None,
        'tempo': audio_features['tempo'] if audio_features else None,
                }
    
    return track_data

def get_playlist_data_v4(playlist_id, sp_connection):

    playlist_tracks = sp_connection.playlist_tracks(playlist_id, fields='items(track(id, name, artists(id, name), popularity, explicit, album(id, name, release_date)))')
    print(len(playlist_tracks['items']))
    playlist_name = sp_connection.playlist(playlist_id)['name']

    # Extract relevant information and store in a list of dictionaries
    music_data = []
    for track_info in playlist_tracks['items']:
        # print(track_info)
        try:
            track_info['track']
            track_info['track']['name']
        except:
            continue
        track_data = get_track_data_v2(track_info=track_info, sp_connection=sp_connection, playlist_id=playlist_id, playlist_name=playlist_name)
        music_data.append(track_data)

    df = pd.DataFrame(music_data)

    return df

def assign_general_genres(df, genre_groupings):
    rev_genre_groups = {tuple(y):x for x,y in genre_groupings.items()}
    
    # df['general_genre_lst'] = df.apply(lambda row: list(set(rev_genre_groups[[k for k in rev_genre_groups.keys() if raw_genre in k][0]] for raw_genre in row['artist_genres'].split(', ') if any([raw_genre in lst for lst in rev_genre_groups.keys()]))), axis=1)
    df['general_genre_lst'] = df.apply(lambda row: list(set(rev_genre_groups[[k for k in rev_genre_groups.keys() if raw_genre in k][0]] if any([raw_genre in lst for lst in rev_genre_groups.keys()]) else 'Other' for raw_genre in row['artist_genres'].split(', ') )), axis=1)
    # rev_genre_groups[[k for k in rev_genre_groups.keys() if 'belgian edm' in k][0]]
    return df
def encode_general_genre(df, fitted_enc=False):
    if fitted_enc == False:
        fitted_enc = MultiLabelBinarizer()
        fitted_enc.fit(df['general_genre_lst'])
        return fitted_enc, pd.DataFrame(columns=fitted_enc.classes_, data=fitted_enc.transform(df['general_genre_lst']))
    else:
        return pd.DataFrame(columns=fitted_enc.classes_, data=fitted_enc.transform(df['general_genre_lst']))
def vectorize_genres(df, fitted_tfidf=False):
    if fitted_tfidf == False:
        fitted_tfidf = TfidfVectorizer(max_features=300)
        tfidf_matrix =  fitted_tfidf.fit_transform(df['artist_genres'])
        vectored_df = pd.DataFrame(tfidf_matrix.toarray())
        vectored_df.columns = ['genre' + "_" + i for i in fitted_tfidf.get_feature_names_out()]
        vectored_df.drop(columns='genre_unknown', inplace=True) # drop unknown genre
        vectored_df.reset_index(drop = True, inplace=True)
        
        return  fitted_tfidf, vectored_df
    else:
        tfidf_matrix =  fitted_tfidf.transform(df['artist_genres'])
        vectored_df = pd.DataFrame(tfidf_matrix.toarray())
        vectored_df.columns = ['genre' + "_" + i for i in fitted_tfidf.get_feature_names_out()]
        vectored_df.drop(columns='genre_unknown', inplace=True) # drop unknown genre
        vectored_df.reset_index(drop = True, inplace=True)
        
        return vectored_df
        

## numerical feature scaling
def scale_audio_features(song_df, fit_scaler=False):
    audio_cols = ['duration_ms', 'danceability', 'energy', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    audio_features = song_df[audio_cols]
    
    if fit_scaler==False:
        fit_scaler = MinMaxScaler()
        fit_scaler.fit(audio_features)
        return fit_scaler, pd.DataFrame(columns=audio_cols, data=fit_scaler.transform(audio_features))
    else:
        return pd.DataFrame(columns=audio_cols, data=fit_scaler.transform(audio_features))
def encode_cat_features(df, fitted_ohe=False):
    key_df = pd.DataFrame(df['key'])
    if fitted_ohe==False:
        fitted_ohe = OneHotEncoder(handle_unknown='ignore')
        fitted_ohe.fit(key_df)
        return fitted_ohe, pd.DataFrame(columns=fitted_ohe.get_feature_names_out(), data=fitted_ohe.transform(key_df).toarray())
    else:
        return pd.DataFrame(columns=fitted_ohe.get_feature_names_out(), data=fitted_ohe.transform(key_df).toarray())
        

def create_feature_table_options(song_options_data, genre_groups):
    # song_options_data['english_likelihood'] = song_options_data[['track_name', 'album_name']].apply(lambda row: assign_english_probability(row), axis=1)
    song_options_data = assign_general_genres(song_options_data, genre_groups)
    fitted_tfidf, ref_genre_features_tfidf = vectorize_genres(song_options_data)
    fitted_scaler, ref_audio_features_scaled = scale_audio_features(song_options_data)
    fitted_ohe, ref_key_feature_ohe = encode_cat_features(song_options_data)
    fitted_enc, ref_general_genre_features_mlb = encode_general_genre(song_options_data)
    ref_song_feature_df = pd.concat([song_options_data[['playlist_name', 'track_name', 'track_id', 'popularity', 'general_genre_lst']].reset_index(drop=True), ref_general_genre_features_mlb, ref_genre_features_tfidf, ref_audio_features_scaled, ref_key_feature_ohe], axis=1)

    return ref_song_feature_df, fitted_tfidf, fitted_scaler, fitted_ohe, fitted_enc
    
def create_feature_table_input(input_song_data, fitted_tfidf, fitted_scaler, fitted_ohe, fitted_enc, genre_groups):
    # input_song_data['english_likelihood'] = input_song_data[['track_name', 'album_name']].apply(lambda row: assign_english_probability(row), axis=1)
    input_song_data = assign_general_genres(input_song_data, genre_groups)
    input_genre_features_tfidf = vectorize_genres(input_song_data, fitted_tfidf)
    input_audio_features_scaled = scale_audio_features(input_song_data, fitted_scaler)
    input_key_feature_ohe = encode_cat_features(input_song_data, fitted_ohe)
    input_general_genre_features_mlb = encode_general_genre(input_song_data, fitted_enc)
    
    input_song_feature_df = pd.concat([input_song_data[['playlist_name', 'track_name', 'track_id', 'popularity', 'general_genre_lst']].reset_index(drop=True), input_general_genre_features_mlb, input_genre_features_tfidf, input_audio_features_scaled, input_key_feature_ohe], axis=1)
    return input_song_feature_df


def compress_playlist_data(playlist_data):
    '''
    if inputted data is a playlist, compress playlist into single row by averaging all df features
    if input is a single song, still run through compressor as no change will occur.
    '''
    
    n_songs = len(playlist_data['general_genre_lst'])
    frequent_genres = [x[0] for x in Counter(playlist_data['general_genre_lst'].explode()).items() if (x[1] / n_songs) > 0.15]
    compressed_data = pd.DataFrame(playlist_data.select_dtypes(exclude='object').mean()).T
    compressed_data['playlist_name'] = ['playlist'] if len(playlist_data) > 1 else playlist_data['playlist_name']
    compressed_data['track_name'] = ['multiple'] if len(playlist_data) > 1 else playlist_data['track_name']
    compressed_data['track_id'] = ['multiple'] if len(playlist_data) > 1 else playlist_data['track_id']
    compressed_data['general_genre_lst'] = [frequent_genres]
    
    compressed_data = pd.concat([playlist_data.sample(0), compressed_data])
    assert len(compressed_data) == 1
    
    return compressed_data

def generate_recommendations_based_on_song(options_df:pd.DataFrame, song_df:pd.DataFrame, n_recos, popularity_requirement=False, same_genres=False):
    assert list(options_df.columns) == list(song_df.columns)
    cols_to_ignore = ['playlist_name', 'track_name', 'track_id', 'popularity', 'general_genre_lst']
    options_df = options_df[~options_df['track_id'].isin(song_df['track_id'].values)].drop_duplicates('track_id')
    
    if popularity_requirement == 'popular':
        options_df = options_df[options_df['popularity'] >= 65]
    elif popularity_requirement == 'unique':
        options_df = options_df[options_df['popularity'] < 65]
        
    if same_genres:
        options_df = options_df[options_df['general_genre_lst'].apply(lambda row: (any(x in row for x in song_df['general_genre_lst'].explode())) | (row == []))]
        
    # display(options_df)
    options_df['sim_score'] = cosine_similarity(options_df.drop(columns=cols_to_ignore).values, song_df.drop(columns=cols_to_ignore).values)
    
    top_recos = options_df.sort_values('sim_score', ascending=False).iloc[:n_recos]
    return top_recos
    