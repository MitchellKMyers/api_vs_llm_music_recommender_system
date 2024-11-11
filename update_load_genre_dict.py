
from mistralai import Mistral
import time
import json
import numpy as np

api_key = '<mistral api key>'
model = "mistral-large-latest"
def group_genre_with_llm(start_id, end_id, cur_genre_list, categories_):
    client = Mistral(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": f'''You are provided with a list of raw music genres that span a wide range of musical styles. Your task is to ingest this list and organize these genres into more general, overarching music categories based on their shared musical characteristics, instrumentation, rhythms, and cultural contexts. The goal is to create a hierarchical structure that groups similar genres together under broader categories.

                            To accomplish this, you should consider the following factors:

                            Musical Characteristics: Group genres that share similar melodies, harmonies, and musical structures.
                            Instrumentation: Consider the types of instruments commonly used in each genre.
                            Rhythms and Beats: Group genres that have similar rhythmic patterns or beats per minute (BPM) ranges.
                            Cultural Context: Account for the historical and cultural background of each genre.
                            Here is the list of raw music genres that you need to cluster:
                            {cur_genre_list[start_id:end_id]}
                            
                            Additional Guidelines:
                                Some genres might fit into multiple categories. Use your best judgment to place them in the most appropriate category based on their dominant characteristics.
                                Every genre needs to be assigned to a category.
                                Assign each genre to one of these categories: {categories_}
                                Please provide the clustered list based on the given raw music genres, following the format and guidelines outlined above.
                                return the result in a short json object
                            ''',
        }
    ]


    chat_response = client.chat.complete(
        model= model,
        messages = messages,
        response_format = {
            "type": "json_object",
        }
    )
    return json.loads(chat_response.choices[0].message.content)


def update_load_genre_conversion_dict(cur_base_playlist_df):
    ## load current mapping dict
    with open('./data/genre_mappings.json', 'r') as file:
        cur_genre_groups = json.load(file)
    mapped_raw_genres = []
    for lst in cur_genre_groups.values():
        mapped_raw_genres += lst

    
    genre_list = list(set(cur_base_playlist_df['artist_genres'].str.split(', ').explode()))
    genre_list.remove('') if '' in genre_list else None
    genre_list.remove('unknown') if 'unknown' in genre_list else None
    
    ## check if there are new raw genres in cur base playlist
    new_genre_list = list(set(genre_list) - set(mapped_raw_genres))
    if len(new_genre_list) > 0:
        ## if there are new genres, use LLM to assign to a general category
        cats = ['Classical', 'Electronic/EDM', 'Rock-n-Roll/Metal', 'Pop', 'R&B/Jazz/Funk', 'Hip-Hop/Rap', 'Latin', 'Indie', 'Country', 'Other']
        print(f'Need to add {len(new_genre_list)} more genres to dict')
        
        raw_dicts = []
        id_lst = np.arange(len(new_genre_list)+25, step=25)
        for i in range(len(id_lst)-1):
            # print(id_lst[i], id_lst[i+1])
            time.sleep(1)
            raw_dicts.append(group_genre_with_llm(id_lst[i], id_lst[i+1], new_genre_list, cats))
        
        for i, g_group in enumerate(raw_dicts):
            # print(i+1)
            for k in cur_genre_groups.keys():
                if k in g_group.keys():
                    cur_genre_groups[k] = cur_genre_groups[k] + g_group[k]
        
        print('Saving updated conversion dict')
        with open('./data/genre_mappings.json', 'w') as js:
            json.dump(cur_genre_groups, js)
    
    return cur_genre_groups