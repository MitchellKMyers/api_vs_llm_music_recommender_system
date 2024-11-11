from flask import Flask, render_template, request, jsonify
from recommend_songs import run_recommender_spotify_direct, run_recommender_llm

app = Flask(__name__)

# Route to serve the HTML form
@app.route('/')
def index():
    page_file = open('./page.html', 'r')

    return page_file.read()

# API route to handle recommendations
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    recommender_type = data.get('recommenderType')

    # Generate recommendations
    if recommender_type == 'spotifyApi':
        input_type = data.get('inputType')
        uri = data.get('uri')
        num_recommendations = int(data.get('numRecommendations'))
        recommendations = run_recommender_spotify_direct(input_type=input_type, input_uri=uri, num_recommendations=num_recommendations) 
    
    elif recommender_type == 'llm':
        sngs = data.get('likedSong')
        likes = data.get('likedGenres')
        dislikes = data.get('dislikedGenres')
        num_recommendations = int(data.get('numRecommendations'))

        recommendations = run_recommender_llm(num_recommendations=num_recommendations, fav_songs=sngs, like_genres=likes, dislike_genres=dislikes)

    # Return recommendations as JSON
    return jsonify(recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
