## Overview
This repo employs both classical "ML" techniques and LLMs to recommend songs based on provided songs, playlists, or user preferences. The accompanied Flask app allows users to select if they would like to receive recommendations directly from Mistral's LLM or through the combination of classical ML techniques and Spotify's API. The information required from the user to generate recommendations is dependent on what product they select.

### Example form and outputs
<img width="375" height="750" alt="Screenshot 2024-11-11 at 2 45 59 PM" src="https://github.com/user-attachments/assets/d26c62bc-5215-4ac7-a221-0edf7a081083">
<img width="350" height="750" alt="Screenshot 2024-11-11 at 2 53 33 PM" src="https://github.com/user-attachments/assets/725843d3-511e-4c07-af6f-623f081f9bde">

## Notes
- For the Spotify API connection, the initial data pull and some of the features are motivated by the work done in this article: https://towardsdatascience.com/part-iii-building-a-song-recommendation-system-with-spotify-cf76b52705e7
- The LLM from Mistral AI is used in two ways:
  1. to assign the raw artist music genres to a broad music category
  2. When recommender type == 'LLM', to output music recommendations based on user inputs
- Chat GPT was used to build out the HTML webpage

## Future Updates
- Increase the size of the music options table
- Add podcast recommendation functionality 
