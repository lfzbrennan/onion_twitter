import numpy as np 
import nltk
from nltk import word_tokenize
import requests
import random
from tweepy import OAuthHandler, API
import os

def get_important_words(headline):
	tokenized = nltk.pos_tag(word_tokenize(headline))
	pos_keep = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBG', 'VBG', 'VBN', 'VBP', 'VBZ']
	important_words = [token[0] for token in tokenized if token[1] in pos_keep]
	return important_words

def process_headline(headline):
	headline = ''.join(c for c in headline if c != '\\n')
	return headline.title()

def get_image_url(headline):
	search_words = get_important_words(headline)
	query = '+'.join(search_words)

	my_key = ""
	request_text = f"https://api.unsplash.com/search/photos?page=1&query={query}&client_id={my_key}"
	data = requests.get(request_text)
	json_data = data.json()
	total = len(json_data["results"])
	index = random.randrange(0, total)
	return json_data["results"][index]["urls"]["regular"]
	
def get_processed_headline_and_image(headline_list):
	headline = np.random.choice(headline_list)
	headline_out = process_headline(headline)

	image_url = get_image_url(headline)

	return headline_out, image_url

def twitter_api():
	auth = OAuthHandler("")
	auth.set_access_token("")
	api = API(auth)
	return api

def tweet_image(message: str, url: str):
	api = twitter_api()
	filename = 'temp.jpg'
	request = requests.get(url, stream=True)
	with open(filename, 'wb') as image:
		for chunk in request:
			image.write(chunk)

	api.update_with_media(filename, status=message)
	os.remove(filename)

def post_tweet(headline_npy):
	headlines = np.load(headline_npy)
	headline_index = random.randint(0, len(headlines))
	headline = headlines[headline_index]
	headlines = np.delete(headlines, headline_index)

	np.save(headline_npy, headlines)

	processed_headline = process_headline(headline)
	image_url = get_image_url(processed_headline)

	tweet_image(processed_headline, image_url)





