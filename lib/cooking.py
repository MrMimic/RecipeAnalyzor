#! /usr/bin/env python3
#! coding: utf-8


import re
import time
import requests
from bs4 import BeautifulSoup


class marmiton(object):

	def __init__(self):
		pass

	def extract_recipe(self, ingredients):
		""""""
		url = 'https://www.marmiton.org/recettes/recherche.aspx?type=all&aqt={}'.format(ingredients)
		# Get top-hint recipe
		result_data = BeautifulSoup(requests.get(url).text, 'lxml')
		top_recipe_page = result_data.find_all('div', attrs={'class': 'recipe-card'})[0].find('a', attrs={'class': 'recipe-card-link'})['href']
		top_recipe_data = BeautifulSoup(requests.get(top_recipe_page).text, 'lxml')
		# Get steps
		recipe_steps = top_recipe_data.find('ol', attrs={'class': 'recipe-preparation__list'}).find_all('li', attrs={'class': 'recipe-preparation__list__item'})
		recipe_steps = [re.findall('\t([A-Z].*?)\t', step.text) for step in recipe_steps]

		return recipe_steps
