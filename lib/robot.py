#! /usr/bin/env python3
#! coding: utf-8

import re
import time
import logging
from pymongo import MongoClient


class intelligence(object):

	def __init__(self):
		
		# Database connector
		tSt = time.time()
		self.mongo_client = MongoClient('localhost', 27017)
		self.mongo_collection = self.mongo_client.RecipeAnalyzor.FoodData
		self.product_names = [product for product in self.mongo_collection.find({'product_name': { '$nin': ["", None, " "] } }, { 'product_name': 1 }) if '+' not in product['product_name']]
		tSp = time.time()
		logging.info('Extracted informations about {} products ({} sec).'.format(len(self.product_names), round(tSp-tSt, 2)))

	def extract_ingredients(self, recipe_steps):
		""""""
		ingredients = []
		for step in recipe_steps:
			print(step)
			#~ ingredients = [ingredient['product_name'] for ingredient in self.product_names if len(re.findall('\\b{}\\b'.format(ingredient['product_name'].lower()), step[0].lower())) > 0  and len(ingredient['product_name']) > 3]  # Considering to update the collection with the product name's length instead of comparing shit
			for i in self.product_names:
				print(i)
				matchs = re.findall('\\b{}\\b'.format(i['product_name'].lower()), step[0].lower())
				if len(matchs) > 0:
					print(i['product_name'])
			
			#~ print(list(set(ingredients)))
			print()
