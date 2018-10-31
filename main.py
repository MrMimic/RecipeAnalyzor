#! /usr/bin/env python3
#! coding: utf-8



import sys
import logging

sys.path.append('./lib')

from cooking import marmiton
from robot import intelligence



if __name__ == '__main__':


	# ALL, DEBUG, INFO, ERROR, FATAL
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	marmiton = marmiton()
	robot = intelligence()

	# Okay, enter some shit
	ingredients = input('Enter recipe name:  ')
	# Extract steps from Marmiton
	recipe_steps = marmiton.extract_recipe(ingredients=ingredients)
	# Now, get me the ingredients list and the way they're prepared
	ingredients = robot.extract_ingredients(recipe_steps=recipe_steps)

