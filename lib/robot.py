#! /usr/bin/env python3
#! coding: utf-8

import re
import os
import time
import pickle
import logging
import argparse
import sklearn_crfsuite
from random import shuffle
from textblob import TextBlob
from pymongo import MongoClient
from textblob_fr import PatternTagger, PatternAnalyzer

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from collections import Counter




class intelligence(object):

	def __init__(self):
		
		# Database connector
		tSt = time.time()
		if 'products_data.pickle' not in os.listdir('./models/'):
			self.mongo_client = MongoClient('localhost', 27017)
			self.mongo_collection = self.mongo_client.RecipeAnalyzor.FoodData
			self.product_names = [product for product in self.mongo_collection.find({'product_name': { '$nin': ["", None, " "] }, 'countries_tags': {'$in': ['en:france']} }, { 'product_name': 1 })]
			with open('./models/products_data.pickle', 'wb') as handler:
				pickle.dump(self.product_names, handler)
			tSp = time.time()
			logging.info('Extracted informations about {} products ({} sec).'.format(len(self.product_names), round(tSp-tSt, 2)))
		else:
			with open('./models/products_data.pickle', 'rb') as handler:
				self.product_names = pickle.load(handler)
			tSp = time.time()
			logging.info('Loaded informations about {} products ({} sec).'.format(len(self.product_names), round(tSp-tSt, 2)))

	def move_training_set(self):
		"""https://www.depends-on-the-definition.com/named-entity-recognition-conditional-random-fields-python/"""

		brat_annotation_folder = '/home/emeric/1_Github/RecipeAnalyzor/data/ANNOTATION/brat-v1.3_Crunchy_Frog/data/brat_folder'
		training_set_folder = '/home/emeric/1_Github/RecipeAnalyzor/models/training_set_crf'
		moved = 0

		for filename in os.listdir(brat_annotation_folder):

			if filename.endswith('.ann'):
				size = os.path.getsize('{}/{}'.format(brat_annotation_folder, filename))

				if size > 0:
					root_name = re.findall('([0-9]{1,10}).ann', filename)[0]
					os.rename('{}/{}.ann'.format(brat_annotation_folder, root_name), '{}/{}.ann'.format(training_set_folder, root_name))
					os.rename('{}/{}.txt'.format(brat_annotation_folder, root_name), '{}/{}.txt'.format(training_set_folder, root_name))
					moved += 1
		print('{} annotation files moved.'.format(moved))

	def word2features(self, sent, i):
		word = sent[i][0]
		postag = sent[i][1]
	
		features = {
			'bias': 1.0,
			'word.lower()': word.lower(),
			'word[-3:]': word[-3:],
			'word[-2:]': word[-2:],
			'word.isupper()': word.isupper(),
			'word.istitle()': word.istitle(),
			'word.isdigit()': word.isdigit(),
			'postag': postag,
			'postag[:2]': postag[:2],
		}
		if i > 0:
			word1 = sent[i-1][0]
			postag1 = sent[i-1][1]
			features.update({
				'-1:word.lower()': word1.lower(),
				'-1:word.istitle()': word1.istitle(),
				'-1:word.isupper()': word1.isupper(),
				'-1:postag': postag1,
				'-1:postag[:2]': postag1[:2],
			})
		else:
			features['BOS'] = True
	
		if i < len(sent)-1:
			word1 = sent[i+1][0]
			postag1 = sent[i+1][1]
			features.update({
				'+1:word.lower()': word1.lower(),
				'+1:word.istitle()': word1.istitle(),
				'+1:word.isupper()': word1.isupper(),
				'+1:postag': postag1,
				'+1:postag[:2]': postag1[:2],
			})
		else:
			features['EOS'] = True
	
		return features

	def sent2features(self, sent):
		return [self.word2features(sent, i) for i in range(len(sent))]

	def sent2labels(self, sent):
		return [label for token, postag, label in sent]

	def convert_training_set(self):
		"""From STandfor format to scikit-learn crfsuite one"""


		training_set_folder = '/home/emeric/1_Github/RecipeAnalyzor/models/training_set_crf'
		crfsuite_annotation = []
		file_list = os.listdir(training_set_folder)

		for filename in file_list:

			if filename.endswith('.ann'):
				root_name = re.findall('([0-9]{1,10}).ann', filename)[0]
				with open('{}/{}.txt'.format(training_set_folder, root_name), 'r') as handler:
					text = handler.read()
				with open('{}/{}.ann'.format(training_set_folder, root_name), 'r') as handler:
					annotations = [ann.strip('\n').split('\t') for ann in handler.readlines()]

				blob = TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

				file_annotation = []

				tagged_tokens = [x[2] for x in annotations if x[0].startswith('T')]
				#~ print(tagged_tokens)

				for token in blob.tags:
					# NOW TAG THE SHIT
					if token[0] in tagged_tokens:
						tag = list(set([x[1].split(' ')[0] for x in annotations if x[0].startswith('T') and x[2] == token[0]]))[0]
					else:
						tag = '0'
					file_annotation.append((token[0], token[1], tag))

				crfsuite_annotation.append(file_annotation)

		return crfsuite_annotation

	def train_crf(self, training_set):
		""""""

		number_of_x = int(0.7*len(training_set))  # Split test / train shit

		train_sents = training_set[:number_of_x]
		test_sents = training_set[-(len(training_set)-number_of_x):]

		X_train = [self.sent2features(s) for s in train_sents]
		y_train = [self.sent2labels(s) for s in train_sents]
		
		X_test = [self.sent2features(s) for s in test_sents]
		y_test = [self.sent2labels(s) for s in test_sents]

		crf = sklearn_crfsuite.CRF(
			algorithm='lbfgs',
			c1=0.1,
			c2=0.1,
			max_iterations=100,
			all_possible_transitions=False
		)

		print('Training')
		tSt = time.time()
		crf.fit(X_train, y_train)
		tSp = time.time()
		print(tSp-tSt)

		with open('models/crf.pickle', 'wb') as handler:
			pickle.dump(crf, handler)

		return crf

	def estimate_crf(self, model, training_set):
		"""Print some quality bullshit PRF about the CRF"""

		print('Training set: {}'.format(len(training_set)))
		number_of_x = int(0.7*len(training_set))  # Split test / train shit

		train_sents = training_set[:number_of_x]
		test_sents = training_set[-(len(training_set)-number_of_x):]

		X_train = [self.sent2features(s) for s in train_sents]
		y_train = [self.sent2labels(s) for s in train_sents]
		
		X_test = [self.sent2features(s) for s in test_sents]
		y_test = [self.sent2labels(s) for s in test_sents]

		crf = model

		labels = list(crf.classes_)  # Remove shitty 0 label
		#~ print(labels)
		labels.remove('0')
		print(labels)

		# Global accuracy of that shit
		y_pred = crf.predict(X_test)

		a = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
		print('Global accuracy: {}\n'.format(a))

		sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
		prf = metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3)
		print(prf)
		print()

		# WHAT THE FUCK ARE YOUR MILESTONES?
		def print_state_features(state_features):
			for (attr, label), weight in state_features:
				print("%0.6f %-8s %s" % (weight, label, attr))

		print("Top positive:")
		print_state_features(Counter(crf.state_features_).most_common(30))
		
		print("\nTop negative:")
		print_state_features(Counter(crf.state_features_).most_common()[-30:])

	def load_model(self):
		""""""
		with open('models/crf.pickle', 'rb') as handler:
			crf = pickle.load(handler)
		return crf

	def labellize_sentence(self, sentence, model):
		"""Use the CRF to labellize some string"""

		# NOW, TAG THAT
		blob = TextBlob(sentence, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
		encoded_sentence = self.sent2features(blob.tags)

		print()
		LABELS = model.predict([encoded_sentence])
		for token, label in zip(blob.tags, LABELS[0]):
			print(token, label)


if __name__ == '__main__':


	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--move', action='store_true', help='Move newly annotated recipes from BRAT folder to training set folder.')
	parser.add_argument('-t', '--train', action='store_true', help='Train a new CRF model with data from training set folder.')
	parser.add_argument('-e', '--estimate', action='store_true', help='Estimate the last model quality.')
	parser.add_argument('-s', '--sentence', type=str, nargs='?', default=None, help='Sentence to labellize.')
	args = parser.parse_args()

	robot = intelligence()

	if args.move is True:
		robot.move_training_set()

	if args.train is True:
		training_set = robot.convert_training_set()  # Convert simple BRAT annotation (StandfordNLP format) to list [(TAG_1, POS_TAG_1, LABEL_TAG_1), (TAG_2, POS_TAG_2, LABEL_TAG_2), ...]
		CRF = robot.train_crf(training_set=training_set)  # Proper training
	else:  # Only load latest CRF
		CRF = robot.load_model()

	if args.estimate is True:
		training_set = robot.convert_training_set()
		robot.estimate_crf(model=CRF, training_set=training_set)

	if args.sentence is not None:
		robot.labellize_sentence(sentence=args.sentence, model=CRF)
