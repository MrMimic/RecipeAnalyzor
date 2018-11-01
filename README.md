# RecipeAnalyzor

Let's see how a deep learning model can learn to cook.


## Data

https://fr.openfoodfacts.org/data

To clean .CSV version:

	cat fr.openfoodfacts.org.products.csv | cut -f 1,2,8,9,10,12,14,17,19,21,23,24,33,35,36,41,47,51,62,69,71-175 > cleaned_export_OFF.csv

******


## Database

Download bson and json files

	https://world.openfoodfacts.org/data/openfoodfacts-mongodbdump.tar.gz

Extract

	tar -xvzf openfoodfacts-mongodbdump.tar.gz

Insert

	mongorestore -d RecipeAnalyzor -c FoodData products.bson 

Check on MongoCompass if everything alright


## CRF

Data: http://cuisinez.free.fr/download.php3

Downloaded dans /data

Launch splitter.py to split this huge cooking file into many smaller one

BRAT tagging

http://brat.nlplab.org/installation.html#quick_start_installation_standalone_server

cd brat folder and copy the split recipe files into the BRAT /data

	python standalone.py

Annotate some shit with BRAT

Launch robot.py in __main__ to copy into the training set folder every annotated shit (don't need to see them into BRAT folder anymore)    method move_training_set()

Now we need to change the annotation format from BRAT to that kind of stuff:

https://eli5.readthedocs.io/en/latest/tutorials/sklearn_crfsuite.html

	[('Melbourne', 'NP', 'B-LOC'),
	 ('(', 'Fpa', 'O'),
	 ('Australia', 'NP', 'B-LOC'),
	 (')', 'Fpt', 'O'),
	 (',', 'Fc', 'O'),
	 ('25', 'Z', 'O'),
	 ('may', 'NC', 'O'),
	 ('(', 'Fpa', 'O'),
	 ('EFE', 'NC', 'B-ORG'),
	 (')', 'Fpt', 'O'),
	 ('.', 'Fp', 'O')]

So, POS tagging, then get back BRAT annotation and combine shit for each token

Then, train the shit on 0.7 * data, estimate on 0.3

We need way more examples

	python3 lib/robot.py -m -t -e

Will MOVE, then TRAIN and finally ESTIMATE the model


#### To reset the training set:

	\ls -1 *.ann |sed "s;\(.*\);rm \1 \&\& echo > \1;" |bash

Wille remove all .ann files and create them back, then copy paste to the BRAT folder


#### IDEAS

Il va falloir faire une tokenisation en prenant en compte les bigrams avec NLTK, pour mieux matcher l'annotation

Pour le moment il faut annoter token par token, vu que TextBlob tokenize le texte, il faut lui passer une regexp, mais ce guignol n'aura plus le POS

******


#### 4 examples

	Global accuracy: 0.5492572427236675
	
	             precision    recall  f1-score   support
	
	 Ingredient      0.786     0.275     0.407        40
	   Quantite      0.846     0.688     0.759        32
	    Cuisson      0.364     0.500     0.421         8
	
	avg / total      0.768     0.463     0.549        80

#### 6 examples

Global accuracy: 0.6428806215448978

             precision    recall  f1-score   support

 Ingredient      0.816     0.525     0.639        59
   Quantite      0.684     0.839     0.754        31
    Cuisson      0.000     0.000     0.000         5

avg / total      0.730     0.600     0.643        95




