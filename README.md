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


