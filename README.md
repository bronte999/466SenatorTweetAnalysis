# CSC 466 Senator Tweet Analysis

Congressional Tweets Final Report

By Bella White, Nathan Lee, & Vance Winstead

Github Link: https://github.com/bronte999/466SenatorTweetAnalysis

## Introduction

The goal of this project was to determine whether Congressional approval ratings or extremism could be predicted based on their Twitter activity. This report describes the analysis of the Congressional Tweets Dataset, a project which documents all tweets from all active members of Congress, starting from June 2017. From this wealth of data only the tweets originating from 2020 were selected for analysis due to its status as an election year and a highly polarizing global pandemic.  

## Data Collection

### Sources:

[Compiled Congressional Tweets](https://github.com/alexlitel/congresstweets)

[Congressional Ideology Scores](https://www.govtrack.us/congress/members/report-cards/2019/house/ideology)

[Approval Rating](https://morningconsult.com/senator-rankings/)

The Approval ratings site did not have an export, so the webpage had to be scraped using Beautiful Soup.

Additionally the Twitter APIv1 was utilized to extract more information from the Congression Tweet Dataset.


## Preprocessing

Initially the project’s scope included all congressional tweets from the year 2020. Almost immediately this became an issue as it wasn’t feasible to store and process that much data. The scope was reduced to 10,000 random samples of tweets from 2020 and restricted to only senators. 

The Congressional Twitter dataset needed to be processed as it only includes the screen name, date posted, URL of the tweet, and the truncated content of the tweet. In order to retrieve meaningful data for each tweet the Twitter API had to be utilized to retrieve tweet statistics, full text, and info about the senator’s Twitter profile. There are two versions of the Twitter API, the second version is relatively new but includes more engagement metrics. However the first version has more documentation and was in a more stable condition so it was chosen for this project. 

The preprocessing steps are included in Senate_Tweet_Preprocessing.ipynb in the linked GitHub repo. 

In order to perform sentiment analysis, the VADER sentiment analysis tool was used. This tool is included as part of the Python package nltk. VADER specializes in analyzing digital communication, as it is able to recognize slang, all caps, and emoticons and factor that into its sentiment analysis in addition to the more traditional analysis of words and punctuation. After running sentiment analysis, VADER will produce four fields: positive, negative, neutral, and compound. The first three are on a scale from 0.0 to 1.0 and are equivalent to the percentage of words in that Tweet that fall into that category. The compound value is on a scale from -1.0 to 1.0 and normalizes all lexicon ratings, allowing you to analyze both positive and negative sentiment at the same time.

A “extremism” score was also calculated by computing the z score of a congress member’s ideology and taking the absolute value. This enables the grouping of the far left with the far right into one group, leaving  centrists in the other group.

To combine the Twitter API and sentiment data with the ideology and approval rating data, the two columns were joined based upon the name of the congressperson. Many Twitter names contained added words to their full name such as “U.S. Senator Bill Cassidy, M.D.” so the two dataframes were joined by comparing the full name in the ideology scores to a substring of the Twitter full name. The implementation of this join can be observed in the code below.

```python
congress_groups['join'] = 1
approval['join'] = 1
tweets_and_approval_full = congress_groups.merge(approval, on='join').drop('join', axis=1)
approval.drop('join', axis=1, inplace=True)
tweets_and_approval_full['match'] = tweets_and_approval_full.apply(lambda x: x.full_name.find(x.Name), axis=1).ge(0)
tweets_and_approval = tweets_and_approval_full[tweets_and_approval_full['match']].drop(columns=['match'])
```
Note: This code was inspired by: https://towardsdatascience.com/joining-dataframes-by-substring-match-with-python-pandas-8fcde5b03933

## Neural Network
The first predictor tried was a multilayer perceptron or neural network. Neural networks are a supervised learning method and can be used for classification. Neural networks are a good application to this problem because neural networks handle continuous values for classification well. The dataset is full of continuous values that would have to be binned for other approaches such as naive bayesian classification.

![alt text](https://scikit-learn.org/stable/_images/multilayerperceptron_network.png)

Image from https://scikit-learn.org/stable/modules/neural_networks_supervised.html

A multilayer perceptron has an input and output. That input is passed through nodes that shape the input with different weights to get an output. Those weights are created by “training” the data by using a “training” set of data, a portion of the data. The perceptron begins as a set of random weights, and in the training phase, an input is passed through to become a likely random output. Then, through a process called backpropagation, the weights are adjusted so that they eventually output something closer to the expected output. 

The neural network was used to predict a senator’s political party based on details of their Twitter account. The input was: following_count, num_tweets_user_posted, num_posts_user_liked,  avg_retweets, avg_compound_sentiment, and avg_likes_per_follower. Once a classifier was developed, feature importance could be run and used to determine what factors most into how senators of different political parties tweet.

To preprocess the data for the network, each column of the data was standardized with the approach of (x - u) / sd to determine how many standard deviations away from the mean each value of x is. 

After several trial runs and experimenting with different network sizes, the accuracy hovered between 70 and 80% which is well above the baseline of guessing one category. One of the trials had a hidden network size  of 2 layers of 70 neurons with a Softmax activation function.
After training the models, test based feature importance was run on the model. Test based feature importance is a process that goes through each predictor feature in the test dataset and randomizes it. After randomization, the neural network uses that data to try to predict the output. If a feature is more significant, the model will not be as accurate since the model used that feature heavily to come to the final decision. If a feature is less significant, the model output will be more correct since the model didn’t use that feature as much to make the final decision. 

The most important feature found is avg_compound_sentiment. That means that the trained neural network used avg_compound_sentiment the most when classifying a senator’s political party based on their Twitter. That implies that one party Tweets distinctly more negatively than the other. After some aggregation, the data confirmed there was a distinct difference between tweet sentiment of different parties, revealing that Democrats’ tweets had an average sentiment of -0.53 compared to a more positive sentiment from the Republicans’ tweets at 0.49. 

## Decision Tree

In order to get an alternative perspective on the feature importance of some of this data, a decision tree was used. Since decision trees naturally sort data points from most to least important as they create branches, they make it very transparent which columns are and aren’t contributing to the accuracy of a model.

The decision tree that was used (based on the decision tree from Lab 4) was able to split continuous variables, but it could not predict continuous variables. That meant that the first step in this process was to bin all of the columns which it would try to predict independently. These columns were "ideology", "extremism", "pos_approval", "neutral_approval", and "neg_approval". Since the data being used for this model only included 80 senators and 24 of those senators would be used for testing, most columns were only given 2-4 bins in order to allow the model to be somewhat generalized.

Next, a script was developed to test all of these columns in quick succession.

```python
from sklearn import tree as skl_tree
xCols = ["following_count", "num_tweets_user_posted", "followers_count", "avg_retweets", "avg_likes", "avg_compound_sentiment"]
yCols = ["ideology", "extremism", "pol_leaning", "is_extreme", "extreme_left", "extreme_right", "pos_approval", "neutral_approval", "neg_approval"]
X = bin_df[xCols]
 	 
for yCol in yCols:
	print(f"Analysing {yCol}...")
	y = bin_df[yCol]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
	col, gr = Decision_tree_helper.select_split2(X_train, y_train)
	print(f"The first branch was {col}, which had a gain ratio of {gr}")
	tree = Decision_tree_helper.make_tree2(X_train,y_train,min_split_count=5)
	rules = Decision_tree_helper.generate_rules(tree)
	default = y_train.mode()[0]
	y_guesses = X_test.apply(lambda x: Decision_tree_helper.make_prediction(rules,x,default),axis=1)
	accuracy = np.count_nonzero(y_guesses == y_test) / len(y_test)
	baseline = np.count_nonzero(default == y_test) / len(y_test)
	clf = skl_tree.DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
	y_guesses_skl = clf.predict(X_test)
	accuracy_skl = np.count_nonzero(y_guesses_skl == y_test) / len(y_test)
	print(f"This model ended up having an accuracy of {accuracy}, whereas the Scikit Learn model had an accuracy of {accuracy_skl}. The baseline was {baseline}\n")
```

This code generated the following output: 

```
Analysing ideology...
The first branch was avg_compound_sentiment<0.40, which had a gain ratio of 0.15698445772957084
This model ended up having an accuracy of 0.36, whereas the Scikit Learn model had an accuracy of 0.28. The baseline was 0.32

Analyzing extremism...
The first branch was num_tweets_user_posted<6923.50, which had a gain ratio of 0.105456372213472
This model ended up having an accuracy of 0.44, whereas the Scikit Learn model had an accuracy of 0.44. The baseline was 0.44

Analyzing pol_leaning...
The first branch was avg_compound_sentiment<0.40, which had a gain ratio of 0.3037776108837696
This model ended up having an accuracy of 0.72, whereas the Scikit Learn model had an accuracy of 0.44. The baseline was 0.48

Analyzing is_extreme...
The first branch was following_count<222.50, which had a gain ratio of 0.06560110083671669
This model ended up having an accuracy of 0.48, whereas the Scikit Learn model had an accuracy of 0.56. The baseline was 0.48

Analyzing extreme_left...
The first branch was followers_count<506987.00, which had a gain ratio of 0.25537803091594086
This model ended up having an accuracy of 0.76, whereas the Scikit Learn model had an accuracy of 0.76. The baseline was 0.76

Analyzing extreme_right...
The first branch was avg_likes<3352.70, which had a gain ratio of 0.11853752375355932
This model ended up having an accuracy of 0.76, whereas the Scikit Learn model had an accuracy of 0.44. The baseline was 0.72

Analyzing pos_approval...
The first branch was num_tweets_user_posted<18431.50, which had a gain ratio of 0.11476548405128674
This model ended up having an accuracy of 0.4, whereas the Scikit Learn model had an accuracy of 0.36. The baseline was 0.64

Analyzing neutral_approval...
The first branch was followers_count<886518.50, which had a gain ratio of 0.1374879679512903
This model ended up having an accuracy of 0.32, whereas the Scikit Learn model had an accuracy of 0.4. The baseline was 0.36

Analyzing neg_approval...
The first branch was following_count<222.50, which had a gain ratio of 0.07406453531849162
This model ended up having an accuracy of 0.36, whereas the Scikit Learn model had an accuracy of 0.36. The baseline was 0.36
```

Most of these models failed to predict any better than the baseline. However, there is one notable exception to this: the pol_leaning column, which had an accuracy of 0.72 and a baseline of 0.48. This column was created by splitting the ideology scores into bins with one less than 0.5 and the other greater than or equal to 0.5. Furthermore, the most important feature in this tree was the compound_sentiment, which had a gain ratio of 0.3038, indicating that it did significantly reduce entropy within the tree. Looking at a graph of ideology against compound sentiment, it is clear there certainly is a trend between the two.

While the trend is a bit less clear when compound sentiment is between 0.1 and 0.4, if you look at the extremes, the partisan divide is very clear. Nobody with an ideology above 0.5 had a compound sentiment below 0.1, and nobody with an ideology lower than 0.5 had an ideology above 0.45. This means that there were no liberals with extremely positive sentiments and there were no conservatives with extremely negative sentiments. This will be analyzed further in the following section.

## Results
From this analysis of Senatorial tweets from 2020 an association between the average compound sentiment and ideological scores can be established. 
