# W266: Subreddit Classification

## Goal
- "Deslanging" the content of reddit posts will make them easier to classify.
- Automatic classification of unstructured data is extremely valuable for businesses. Eg. Targetted ads, clustering users into different categories without explicit labels.
- Had originally wanted to do Tweet classification but there wasn’t enough labeled data from the Twitter API - they had just started to roll out tweet labeling. In addition, all labeled tweets often came from Verified/ official accounts, and there was very little slang in those tweets. Tweets with lots of slang in them were unlabeled.

## Role
- End to end. Pulled posts from reddit via API and web scraping
- Use NLTK to preprocess words (remove punctuation, stop words, lowercase)
- Applied deslanged conversion via SlangIt dictionary
-	Ran Naïve Bayes (baseline) and BERT model on many different permutations of data (similar subreddits, random subreddits, slang-heavy subreddits, n = 5k, 10k, 25k)

## Impact
- Even though BERT outperformed the baseline and did a fantastic job classifying subreddits (83% F1 score), all iterations of all models did slightly worse (1-2%) after deslanging.
- The unknown words (slang) were actually used by BERT to categorize the posts into the appropriate subreddit.
- Without the unknown tokens which appeared in certain reddit subreddits, the de-slanged words now look the same as the other subreddits, which made it more difficult for the model to predict the correct category.


## Challenges


## Interesting Findings

