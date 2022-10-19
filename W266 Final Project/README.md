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


## Challenges


## Interesting Findings

