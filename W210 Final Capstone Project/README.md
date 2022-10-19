## Goal:
- ML enabled stock picking system for retail investor.
- Retail investors do not have access to ML-driven quant funds in current day and age. Minimum = 100K, 1M+ to get access to quant funds, or are just closed to outsiders. Our platform is a ML driven stock selection solution for retail investors for a very low minimum.

## Role:
- Due to expertise in finance, I spearheaded feature engineering/ selection/ creation portion. 50 out of 130 features were significant. We used PCA & SHAP to narrow down number of features.

## Impact:
- Accuracy was low 55-60% with XGBoost + Gridsearch. However, despite low accuracy, the winning picks outperformed the losing picks. Over roughly 20 years (2003-2022), we outperformed the SPX by 2x.

## Challenges:
- Obtaining data with enough history was a challenge. Could not combine multiple different sources because different tags, ways of calculating features. 
  - Solution: Used WRDS from Upenn database, since 1990.
- Deciding to use daily monthly or quarterly data. 
  - Solution: Monthly so that there is enough training data and less noise.
- Getting the right price to plot correctly post-split was a challenge. If we used a market cap filter eg. <$300M, there would be gaps in our data since some stocks eg. AAPL would dip below $300M market cap periodically in our training set. 
  - Solution: Above $5 in pre split price was a more lenient filter and allowed for more rows to be retained.

## Interesting Findings:
-	Traditional metrics (eg. P/E, FCF, volatility, R&D) were more important to the model than most of the features we created 
- Model did not think Y/Y percentage changes were important, contrary to my initial belief. Eg. Earnings growth of 20%+. 

