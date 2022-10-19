## Goal
- Predict flight delays for 4 reasons:
- If something can be done to prevent delays: 
  - 1) Less delays = Happier customers = Higher LTV/ customer loyalty
  - 2) Less financial costs for airline
- Even if nothing can be done to prevent delays: 
  - 3) inform the customer ASAP so they are alerted of this situation as soon as they can 
  - 4) the airline company is more able to make appropriate arrangements ASAP (call in correct staff, make preparations, fixes).

## Role
- I was in charge of the Weather dataset (600M rows, 200 features) to find the most important features that were predictive of flight delays.
- Data cleaning was the focus here - most/all of the data had to be unnested, weather codes were in strings, had to be binned properly and translated into numerical format.
- Used descriptive statistics to create a binary prediction variable

## Impact
- 84% recall in random forest vs. 68% linear regression.
- We focused on recall because we have an imbalanced dataset where most flights are not delayed. So a dumb classifier that just classifies all flights as “not delayed” will have a high accuracy. Recall will penalize the model for classifying delayed flights as on time.
- Assumes that false negatives are more detrimental to both the airline and customers:
  - False negative: Flight is predicted to be on time but actually delayed. Everyone is Unprepared. Higher financial and time costs.
  - False positive: Flight is predicted to be delayed but on time. Good news for everyone. Lower financial and time costs.
