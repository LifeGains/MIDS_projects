# W203: Race vs. Hospital Preparedness on COVID Mortality Rates

## Goal:
- Determine which variable was more significant when it comes to COVID mortality rates.
- There had been a lot of debate/ rumors/ fake news around the time of COVID that COVID targeted people of a certain race, and we decided this was a great project to apply our knowledge of regression to test which variable was more significant.
- The business aspect is multifaceted. For hospitals/ scientists, they will be better equipped in knowing which variable is more likely to impact fatality. For social media businesses, they will know which posts are fake news and which posts are not, and will have the scientific/ statistical backing to delete those posts and inform the users spreading malicious information.


## Role:
- End to end. 
- Determined the outcome variable, which features were going to be in the 3 different models, making sure they all comply with the 7 CLM assumptions:
  - 1. Error term = 0 mean
  - 2. Error term = Constant variance
  - 3. Error terms = Uncorrelated
  - 4. Features + Error terms = Uncorrelated
  - 5. Features = Uncorrelated
  - 6. Linear model
  - 7. Errors = Normally distributed


## Impact:
- We found that hospital preparedness was more important than race (higher % increase in R2, larger standardized coefficient).
- On average, for every 1 unit increase in hospital preparedness, the COVID mortality rate decreases by about 45%.
- On average, for every 1% increase in the black cases, our mortality rate increases by about 0.19%.


## Challenges:
- This was a very small dataset of high level data from 46 states (after N/A’s were removed). We did not verify the quality of the data with other sources, took it for face value. 
  - Solution: We worked with what we had and did our best given our knowledge at the time.
- At this stage of the master’s program, we did not know how to create features to compliment the current features in the models.
  - Solution: We worked with what we had and did our best given our knowledge at the time.
- I was new to R at this time of project.
  - Solution: Baptism by fire - spend extra hours learning how to use it.


## Interesting Findings:
- Race was not that large of a factor as expected. You could argue that poorer communities are disproportionately affected by COVID, and since most of the poorer communities are composed of people of color. But you cannot say that simply being a certain race increases your likelihood of getting the coronavirus.

