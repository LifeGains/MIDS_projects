# W281: Image Classification

## Goal
- Accurately classify 20 different animal classes with a tiny dataset. Use different image manipulation/ image processing techniques to preprocess the images before using them to train several ML models.
- We don’t need to go into why image classification is important. Google: Search by image, Google Photos, search by face. Extremely important and useful if an algo can correctly classify images without human labeling.

## Role
- End to end. This was a solo project.
- High accuracy was a secondary goal in this project. The primary goal was manipulating the images. Eg. Using various types of edge detection as well as techniques for image transformation (black and white, cropping, rotating, flipping, zooming) in order to increase the count of our dataset.

## Impact
- 32% SVM, 46% CNN.
-	Total of 1500 images. 1200 training, 300 test. Divide it by 20 classes = 60 training images per class. Therefore, it was difficult for both myself and my classmates to get a classifier to achieve a high percentage accuracy with only n = 60 (without borrowing a pre-trained neural network like ResNet).
-	Even if our dataset grew by 4x with the image transformation techniques, with a n = 240, the models would still underfit. We did not use GPUs/ cloud instances for this project. Obviously a pretrained ResNet would do well but that would be “cheating.”

## Challenges
- See above. Raising the accuracy to >50% was very difficult with a small dataset.
  - Solution: Find and create as many training images as I could with edge detection, random flipping/ scaling/ zooming. Or we would simply use ResNet in real life image classification applications and build on top of it.

## Interesting Findings
-	Even the CNN did not do so well (46% vs. 80%+ was what I expected). SVM, Logistic regression, and KNN were the models we were supposed to use to optimize. I tried CNN just to see what was possible, and it was only 15% better vs. SVM, which is definitely a large jump, but still not better than a coin flip.  
