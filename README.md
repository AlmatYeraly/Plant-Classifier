# Plant Classifier
A plant classifier built by Almat Yeraly and Jude Battista with the purpose of learning Convolutional Neural Networks.

The images were scraped from Bing using Bing's Image Search API. The main limitation of scraping images from Bing was that we were only able to scrape about 500 unique images per type of house plant. Even though there is a possibility to scrape thousands of images, most of them were irrelevant and duplicates of each other. Eventually, having only 500 images per plant turned out to be our limitation with the accuracy of our model.

We built two VGGNet architectures: the first one is directly copied from [this article](https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9) and the second one is the version A of VGGNet. 

While building the first architecture, we trained the model with only two types of house plants - Aloe Vera and Peace Lilies, since the tutorial only had two classifications as well (dogs and cats). We were able to achieve about 80% training accuracy rate. Then, we decided to add two more types of plants - Spider Plant and Dumb Cane. Our choice for plants were that they are pretty common, according to a Google search. When adding two more types of plants, our model gave us about 75% training accuracy rate. Since we did not have a big dataset and we added two more classifications, it was expected that the accuracy rate would fall. This model can be found in the folder VGG Naive.

We were interested to see how the actual VGGNet architecture would perform with our dataset, hence why we decided to build the A version of VGGNet. Our reference was [this article](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/). The first thing that we noticed from the beginning was that the training process was much longer, roughly 10 min vs 4 hours. After the training was done, we got about 45% training accuracy rate. Our explanation for such low accuracy rate compared to the simplified version of VGGNet is the lack of a bigger dataset.

Overall, our goal was to build an image classifier with multiple classifications using CNNs, and we think that we were successful in achieving this goal. 
