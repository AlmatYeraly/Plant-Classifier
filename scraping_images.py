from bs4 import BeautifulSoup
import requests
from PIL import Image
from io import BytesIO
import re

## Setting up the Bing API     
from azure.cognitiveservices.search.imagesearch import ImageSearchAPI
from msrest.authentication import CognitiveServicesCredentials

subscription_key = "0c9c1cb7f74342dcbc921c5cee0843a6"
search_term = "aloe vera houseplant"
client = ImageSearchAPI(CognitiveServicesCredentials(subscription_key))


## Defining a function that will pull the images from 
## search_term into a list of ImageObject
def pull_images(search_term, num_of_images):
    num = int(num_of_images/150)
    images = []
    for i in range(0, num+1):
        img = client.images.search(query=search_term, count=150, offset=(150*i))
        images += img.value
    return images

## Pulling images
images = pull_images(search_term, 3000)

## Downloading images
for i in range(0, len(links)):
    try:
        imgobj = requests.get(link[i].content_url)
        title = 'aloevera'+str(i+1)
        img = Image.open(BytesIO(imgobj.content))
        img.save('./scraped/aloe/'+title, img.format)
    except IOError:
        pass