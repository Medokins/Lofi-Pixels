import requests
import json

def get_data(per_page=10, image_type="sunset"):
    with open('data/api_key.json') as file:
        api_keys = json.load(file)
    api_key = api_keys['api_key']
    url = 'https://api.flickr.com/services/rest/'

    params = {
        'method': 'flickr.photos.search',
        'api_key': api_key,
        'text': image_type,
        'sort': 'relevance',
        'per_page': per_page,
        'format': 'json',
        'nojsoncallback': 1
    }

    response = requests.get(url, params=params)
    data = response.json()

    for photo in data['photos']['photo']:
        photo_url = f"https://farm{photo['farm']}.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}.jpg"
        image_data = requests.get(photo_url).content
        filename = f"data/downloaded/{photo['id']}.jpg"
        with open(filename, 'wb') as file:
            file.write(image_data)
        print(f"Downloaded: {filename}")
        