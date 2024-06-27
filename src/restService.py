import requests


class RestService:

    # Constructor to use http rest service to communicate with the api
    def __init__(self, url):
        self.base_url = 'http://localhost:5555/api/'
        self.url = self.base_url + url
        self.headers = {
            'Content-Type': 'application/json',
        }

    # Method to send the message to the api
    def post(self, message):
        response = requests.post(self.url, headers=self.headers, json=message)
        return response.json()

    # Method to get the message from the api
    def get(self):
        response = requests.get(self.url, headers=self.headers)
        return response.json()
