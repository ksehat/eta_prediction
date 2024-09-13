import requests

url = 'http://127.0.0.1:8080/predict'

payload = {
    "features": {
        "city_id": 'C',
        "accept_event_timestamp": "2024-09-10T12:34:56Z",
        "origin_lat": 35.6892,
        "origin_lon": 51.3890,
        "destination_lat": 36.292,
        "destination_lon": 52.3890,
        "edd": 12000,
        "provider_A": 3600,
        "provider_B": 3700,
        "provider_C": 3400,
        "provider_D": 3300
    }
}

try:
    # Send a POST request to the API
    response = requests.post(url, json=payload)

    # Check if the request was successful
    response.raise_for_status()

    # Print the response from the API
    print("Response from API:", response.json())

except requests.exceptions.RequestException as e:
    # Print any errors that occur
    print("An error occurred:", e)
