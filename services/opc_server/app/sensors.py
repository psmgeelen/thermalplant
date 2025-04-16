import requests
import random
from asyncua import ua

class RandomSensor(object):

    def __init__(self):
        self.name = "RandomSensor"
        self.sources = [
            {
                "name": "random",
                "datatype": ua.Variant(0.0, ua.VariantType.Double),
                "historize": True,
                "historize_length": 1000,
                "func": self.read_value,
                "writable": True

            },
        ]

    @staticmethod
    def read_value():
        return random.random()

class ThermalSensor(object):

    def __init__(self,  prefix:str, base_url: str):
        self.name = "thermalSensor" + prefix
        self.base_url = base_url
        self.sources = [
            {
                "name": "MAX6675 Temperature Sensor",
                "datatype": ua.Variant(0.0, ua.VariantType.Double),
                "historize": True,
                "historize_length": 1000,
                "func": self.read_value,
                "writable": True
            },
        ]

    def read_value(self):
        """
        Fetches the temperature of the upper segment from the `/temperature_upper` endpoint.

        Args:
            base_url (str): The base URL of the API including the endpoint path. Defaults to `http://localhost:8000/temperature_upper`.

        Returns:
            float: The temperature read from the upper segment if successful.

        Raises:
            ConnectionError: If the API is unreachable.
            requests.exceptions.HTTPError: If the API returns an HTTP error.
            ValueError: If the response is invalid or cannot be parsed correctly.
        """
        try:
            # Sending the GET request to the specified endpoint
            response = requests.get(self.base_url, timeout=10)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

            # Try to parse the response as a temperature value
            temp = float(response.text.replace('"',''))  # Assuming the response is a number in string format
            return temp

        except requests.ConnectionError:
            raise ConnectionError("Failed to connect to the server. Please check the API endpoint or your network.")

        except requests.Timeout:
            raise ConnectionError("The request to the server timed out. Please try again later.")

        except requests.exceptions.HTTPError as http_err:
            raise requests.exceptions.HTTPError(f"HTTP error occurred: {http_err}")

        except ValueError:
            raise ValueError("The response from the server is invalid and could not be converted to a float.")