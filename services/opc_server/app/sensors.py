import requests
import random
import time
import logging
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
        
class RPMSensor(object):

    def __init__(self,  base_url: str):
        self.name = "rpmSensor"
        self.base_url = base_url
        self.sources = [
            {
                "name": "TCRT5000 Infrared Reflection Tracking Sensor",
                "datatype": ua.Variant(0.0, ua.VariantType.Double),
                "historize": True,
                "historize_length": 1000,
                "func": self.read_value,
                "writable": True
            },
        ]

    def read_value(self):
        """
        Fetches the temperature of the upper segment from the `/rpm` endpoint.

        Args:
            base_url (str): The base URL of the API including the endpoint path. Defaults to `http://localhost:8000/rpm`.

        Returns:
            float: The rpm read from Infrared sensor.

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
            rpm = float(response.text.replace('"',''))  # Assuming the response is a number in string format
            return rpm

        except requests.ConnectionError:
            raise ConnectionError("Failed to connect to the server. Please check the API endpoint or your network.")

        except requests.Timeout:
            raise ConnectionError("The request to the server timed out. Please try again later.")

        except requests.exceptions.HTTPError as http_err:
            raise requests.exceptions.HTTPError(f"HTTP error occurred: {http_err}")

        except ValueError:
            raise ValueError("The response from the server is invalid and could not be converted to a float.")

class AudioSensorMfcc(object):

    def __init__(self, base_url: str):
        self.name = "AudioSensor MFCC"
        self.base_url = base_url
        self.cache = None
        self.last_update = 0
        self.update_interval = 1.0  # Update interval in seconds
        self.key_functions = {}  # Maps keys to their corresponding functions
        self.sources = []  # Initialize empty sources

        # Get initial data and set up sources
        self.update_cache()
        self.initialize_sources()

    def initialize_sources(self):
        """
        Initialize or update the sources based on the current cache
        """
        if not self.cache:
            return

        # Create new sources list
        new_sources = []
        
        # Set of keys we've already processed
        existing_keys = set(self.key_functions.keys())
        current_keys = set(self.cache.keys())
        
        # Add new keys
        for key in current_keys:
            if key not in existing_keys:
                # Create a closure to capture the specific key for each source
                def get_value_for_key(key=key):
                    return self.read_value(key)
                
                # Store the function so we can reuse it
                self.key_functions[key] = get_value_for_key
                
            # Add the source definition
            new_sources.append({
                "name": key,
                "datatype": ua.Variant(0.0, ua.VariantType.Double),
                "historize": True,
                "historize_length": 1000,
                "func": self.key_functions[key],
                "writable": True
            })
            
        # Remove functions for keys that no longer exist
        for key in existing_keys - current_keys:
            del self.key_functions[key]
            
        # Only update if this isn't the initial setup
        if self.sources:
            logging.info(f"MFCC sources updated, added: {current_keys - existing_keys}, removed: {existing_keys - current_keys}")
            
        self.sources = new_sources

    def update_cache(self):
        """
        Updates the cached data from the API endpoint
        """
        try:
            # Sending the GET request to the specified endpoint
            response = requests.get(self.base_url, timeout=10)
            response.raise_for_status()
            
            # Store the old keys to detect changes
            old_keys = set(self.cache.keys()) if self.cache else set()
            
            # Update the cache
            self.cache = response.json()
            self.last_update = time.time()
            
            # Check if the set of keys has changed
            new_keys = set(self.cache.keys())
            if old_keys != new_keys:
                self.initialize_sources()
            
        except Exception as e:
            # If update fails, keep the old cache but log the error
            if self.cache is None:  # Only raise if we don't have any data yet
                raise
            logging.error(f"Failed to update MFCC data: {str(e)}")

    def read_value(self, key):
        """
        Returns a specific value from the cached audio data.
        
        Args:
            key (str): The specific frequency key to retrieve
            
        Returns:
            float: The value for the specified key
        """
        # Check if we need to update the cache
        if time.time() - self.last_update > self.update_interval:
            self.update_cache()
            
        # Return the specific value for the requested key
        if self.cache and key in self.cache:
            return float(self.cache[key])
        
        # If key is not found, return 0.0 and log error
        logging.error(f"Key {key} not found in MFCC data")
        return 0.0

class AudioSensorSpectral(object):

    def __init__(self, base_url: str):
        self.name = "AudioSensor Spectral"
        self.base_url = base_url
        self.cache = None
        self.last_update = 0
        self.update_interval = 1.0  # Update interval in seconds
        self.key_functions = {}  # Maps keys to their corresponding functions
        self.sources = []  # Initialize empty sources

        # Get initial data and set up sources
        self.update_cache()
        self.initialize_sources()

    def initialize_sources(self):
        """
        Initialize or update the sources based on the current cache
        """
        if not self.cache:
            return

        # Create new sources list
        new_sources = []
        
        # Set of keys we've already processed
        existing_keys = set(self.key_functions.keys())
        current_keys = set(self.cache.keys())
        
        # Add new keys
        for key in current_keys:
            if key not in existing_keys:
                # Create a closure to capture the specific key for each source
                def get_value_for_key(key=key):
                    return self.read_value(key)
                
                # Store the function so we can reuse it
                self.key_functions[key] = get_value_for_key
                
            # Add the source definition
            new_sources.append({
                "name": key,
                "datatype": ua.Variant(0.0, ua.VariantType.Double),
                "historize": True,
                "historize_length": 1000,
                "func": self.key_functions[key],
                "writable": True
            })
            
        # Remove functions for keys that no longer exist
        for key in existing_keys - current_keys:
            del self.key_functions[key]
            
        # Only update if this isn't the initial setup
        if self.sources:
            logging.info(f"Spectral sources updated, added: {current_keys - existing_keys}, removed: {existing_keys - current_keys}")
            
        self.sources = new_sources

    def update_cache(self):
        """
        Updates the cached data from the API endpoint
        """
        try:
            # Sending the GET request to the specified endpoint
            response = requests.get(self.base_url, timeout=10)
            response.raise_for_status()
            
            # Store the old keys to detect changes
            old_keys = set(self.cache.keys()) if self.cache else set()
            
            # Update the cache
            self.cache = response.json()
            self.last_update = time.time()
            
            # Check if the set of keys has changed
            new_keys = set(self.cache.keys())
            if old_keys != new_keys:
                self.initialize_sources()
            
        except Exception as e:
            # If update fails, keep the old cache but log the error
            if self.cache is None:  # Only raise if we don't have any data yet
                raise
            logging.error(f"Failed to update spectral data: {str(e)}")

    def read_value(self, key):
        """
        Returns a specific value from the cached audio data.
        
        Args:
            key (str): The specific frequency key to retrieve
            
        Returns:
            float: The value for the specified key
        """
        # Check if we need to update the cache
        if time.time() - self.last_update > self.update_interval:
            self.update_cache()
            
        # Return the specific value for the requested key
        if self.cache and key in self.cache:
            return float(self.cache[key])
        
        # If key is not found, return 0.0 and log error
        logging.error(f"Key {key} not found in spectral data")
        return 0.0