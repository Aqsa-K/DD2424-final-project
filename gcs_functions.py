import os
import json
# from google.colab import files
# uploaded = files.upload()

# Assuming the filename of your key is 'service-account-file.json'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'xenon-lantern-421513-f44447d18aae.json'

from google.cloud import storage

# This creates a client that uses the specified service account credentials
client = storage.Client()
# Now you can interact with Google Cloud Storage using this client

# Lists all the buckets
# buckets = list(client.list_buckets())
# print(buckets)


def read_from_storage(bucket_name, file_name):
    """Reads content from a file in Google Cloud Storage.

    Args:
        bucket_name (str): The name of the GCS bucket.
        file_name (str): The name of the file to read from the bucket.

    Returns:
        str: The content of the file.
    """
    # Initialize the Google Cloud Storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(file_name)

    # Download the content as a byte string
    content = blob.download_as_bytes()

    # Convert to string if necessary (assuming the file is text-based)
    return content.decode('utf-8')

# Example usage:
# content = read_from_storage('experimentresults', 'new-file.txt')
# print(content)

def write_to_storage(bucket_name, file_name, data):
    """Writes data to a file in Google Cloud Storage.

    Args:
        bucket_name (str): The name of the GCS bucket.
        file_name (str): The name of the file to write in the bucket.
        data (str): The data to write to the file.

    Returns:
        None
    """
    # Initialize the Google Cloud Storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(file_name)

    # Upload the data
    blob.upload_from_string(data)


# Example usage:
# write_to_storage('experimentresults', 'new-file-1.txt', 'Hello, World!')


def create_and_write_file(file_name, text_string):
    """
    Creates a file and writes a specified text string to it.

    Args:
        file_name (str): The name of the file to create.
        text_string (str): The text string to write to the file.

    Returns:
        None
    """
    # Open the file in write mode ('w'). If the file doesn't exist, it will be created.
    # If the file exists, it will be overwritten.
    with open(file_name, 'w') as file:
        file.write(text_string)

# Example usage:
# create_and_write_file('example.txt', 'Hello, this is a sample text.')


def write_json_to_gcs(bucket_name, destination_blob_name, data):
    """
    Writes JSON data to a file in Google Cloud Storage.

    Args:
        bucket_name (str): The name of the GCS bucket.
        destination_blob_name (str): The path within the bucket to save the file.
        data (str): JSON string to write to the file.
    """
    # Create a temporary file
    temp_file = "/tmp/tempfile.json"
    with open(temp_file, "w") as file:
        file.write(data)

    # Initialize the Google Cloud Storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Upload the file
    blob.upload_from_filename(temp_file)

    # Optionally, remove the temporary file if not needed
    os.remove(temp_file)


def read_json_from_gcs(bucket_name, source_blob_name):
    """
    Reads a JSON file from Google Cloud Storage and parses the JSON content.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_blob_name (str): The name of the blob (file) in the GCS bucket.

    Returns:
        dict: The parsed JSON data as a dictionary.
    """
    # Initialize the Google Cloud Storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(source_blob_name)

    # Download the content as a string
    json_data = blob.download_as_string()

    # Parse the JSON string into a Python dictionary
    data = json.loads(json_data)

    return data


# import json
#
# data = {
#     "name": "Example",
#     "age": 30,
#     "city": "New York"
# }
#
# # Convert the data to JSON format
# json_data = json.dumps(data)
#
# # Example usage:
# write_json_to_gcs('experimentresults', 'examplefile.json', json_data)
#
# # Example usage:
# data = read_json_from_gcs('experimentresults', 'examplefile.json')
# print(data)

