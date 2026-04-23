import os
from dotenv import load_dotenv
from apify_client import ApifyClient
import sys

load_dotenv()

token = os.getenv("APIFY_API_TOKEN")
print(f"Token found: {token[:10]}...{token[-5:]}" if token else "Token NOT found")

client = ApifyClient(token)
try:
    print("Attempting to get user info from Apify...")
    user = client.user().get()
    print(f"Success! Username: {user.get('username')}")
except Exception as e:
    print(f"Apify Error: {e}")

try:
    import pandas
    print("Pandas is installed.")
except ImportError:
    print("Pandas is NOT installed.")
