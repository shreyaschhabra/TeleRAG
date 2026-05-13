import os
import requests

def test_key(key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    response = requests.get(url)
    if response.status_code == 200:
        print(f"VALID KEY FOUND: {key}")
        return True
    else:
        print(f"Invalid: {key} (Status: {response.status_code})")
        return False

keys_to_test = [
    # Original transcription
    "AIzaSyD2hfPzz_nthwOoBFDRvUZVXZUqgSzrBus",
    
    # Try zero instead of O
    "AIzaSyD2hfPzz_nthw0oBFDRvUZVXZUqgSzrBus",
    "AIzaSyD2hfPzz_nthwO0BFDRvUZVXZUqgSzrBus",
    "AIzaSyD2hfPzz_nthw00BFDRvUZVXZUqgSzrBus",

    # Try lowercase L just in case
    "AlzaSyD2hfPzz_nthwOoBFDRvUZVXZUqgSzrBus",
    "AlzaSyD2hfPzz_nthw0oBFDRvUZVXZUqgSzrBus",

    # Try hyphen instead of underscore
    "AIzaSyD2hfPzz-nthwOoBFDRvUZVXZUqgSzrBus",
    "AIzaSyD2hfPzz-nthw0oBFDRvUZVXZUqgSzrBus",
    
    # Check for lowercase L vs capital I in the rest of the string
    "AIzaSyD2hfPzz_nthwOoBFDRvUZVXZUqgSzrBus",
    
    # Let's try maybe 'v' is 'V' or 'U' is 'u'?
    "AIzaSyD2hfPzz_nthwOoBFDRvUZVxZUqgSzrBus"
]

for k in keys_to_test:
    if test_key(k):
        break
