import requests

def verify_website(url):
    """
    Verifies if a website exists by sending an HTTP GET request.
    Returns (True, status_code) if the status code is < 400.
    Returns (False, error_message) otherwise.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        # Using a timeout to prevent hanging and a realistic user agent
        response = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
        if response.status_code < 400:
            return True, f"Status: {response.status_code}"
        else:
            return False, f"Status: {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Error: Timeout"
    except requests.exceptions.ConnectionError:
        return False, "Error: Connection Error"
    except requests.exceptions.RequestException as e:
        return False, f"Error: {str(e)}"

if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "http://www.imobiliariaalegro.com/",
        "https://thiswebsiteisdefinitelyfake12345.com",
        "https://www.bristolhotelva.com/"
    ]
    
    for url in test_urls:
        exists, details = verify_website(url)
        print(f"URL: {url} | Exists: {exists} | Details: {details}")


