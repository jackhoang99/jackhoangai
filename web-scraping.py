import requests
from bs4 import BeautifulSoup

# List of URLs to scrape
urls = [
    "https://trahoang.com/",
    "https://trahoang.com/about",
    "https://trahoang.com/treatment-focus",
    "https://trahoang.com/faqs",
    "https://trahoang.com/contact",
]

for url in urls:
    # Send a GET request to the website
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the webpage
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract all paragraph text from the webpage
        paragraphs = soup.find_all("p")
        paragraph_text = "\n\n".join([p.get_text() for p in paragraphs])

        # Print the paragraph text content
        print(f"Content from {url}:\n")
        print(paragraph_text)
        print("\n" + "=" * 80 + "\n")
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
