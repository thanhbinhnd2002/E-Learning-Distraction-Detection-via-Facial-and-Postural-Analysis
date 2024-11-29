
import requests
from bs4 import BeautifulSoup

# initialize the list of discovered urls
# with the first page to visit
urls = ["https://app.studystream.live/focus/room"]

# until all pages have been visited
while len(urls) != 0:
    # get the page to visit from the list
    current_url = urls.pop()

    # crawling logic
    response = requests.get(current_url)
    soup = BeautifulSoup(response.content, "html.parser")

    link_elements = soup.select("a[href]")
    # videos = soup.find_all("video")
    for link_element in link_elements:
        url = link_element['href']
        if "https://www.scrapingcourse.com/ecommerce/" in url:
            urls.append(url)
