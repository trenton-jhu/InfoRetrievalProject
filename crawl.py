import json
from queue import Queue

from urllib import parse, request
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0'}
base = "https://www.imdb.com/title/"


def extract_info(url):
    data = {}
    req = request.Request(url, headers=headers)
    soup = BeautifulSoup(request.urlopen(req), 'html.parser')

    rating = soup.find("span", {"itemprop": "ratingValue"})
    data["rating"] = float(rating.string)

    num_rating = soup.find("span", {"itemprop": "ratingCount"})
    data["ratingCount"] = int(num_rating.string.replace(',', ''))

    title = soup.find("div", {'class': 'titleBar'}).find("h1")
    data["name"] = title.contents[0].replace(u'\xa0', u'')

    genre = soup.find("div", {'class': 'subtext'})
    data["genre"] = [i.text.strip() for i in genre.contents
                     if len(i.string.strip()) > 3 and i.string.strip().isalpha()][0]

    summary = soup.find("div", {'class': 'summary_text'})
    data["summary"] = summary.text.strip()

    credit = soup.find_all("div", {'class': 'credit_summary_item'})[0]
    data["director"] = [x.text for x in credit.find_all("a")]

    storyline = soup.find("div", {'id': 'titleStoryLine', 'class': 'article'})
    data['storyline'] = storyline.find_all("span")[1].text.strip()

    cast_list = soup.find("table", {'class': 'cast_list'}).find_all("a")
    data['cast'] = [x.text.strip() for x in cast_list if x.text and x['href'].startswith("/name/")]

    hrefs = [x.find("a") for x in soup.find_all("div", {'class': 'rec_item'})]
    recs = [x['href'].split("/")[2] for x in hrefs]

    runtime = soup.find("time")['datetime']
    minutes = ''.join([i for i in runtime if i.isdigit()])
    data['runtime'] = int(minutes)

    return data, recs


def crawl(root, max_links=100):
    queue = Queue()
    queue.put(root)
    visited = set()
    data = []

    while not queue.empty():
        if len(visited) >= max_links:
            break
        link = queue.get()
        if link in visited:
            continue
        else:
            visited.add(link)
        if len(visited) % 10 == 0:
            print(f"Crawled {len(visited)} pages.")

        try:
            page, links = extract_info(base + link)
            data.append(page)
            for x in links:
                queue.put(x)
        except Exception as e:
            print(e, base + link)

    urls = [base + x for x in visited]
    return data, urls


if __name__ == '__main__':
    data, visited = crawl("tt1345836", 1)
    with open('movies.json', 'w') as outfile:
        json.dump(data, outfile, indent=2)
    with open('visited.json', 'w') as outfile:
        json.dump(visited, outfile, indent=2)
