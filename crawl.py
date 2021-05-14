import json
import random
from queue import Queue
from urllib import request

from bs4 import BeautifulSoup
from tqdm import tqdm


class MovieCrawler:
    """
    Crawler for IMDB movie home page
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    base = "https://www.imdb.com/title/"

    def __init__(self):
        self.visited = set()
        self.crawled = []
        self.result = []

    def output(self, data_file="movies.json", visited_file="visited.json"):
        with open(data_file, 'w') as outfile:
            json.dump(self.result, outfile, indent=2)
        with open(visited_file, 'w') as outfile:
            json.dump(self.crawled, outfile, indent=2)

    def crawl(self, root, max_links=100, genre="Action"):
        queue = Queue()
        queue.put(root)
        count = 0
        while not queue.empty():
            if count >= max_links:
                break
            link = queue.get()
            if link in self.visited:
                continue
            if count % 100 == 0:
                print(f"Crawled {count} pages.")

            try:
                page, links = self.extract_info(link, genre)
                count += 1
                self.crawled.append(self.base + link)
                self.result.append(page)
                self.visited.add(link)
                for x in links:
                    queue.put(x)
            except Exception as _:
                self.visited.add(link)

    def extract_info(self, link, genre="Action"):
        data = {"id": link}
        url = self.base + link
        data["url"] = url
        req = request.Request(url, headers=self.headers)
        soup = BeautifulSoup(request.urlopen(req), 'html.parser')

        subtext = soup.find("div", {'class': 'subtext'})
        genres = [i.text.strip() for i in subtext.contents
                  if len(i.string.strip()) > 3 and i.string.strip().isalpha()]
        if genre not in genres:
            raise Exception("Skipping because different genre")

        title = soup.find("div", {'class': 'titleBar'}).find("h1")
        data["name"] = title.contents[0].replace(u'\xa0', u'')

        image = soup.find("div", {'class': 'poster'}).find("img")
        data["image"] = image['src']

        data["genre"] = genres

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

        rating = soup.find("span", {"itemprop": "ratingValue"})
        data["rating"] = float(rating.string)

        num_rating = soup.find("span", {"itemprop": "ratingCount"})
        data["ratingCount"] = int(num_rating.string.replace(',', ''))

        return data, recs


class ReviewCrawler:
    """
    Crawler for IMDB movie reviews
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    base = "https://www.imdb.com/title/"

    def __init__(self):
        self.result = []
        self.count = 0

    def output(self, data_file="reviews-large.json"):
        with open(data_file, 'w') as outfile:
            json.dump(self.result, outfile, indent=2)

    def crawl(self, visited="visited.json", ratio=1.0):
        visited_links = json.load(open(visited))
        k = int(len(visited_links) * ratio)
        for link in tqdm(random.sample(visited_links, k)):
            try:
                self.extract_info(link)
            except Exception:
                self.count += 1
        print(self.count)

    def extract_info(self, link):
        for i in [1, 2, 9, 10]:
            label = "negative" if i < 5 else "positive"
            url = link + "/reviews?spoiler=hide&sort=helpfulnessScore&dir=desc&ratingFilter=" + str(i)
            req = request.Request(url, headers=self.headers)
            soup = BeautifulSoup(request.urlopen(req), 'html.parser')
            review_titles = soup.find_all("a", {"class": "title"})
            review_contexts = soup.find_all("div", {"class": "text show-more__control"})
            for j in range(len(review_titles)):
                self.result.append({
                    "title": review_titles[j].text.strip(),
                    "label": label,
                    "rating": i,
                    "text": review_contexts[j].text.strip()
                })


def main():
    # Crawl movie data
    crawler = MovieCrawler()
    crawler.crawl("tt1345836", 1000, "Action")
    crawler.crawl("tt2674426", 1000, "Romance")
    crawler.crawl("tt0109686", 1000, "Comedy")
    crawler.output()

    # Crawl review data
    crawler_review = ReviewCrawler()
    crawler_review.crawl("visited.json", ratio=0.1)
    crawler_review.output()


if __name__ == '__main__':
    main()
