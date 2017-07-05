# talk3

## additional installation steps
* get model for sentence splitter: `python3 -m nltk.downloader punkt`
* get lexicon for "vader" feature: `python3 -m nltk.downloader vader_lexicon`

## crawler
* crawls https://tinnitustalk.com, stores the result in `data/talk3_posts.csv`
* do only run if really necessary, better get the existing file
* to run:
```bash
~/talk3/crawler$ scrapy crawl tinnitustalk -L INFO
```

## topic assignment
* takes the crawled posts and a definiton file (e.g. committed in `data/treatment_definitons`),
outputs all posts that a treatment was detected in with the treatment name in an additional column
* example call:
```bash
< data/talk3_posts.csv python3 topic_assignment/detect_treatment.py -d data/treatment_definitons.txt > data/treatment_detected.csv
```
