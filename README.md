# TALK-3: TinnitusTalk.com classification and visualisation

## installation
* check out repository
* install Python dependencies for crawler, topic assignment, classification and aggregation-api: `pip3 install -r requirements.txt`

## crawler
* crawls https://tinnitustalk.com, stores the result in `data/talk3_posts.csv`
* do only run if really necessary, better get the existing file
* to run:
```bash
./crawler$ scrapy crawl tinnitustalk -L INFO
```

## topic assignment
* takes the crawled posts and a definiton file (e.g. committed in `data/treatment_definitons`),
outputs all posts that a treatment was detected in with the treatment name in an additional column
* requires a model for sentence splitter: `python3 -m nltk.downloader punkt`
* example call:
```bash
< data/talk3_posts.csv python3 topic_assignment/detect_treatment.py -d data/treatment_definitons.txt > data/treatment_detected.csv
```

## classifier
* trains a model to classify sentences by sentiment polaritiy and personal experience
* requires lexicon for "vader" feature: `python3 -m nltk.downloader vader_lexicon`
* requires corpora for "textblob" features: `python3 -m textblob.download_corpora`
* expects the extracted sentences as `data/treatment_detected.csv`
* stores the classified sentences as `data/sentences_classified.csv`
* to run:
```bash
./classifier$ python3 RandomForestTool.py
```

## aggregation-api
* aggregates the data from classifier and topic assigment, makes it accessible via a JSON API.
* you have to create a config-file named `config.cfg` in the folder. A template is provided as `config-template.cfg`.
* to run:
```bash
./aggregation-api$ python3 app.py
```

## visualisation-app
* a web-application visualising the data served by the API
* requires aggregation-api to be running, its location can be configured with the `api_base_url` config value in `aurelia_project/environments`
* to install dependencies, first [install Aurelia](https://aurelia.io/hub.html#/doc/article/aurelia/framework/latest/the-aurelia-cli/1), then run `npm install` from the `visualisation-app` directory
* to run, either just open `index.html` or run `au run --watch` for starting a small html server
* to build for production, run `au build --env prod`, then copy `index.html` and `scripts/` to the production machine
