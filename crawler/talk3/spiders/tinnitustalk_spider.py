# -*- coding: utf-8 -*-

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import Selector

from lxml import html, etree
import re
from datetime import datetime

from talk3.items import PostItem

USEFUL_SUBFORUMS = {
    "support.2",
    "collaboration-space.109",
    "research-news.4",
    "treatments.13",
    "alternative-treatments-and-research.27",
    "introduce-yourself.11",
    "support.52",
    "support.55",
    "success-stories.47"
}


class PreRestrictedLinkExtractor(LinkExtractor):
    def extract_links(self, response):
        subforum = response.css("#pageDescription a").xpath('@href').extract_first()
        if subforum and subforum[7:-1] not in USEFUL_SUBFORUMS:
            return []
        return super(PreRestrictedLinkExtractor, self).extract_links(response)


class TinnitusTalkSpider(CrawlSpider):

    name = "tinnitustalk"
    allowed_domains = [
        "tinnitustalk.com"
    ]
    start_urls = ["https://www.tinnitustalk.com/forums/" + sf for sf in USEFUL_SUBFORUMS]

    rules = [
        Rule(LinkExtractor(allow=[
            r"\/forums\/.+\.\d+\/page\-\d+$",
        ])),
        Rule(PreRestrictedLinkExtractor(allow=[
            r"\/threads\/.+\.\d+\/(?:page\-\d+)?$",
        ]), callback="extract_posts", follow=True)
    ]

    def clean_text(self, text):
        html_elem = html.fragment_fromstring(text)
        # remove the quoted text
        etree.strip_elements(html_elem, "div", with_tail=False)
        string = html.tostring(html_elem, method="text", encoding="unicode").strip()
        # merge multiple newlines to one
        string = re.sub('\n\n+', '\n', string)
        # remove the tab character
        return string.replace('\t', '')

    def extract_posts(self, response):

        sel = Selector(response)
        # get the unique id of the subforum
        try:
            subforum = sel.css("#pageDescription a").xpath('@href').extract_first()[7:-1]
        except TypeError:
            return

        if subforum not in USEFUL_SUBFORUMS:
            return

        # get unique id of the thread from the URL
        thread_id = re.search(
            "\/threads\/(.+\.\d+)\/(?:page\-\d+)?$", response.url
        ).group(1)

        posts = sel.css(".messageList .message")
        for post in posts:
            p = PostItem()
            p["subforum"] = subforum
            p["post_id"] = post.xpath('@id').extract_first()
            p["text"] = self.clean_text(post.css(".messageText").extract_first())
            timestring = post.css('.DateTime').xpath('@title').extract_first()
            timestring = timestring or post.css('.DateTime ::text').extract_first()
            timestring = datetime.strptime(timestring, "%b %d, %Y at %I:%M %p")
            p["timestamp"] = timestring.isoformat()
            author = post.css('.username').xpath('@href').extract_first()
            p["author_id"] = author[8:-1] if author else "GUEST-" + p["post_id"]
            p["url"] = post.css(".publicControls a").xpath('@href').extract_first()
            p["thread_id"] = thread_id
            p["position_in_thread"] = post.css(".publicControls a::text").extract_first()[1:]
            p["agrees"] = post.css(".likesSummary div ul").xpath(
                "li[img[@title='Agree']]/strong/text()").extract_first() or '0'
            yield p
