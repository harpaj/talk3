# -*- coding: utf-8 -*-

from scrapy import Item, Field


class PostItem(Item):
    subforum = Field()
    post_id = Field()
    text = Field()
    timestamp = Field()
    author_id = Field()
    url = Field()
    thread_id = Field()
    thread_name = Field()
    position_in_thread = Field()
    agrees = Field()
