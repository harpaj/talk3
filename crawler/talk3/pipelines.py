# -*- coding: utf-8 -*-

from scrapy.exceptions import DropItem
import csv
import atexit


class Talk3Pipeline(object):
    def __init__(self):
        self.fh = open("talk3_posts.csv", "w")
        fields = [
            "subforum", "post_id", "text", "timestamp", "author_id", "url",
            "thread_id", "position_in_thread", "agrees"]
        self.writer = csv.DictWriter(self.fh, fields)
        self.writer.writeheader()
        self.seen_posts = set()
        atexit.register(self.close_handler)

    def process_item(self, item, spider):
        ditem = dict(item)
        post_id = ditem["post_id"]
        if post_id in self.seen_posts:
            raise DropItem("Duplicate post found: %s" % item)
        else:
            self.seen_posts.add(post_id)
            self.writer.writerow(ditem)
        return item

    def close_handler(self):
        self.fh.close()
