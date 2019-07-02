#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import urllib.request
import codecs
import glob
import os

def get_html_from_link(link):
    try:
        with urllib.request.urlopen(link) as r:
            tmp_html = r.read()
            return tmp_html
    except:
        print('This url failed to be open', link)


def handling_encoding(html_file):
    return codecs.decode(html_file, 'utf-8')


def write_words_in_file(betterfile, soup):
    title = soup.title.string.split(" ")
    with codecs.open(title[2]+".txt", "a", "utf-8") as stream:
        text = " "
        for lis in soup.find_all('li'):
            for ahrefs in lis.find_all('a'):
                link = ahrefs.get('title')
                if link is not None:
                    text = text + '\n' + link
        stream.write(text)


def extract_lists_from_links(link):
    html = handling_encoding(get_html_from_link(link))
    soup = BeautifulSoup(html, 'html.parser')
    write_words_in_file(html,soup)


def make_txts_into_lists():
    tmp = glob.glob("*.txt")
    set_of_words = set()
    for file in tmp:
        with codecs.open(file, 'r', 'utf-8') as f:
            for line in f:
                if " " in line:
                    pass
                else:
                    set_of_words.add(line)
    return list(set_of_words)

def clean_the_csvs(csvfile):
