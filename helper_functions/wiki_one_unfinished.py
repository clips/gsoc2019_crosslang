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
# # # табуированная лексика
# extract_lists_from_links('https://ru.wiktionary.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%A2%D0%B0%D0%B1%D1%83%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%BD%D0%B0%D1%8F_%D0%BB%D0%B5%D0%BA%D1%81%D0%B8%D0%BA%D0%B0/ru')
# # # матерные выражения.1
# extract_lists_from_links('https://ru.wiktionary.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%9C%D0%B0%D1%82%D0%B5%D1%80%D0%BD%D1%8B%D0%B5_%D0%B2%D1%8B%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F/ru')
# # # матерные выражения.2
# extract_lists_from_links('https://ru.wiktionary.org/w/index.php?title=%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%9C%D0%B0%D1%82%D0%B5%D1%80%D0%BD%D1%8B%D0%B5_%D0%B2%D1%8B%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F/ru&pagefrom=%D0%B7%D0%B0%D0%BF%D0%B8%D0%B7%D0%B4%D0%B5%D1%82%D1%8C%D1%81%D1%8F#mw-pages')
# # # матерныевыражения.3
# extract_lists_from_links('https://ru.wiktionary.org/w/index.php?title=%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%9C%D0%B0%D1%82%D0%B5%D1%80%D0%BD%D1%8B%D0%B5_%D0%B2%D1%8B%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F/ru&pagefrom=%D0%BF%D0%BE%D0%B4%D0%BE%D1%85%D1%83%D0%B5%D0%B2%D1%88%D0%B8%D0%B9#mw-pages')
# # # бранные выражения.1
# extract_lists_from_links('https://ru.wiktionary.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%91%D1%80%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5_%D0%B2%D1%8B%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F/ru')
# # # бранные выражения.2
# extract_lists_from_links('https://ru.wiktionary.org/w/index.php?title=%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%91%D1%80%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5_%D0%B2%D1%8B%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F/ru&pagefrom=%D0%BF%D1%80%D0%BE%D0%BA%D0%BB%D1%8F%D1%82%D1%8B%D0%B9%0A%D0%BF%D1%80%D0%BE%D0%BA%D0%BB%D1%8F%D1%82%D1%8B%D0%B9#mw-pages')
#

# # время
# extract_lists_from_links('https://ru.wiktionary.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%92%D1%80%D0%B5%D0%BC%D1%8F/ru')
# # год
# extract_lists_from_links('https://ru.wiktionary.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%93%D0%BE%D0%B4/ru')
# # сутки
# extract_lists_from_links('https://ru.wiktionary.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%A1%D1%83%D1%82%D0%BA%D0%B8/ru')
# # дни недели
# extract_lists_from_links('https://ru.wiktionary.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%94%D0%BD%D0%B8_%D0%BD%D0%B5%D0%B4%D0%B5%D0%BB%D0%B8/ru')
# # месяцы
# extract_lists_from_links('https://ru.wiktionary.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%9C%D0%B5%D1%81%D1%8F%D1%86%D1%8B/ru')
#
# #научн.1 -
# extract_lists_from_links('https://ru.wiktionary.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%9D%D0%B0%D1%83%D1%87%D0%BD%D1%8B%D0%B5_%D0%B2%D1%8B%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F/ru')
# # офиц. 1
# extract_lists_from_links('https://ru.wiktionary.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%9E%D1%84%D0%B8%D1%86%D0%B8%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5_%D0%B2%D1%8B%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F/ru')
# # офиц. 2
# extract_lists_from_links('https://ru.wiktionary.org/w/index.php?title=%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%9E%D1%84%D0%B8%D1%86%D0%B8%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5_%D0%B2%D1%8B%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F/ru&pagefrom=%D0%BF%D1%80%D0%B5%D0%B4%D1%83%D0%B2%D0%B5%D0%B4%D0%BE%D0%BC%D0%B8%D1%82%D1%8C%0A%D0%BF%D1%80%D0%B5%D0%B4%D1%83%D0%B2%D0%B5%D0%B4%D0%BE%D0%BC%D0%B8%D1%82%D1%8C#mw-pages')
list = make_txts_into_lists()
#print(list)
list.sort()
#print(list)
with codecs.open("profanity.txt", "a", "utf-8") as stream:
    try:
        for word in list:
            if word is not None:
                stream.write(word + ', \n')
    except:
        pass
