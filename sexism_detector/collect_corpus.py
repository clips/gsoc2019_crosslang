#!/usr/bin/python
import requests
import csv
from codes import vk_api_vers,service_token
from helper_functions import get_html_from_link,handling_encoding
from bs4 import BeautifulSoup

# this is made from vk via vk API
# Important! For this function to work, you have to first visit the file codes and input your service token and api version

def make_corpus(name,community, query_list, service_token,vk_api_vers):
    with open(name, 'a', newline='', encoding="utf-8") as csvfile:
        headers = ['Post_id','Comment_id','Label','Text']
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for query in query_list:
            # Step one: we get the list of id's of topics which have to do something with the US
            url = 'https://api.vk.com/method/wall.search?owner_id={}&own=1&owners_only=1&query=\'{}\'&access_token={}&v={}'.format(community,query,service_token,vk_api_vers)
            topic_ids = requests.get(url)
            for post_id in topic_ids.json()["response"]['items']:
                new_url = 'https://api.vk.com/method/wall.getComments?owner_id={}&post_id={}&need_likes=1&count=100&sort=asc&preview_length=0&access_token={}&v={}'.format(community,post_id['id'],service_token,vk_api_vers)
                comments = requests.get(new_url)
                for cur_comment in comments.json()["response"]['items']:
                    try:
                        writer.writerow({'Post_id':post_id['id'], 'Comment_id': cur_comment['id'],'Label': 'sexist', 'Text': cur_comment['text']})
                    except:
                        pass


# Translation: "sexism", "meToo", "sexual harassment", "decriminalization of domestic violence', 'rape', 'feminism', 'Shurigina', 'harassment'
query_list= {'сексизм',"meToo",'сексуальные домогательства', 'декриминализация побоев','изнасилование','феминизм','Шурыгина','домогательства'}

# This function is forum specific, oriented for "Anti-female" russian forum
# To use the function: follow the link to the forum https://antiwomen.ru/index.php and plug in links to any topic you want to scrape

def make_corpus_ant_forum(name,link_to_topic):
    with open(name, 'a', newline='', encoding="utf-8") as csvfile:
        headers = ['Label', 'Text']
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        html = handling_encoding(get_html_from_link(link_to_topic))
        soup = BeautifulSoup(html, 'html.parser')
        number_of_pages = soup.find_all("div", {"class": "pagination"})
        number_of_pages = str(number_of_pages)
        number_of_pages = number_of_pages.split(' ',2)[1]
        number_of_pages = number_of_pages.split('>')[1]
        x = 0
        for text in soup.find_all("div", {"class": "content"}):
            try:
                writer.writerow({'Label': 'sexist', 'Text': text.get_text()})
            except:
                pass
        while x < int(number_of_pages):
            x = x + 25
            html = handling_encoding(get_html_from_link(link_to_topic+'&start='+str(x)))
            soup = BeautifulSoup(html, 'html.parser')
            for text in soup.find_all("div", {"class": "content"}):
                try:
                    writer.writerow({'Label': 'sexist', 'Text': text.get_text()})
                except:
                    pass

# Similarly made specifically for another Russian-speaking forum. More information about it on the project github page.
# To use the function: follow the link to the forum https://holywarsoo.net/index.php and plug in links to any topic you want to scrape

def make_corpus_holisoo(name, link_to_topic,amount_pages):
    with open(name, 'w', newline='', encoding="utf-8") as csvfile:
        headers = ['Label', 'Text']
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        x = 2
        while x < amount_pages:
            html = handling_encoding(get_html_from_link(link_to_topic + '&p=' + str(x)))
            soup = BeautifulSoup(html, 'html.parser')
            for text in soup.find_all("div", {"class": "postmsg"}):
                try:
                    unwanted = text.find("div", {"class": "quotebox"})
                    unwanted.extract()
                    writer.writerow({'Label': 'non_sexist', 'Text': text.getText()})
                except:
                    pass
            x = x + 1














